/**
 * Functions for the neutrino source population constraints model.
 */

#include interpolation.stan

real Mpc_to_m() {

  return 3.086E+22;

}

real Mpc_to_cm() {
  
  return 3.086E+24;
  
}

real m_to_cm() {

  return 100;

}

real yr_to_s() {

  return 3.154E+7;

}

/**
 * Hubble constant
 */
real H0() {

  return 70; // km s^-1 Mpc^-1

}

real Om() {
  
  return 0.3;
  
}

real Ol() {

  return 0.7;

}

/**
 * Hubble distance
 */
real DH() {

  return 3e5 / H0(); // Mpc

}

real E(real z) {

  real Omp = Om() * pow(1+z, 3);

  return sqrt(Omp + Ol());

}

real xx(real z) {

  return ( (1 - Om()) / Om() ) / pow(1+z, 3);

}

real phi(real x) {

  real x2 = pow(x, 2);
  real x3 = pow(x, 3);
  real numerator = 1.0 + (1.320 * x) + (0.4415 * x2) + (0.02656 * x3);
  real denominator = 1.0 + (1.392 * x) + (0.5121 * x2) + (0.03944 * x3);

  return numerator / denominator;
  
}

/**
 * Luminosity distance [Mpc]. 
 * Approximation from Adachi & Kasai 2012.
 */
real luminosity_distance(real z) {

  real x = xx(z);
  real zp = 1+z;

  real A = 2 * DH() * zp / sqrt(Om());
  real B = phi(xx(0)) - ((1 / sqrt(zp)) * phi(x));
  
  return A * B; // Mpc
    
}

/**
 * Comoving distance [Mpc].
 */
real comoving_distance(real z) {

  return luminosity_distance(z) / (1+z);

}

/**
 * Differential comoving volume [Mpc^3].
 */
real differential_comoving_volume(real z) {

  real dc = comoving_distance(z);

  return ( 4*pi() * DH() * pow(dc, 2) ) / E(z);

}

/**
 * Source evolution shape [dimensionless].
 * To be multiplied by n0 [Mpc^-3].
 */
real source_evolution(real z, real p1, real p2, real zc) {

  return pow(1+z, p1) / pow(1+(z/zc), p2);
  
}


/**
 * Integrand to get differential flux normalisation. 
 */
real[] differential_flux_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {

  real gamma;
  real p1;
  real p2;
  real zc;
  
  real dstatedz[1];
    
  p1 = params[1];
  p2 = params[2];
  zc = params[3];
  gamma = params[4];
  
  dstatedz[1] = (source_evolution(z, p1, p2, zc) * pow(1+z, -(gamma)) ) / E(z);

  return dstatedz;
}


/**
 * Point source flux normalisation for an unbounded power law.
 */
real point_source_flux_norm_upl(real L, real z, real Emin, real Enorm, real gamma) {

  real dl = luminosity_distance(z) * Mpc_to_cm();
  real A = L / (4*pi()*pow(dl, 2)); // TeV cm^-2 s^-1
  real B = ( (gamma-2) * pow(Enorm, -gamma) * pow(1+z, 2-gamma) ) / pow(Emin, 2-gamma);
						   
  return A * B;
}

/**
 * Factor to handle gamma=2 case elegantly for bounded power laws.
 * See Lipari 2006 equation 9.
 */
real k_gamma(real gamma, real Emin, real Emax) {

  real output;

  if (gamma == 2) {

    output = 1 / log(Emax/Emin);
    
  }
  else {

    output = (gamma-2) * pow(Emin, gamma-2) / (1 - pow(Emin/Emax, gamma-2));
    
  }

  return output;
}

/**
 * Calculate the point source flux normalisation
 * for the case of a bounded power law (Emin, Emax).
 */
real point_source_flux_norm(real L, real z, real Emin, real Emax, real Enorm, real gamma) {

  real dl = luminosity_distance(z) * Mpc_to_cm();
  real A = L / (4*pi()*pow(dl, 2)); // TeV cm^-2 s^-1
  real B = pow(Enorm, -gamma) * pow(1+z, 2-gamma) * k_gamma(gamma, Emin, Emax);
						   
  return A * B;
}


/**
 * P(det| phi) from point source search.
 */
real detection_probability(real phi_norm, real centre, real width) {

  return 1 / ( 1 + pow(width, -( log(phi_norm)-log(centre)) ) );

}

/**
 * Integrand to get expected number of sources detected
 * for the case of an unbounded power law. 
 */
real[] Ns_integrand_upl(real z, real[] state, real[] params, real[] x_r, int[] x_i) {

  real p1 = params[1];
  real p2 = params[2];
  real zc = params[3];
  real L = params[4];
  real Emin = params[5];
  real Enorm = params[6];
  real gamma = params[7];
  real centre = params[8];
  real width = params[9];
  real dstatedz[1];
  real phi_norm;
  
  /* Calculate point source flux normalisation [TeV^-1 cm^-2 s^-1] */
  phi_norm = point_source_flux_norm_upl(L, z, Emin, Enorm, gamma);
  
  dstatedz[1] =  (detection_probability(phi_norm, centre, width)
		  * source_evolution(z, p1, p2, zc) * differential_comoving_volume(z));
  
  return dstatedz;
}


/**
 * Integrand to get expected number of sources detected 
 * for the case of a bounded power law flux (Emin, Emax).
 */
real[] Ns_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {

  real p1 = params[1];
  real p2 = params[2];
  real zc = params[3];
  real L = params[4];
  real Emin = params[5];
  real Enorm = params[6];
  real gamma = params[7];
  real centre = params[8];
  real width = params[9];
  real Emax = params[10];
  real dstatedz[1];
  real phi_norm;
  
  /* Calculate point source flux normalisation [TeV^-1 cm^-2 s^-1] */
  phi_norm = point_source_flux_norm(L, z, Emin, Emax, Enorm, gamma);
  
  dstatedz[1] =  (detection_probability(phi_norm, centre, width)
		  * source_evolution(z, p1, p2, zc) * differential_comoving_volume(z));
  
  return dstatedz;
}


/**
 * Integrand to get expected total number of sources. 
 */
real[] Ntot_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {

  real p1 = params[1];
  real p2 = params[2];
  real zc = params[3];
  real dstatedz[1];
   
  dstatedz[1] =	source_evolution(z, p1, p2, zc) * differential_comoving_volume(z);
  
  return dstatedz;
}


/**
 * Differential flux normalisation Phi_norm [TeV^-1 cm^-2 s^-1 sr^-1]
 * for the case of an unbounded power law.
 */
real differential_flux_normalisation_upl(real n0, real L, real gamma, real Emin, real Enorm,
				     vector theta, real zmin, data real[] x_r, int[] x_i) {

  real params[4];
  real integration_result[1,1];
  real state0[1];

  params[1] = theta[1];
  params[2] = theta[2];
  params[3] = theta[3];
  params[4] = gamma;

  state0[1] = 0.0;
  
  integration_result = integrate_ode_rk45(differential_flux_integrand, state0, zmin, x_r, params, x_r, x_i);

  return (  ( DH() / (4*pi()) ) * (gamma-2) * (pow(Enorm, -gamma)/pow(Emin, 2-gamma))
	    * n0 * L * integration_result[1,1] * pow(Mpc_to_cm(), -2)  ); // [TeV^-1 cm^-2 s^-1]

}

/**
 * Differential flux normalisation, Phi_norm [TeV^-1 cm^-2 s^-1 sr^-1]
 * for the case of a bounded power law (Emin, Emax).
 */
real differential_flux_normalisation(real n0, real L, real gamma, real Emin, real Emax, real Enorm,
				     vector theta, real zmin, data real[] x_r, int[] x_i) {

  real params[4];
  real integration_result[1,1];
  real state0[1];

  params[1] = theta[1];
  params[2] = theta[2];
  params[3] = theta[3];
  params[4] = gamma;

  state0[1] = 0.0;
  
  integration_result = integrate_ode_rk45(differential_flux_integrand, state0, zmin, x_r, params, x_r, x_i);

  return (  ( DH() / (4*pi()) ) * pow(Enorm, -gamma) * k_gamma(gamma, Emin, Emax) 
	    * n0 * L * integration_result[1,1] * pow(Mpc_to_cm(), -2)  ); // [TeV^-1 cm^-2 s^-1]

}


/**
 * Expected detected number of sources
 * for the case of an unbounded power law.
 */
real expected_sources_upl(real n0, vector theta, real zmin, real Omega_obs,
		      real L, real Emin, real Enorm, real gamma,
		      vector sindec_bins, vector gamma_grid, matrix Pdet_cen, matrix Pdet_width, 
		      data real[] x_r, int[] x_i) {

  int Nd = num_elements(sindec_bins)-1;
  vector[Nd] Ns_ex;
  real params[9];
  real integration_result[1,1];
  real state0[1];
  real Omega_bin;
  
  params[1] = theta[1];
  params[2] = theta[2];
  params[3] = theta[3];
  params[4] = L;
  params[5] = Emin;
  params[6] = Enorm;
  params[7] = gamma;

  /* Sum over declination bins */
  for (i in 1:Nd) {

    params[8] = interpolate(gamma_grid, Pdet_cen[i]', gamma);
    params[9] = interpolate(gamma_grid, Pdet_width[i]', gamma);
    state0[1] = 0.0;

    Omega_bin = 2*pi() * (sindec_bins[i+1] - sindec_bins[i]);
    
    integration_result = integrate_ode_rk45(Ns_integrand_upl, state0, zmin, x_r, params, x_r, x_i);
    Ns_ex[i] = (Omega_bin / (4 *pi()) ) * n0 * integration_result[1,1];

  }
  
    return sum(Ns_ex);
}


/**
 * Expected detected number of sources
 * for the case of a bounded power law (Emin, Emax).
 */
real expected_sources(real n0, vector theta, real zmin, real Omega_obs,
			  real L, real Emin, real Emax, real Enorm, real gamma,
			  vector sindec_bins, vector gamma_grid, matrix Pdet_cen, matrix Pdet_width, 
			  data real[] x_r, int[] x_i) {

  int Nd = num_elements(sindec_bins)-1;
  vector[Nd] Ns_ex;
  real params[10];
  real integration_result[1,1];
  real state0[1];
  real Omega_bin;
  
  params[1] = theta[1];
  params[2] = theta[2];
  params[3] = theta[3];
  params[4] = L;
  params[5] = Emin;
  params[6] = Enorm;
  params[7] = gamma;
  params[10] = Emax;

  /* Sum over declination bins */
  for (i in 1:Nd) {

    params[8] = interpolate(gamma_grid, Pdet_cen[i]', gamma);
    params[9] = interpolate(gamma_grid, Pdet_width[i]', gamma);
    state0[1] = 0.0;

    Omega_bin = 2*pi() * (sindec_bins[i+1] - sindec_bins[i]);
    
    integration_result = integrate_ode_rk45(Ns_integrand, state0, zmin, x_r, params, x_r, x_i);
    Ns_ex[i] = (Omega_bin / (4 *pi()) ) * n0 * integration_result[1,1];

  }
  
    return sum(Ns_ex);
}

/**
 * Expected total number of sources in the Universe.
 */
real expected_total_sources(real n0, vector theta, real zmin, real Omega_obs,
			    data real[] x_r, int[] x_i) {

  real Ntot_ex;
  real params[3];
  real integration_result[1,1];
  real state0[1];
  
  params[1] = theta[1];
  params[2] = theta[2];
  params[3] = theta[3];

  state0[1] = 0.0;

  integration_result = integrate_ode_rk45(Ntot_integrand, state0, zmin, x_r, params, x_r, x_i);
  Ntot_ex = (Omega_obs / (4 *pi()) ) * n0 * integration_result[1,1];
  
  return Ntot_ex;
}
