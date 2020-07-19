/**
 * A hierarhical model for the IceCube constraints on
 * neutrino source populations.
 *
 * @author Francesca Capel
 * @date January 2020
 */

functions {
  
#include nu_pop.stan
  
}

data {

  /* Spectrum */
  real Phi_norm_det;
  real gamma_det;
  real Emin;
  real Emax;
  real Enorm;
  int Ns_det;

  /* Redshift range */
  real<lower=0> zmin;
  real<lower=zmin, upper=10> zmax;
 
  /* Detector */
  real sigma_Phi;
  real sigma_gamma;

  /* Detection probability */
  int Nd;
  int Ng;
  matrix[Nd, Ng] Pdet_cen;
  matrix[Nd, Ng] Pdet_width;
  vector[Nd+1] sindec_bins;
  vector[Ng] gamma_grid;

  /* Priors */
  real L_mean;
  real L_sd;
  real n0_mean;
  real n0_sd;

  real p1_sd;
  real p2_sd;
  real zc_sd;  
  real p1_mean;
  real p2_mean;
  real zc_mean;

  /* Options */
  int flux_only;
  int Ns_only;
  
}


transformed data {

  real x_r[1];
  int x_i[0];

  real Omega_obs = 2*pi() * (sindec_bins[Nd+1] - sindec_bins[1]);
  
  x_r[1] = zmax;
  
}

parameters {
  
  real<lower=1e-12, upper=1e-1> n0;
  real<lower=1e35, upper=1e50> L;

  real<lower=1.0, upper=4.0> gamma;

  real<lower=(p1_mean-6*p1_sd), upper=(p1_mean+6*p1_sd)> p1;
  real<lower=(p2_mean-6*p2_sd), upper=(p2_mean+6*p2_sd)> p2;
  real<lower=(zc_mean-6*zc_sd), upper=(zc_mean+6*zc_sd)> zc;
      
}

transformed parameters {
  
  real Phi_norm;
  real Ns_ex;
  real Ntot_ex;
  real w;
  vector[3] theta;

  theta[1] = p1;
  theta[2] = p2;
  theta[3] = zc;

  if (!Ns_only) {

    /* Calculate the differential flux normalisation at Emin at Earth [TeV^-1 cm^-2 s^-1 sr^-1] */
    Phi_norm = differential_flux_normalisation(n0, L, gamma, Emin, Emax, Enorm, theta, zmin, x_r, x_i);

    Ntot_ex = expected_total_sources(n0, theta, zmin, Omega_obs, x_r, x_i);
    w = poisson_lpmf(0 | Ntot_ex);
    //w = poisson_lcdf(100 | Ntot_ex);
    
  }

  if (!flux_only) {

    /* Calculate the expected number of sources in zth */
    Ns_ex = expected_sources(n0, theta, zmin, Omega_obs, L, Emin, Emax, Enorm, gamma,
			     sindec_bins, gamma_grid, Pdet_cen, Pdet_width, x_r, x_i);
    
  }
  
}

model {

  if (!flux_only) {

    target += poisson_lpmf(Ns_det | Ns_ex);

  }

  if (!Ns_only) {

    //Phi_norm_det ~ lognormal(log(Phi_norm), sigma_Phi);

    target += log_sum_exp(w + lognormal_lpdf(Phi_norm_det | log(1e-30), sigma_Phi),
    			  log1m(exp(w)) + lognormal_lpdf(Phi_norm_det | log(Phi_norm), sigma_Phi));

    gamma_det ~ normal(gamma, sigma_gamma);

  }

  /* priors */
  L ~ lognormal(L_mean, L_sd);
  n0 ~ lognormal(n0_mean, n0_sd);

  p1 ~ normal(p1_mean, p1_sd);
  p2 ~ normal(p2_mean, p2_sd);
  zc ~ normal(zc_mean, zc_sd);
  
}


