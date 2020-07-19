#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import h5py
import time

"""
Script to run parallel simulations of sources.

Now with declination dependence. 
"""

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux
from icecube_tools.source.source_model import PointSource

from icecube_tools.simulator import Simulator


COMM = MPI.COMM_WORLD

if COMM.rank == 0:

    FILE_STEM = "/cfs/klemming/nobackup/c/capel/neutrino_model/"
    
    # Setup
    N = int(1e3)
    Emin = 5e2 # GeV
    Emax = 1e8 # GeV
    max_cosz = 0.1
    dec_to_sim = np.arcsin(np.linspace(-0.08, 0.97, 30))
    index_to_sim = np.array([1.8, 2.0, 2.19, 2.4, 2.6, 2.8]) #np.array([2.0, 2.19]) #np.linspace(1.8, 2.6, 10)   
    output_file = FILE_STEM + 'output/sim/source_1e3_6_30_new_Emin5e2.h5'

    # Effective area
    Aeff_filename = FILE_STEM + 'input/IC86-2012-TabulatedAeff.txt'
    effective_area = EffectiveArea(Aeff_filename)

    # Energy resolution
    eres_file = FILE_STEM + 'input/effective_area.h5'
    energy_res = EnergyResolution(eres_file)

    # Angular resolution
    Ares_file = FILE_STEM + 'input/IC86-2012-AngRes.txt'
    ang_res = AngularResolution(Ares_file)

    # Detector
    detector = IceCube(effective_area, energy_res, ang_res)

    # Shared runs
    N_per_core = int(N / COMM.size)

    start_time = time.time()

else:

    detector = None
    dec_to_sim = None
    index_to_sim = None
    N_per_core = None
    Emin = None
    Emax = None
    
detector = COMM.bcast(detector, root=0)
dec_to_sim = COMM.bcast(dec_to_sim, root=0)
index_to_sim = COMM.bcast(index_to_sim, root=0)
N_per_core = COMM.bcast(N_per_core, root=0)
Emin = COMM.bcast(Emin, root=0)
Emax = COMM.bcast(Emax, root=0)

# Parallel simulation

true_energies = np.zeros((len(dec_to_sim), len(index_to_sim), N_per_core))
reco_energies = np.zeros((len(dec_to_sim), len(index_to_sim), N_per_core))
ras = np.zeros((len(dec_to_sim), len(index_to_sim), N_per_core))
decs = np.zeros((len(dec_to_sim), len(index_to_sim), N_per_core))

for i, src_dec in enumerate(dec_to_sim):

    for j, index in enumerate(index_to_sim):

        # Source
        power_law = PowerLawFlux(5e-19, 1e5, index, lower_energy=Emin, upper_energy=Emax)
        source = PointSource(flux_model=power_law, coord=(np.pi, src_dec))

        simulator = Simulator([source], detector)
        simulator.max_cosz = 0.1
        simulator.run(N_per_core, show_progress=False)
    
        true_energies[i][j] = simulator.true_energy
        reco_energies[i][j] = simulator.reco_energy
        ras[i][j] = simulator.ra
        decs[i][j] = simulator.dec

    if COMM.rank == 0:
        print('src_dec = %.2f completed...' % src_dec)
    
# Gather results and save

true_energies = MPI.COMM_WORLD.gather(true_energies, root=0)
reco_energies = MPI.COMM_WORLD.gather(reco_energies, root=0)
ras = MPI.COMM_WORLD.gather(ras, root=0)
decs = MPI.COMM_WORLD.gather(decs, root=0)

if COMM.rank == 0:

    true_energy = np.transpose(true_energies, (1, 2, 0, 3))
    reco_energy = np.transpose(reco_energies, (1, 2, 0, 3))
    ra = np.transpose(ras, (1, 2, 0, 3))
    dec  = np.transpose(decs, (1, 2, 0, 3))
    
    print("Done!")
    print("time:", time.time() - start_time)

    with h5py.File(output_file, 'w') as f:

        for i, src_dec in enumerate(dec_to_sim):
            
            folder = f.create_group('dec_%.2f' % src_dec)

            for j, index in enumerate(index_to_sim):
            
                subfolder = folder.create_group('index_%.2f' % index)
        
                subfolder.create_dataset('true_energy', data=np.concatenate(true_energy[i][j]))
        
                subfolder.create_dataset('reco_energy', data=np.concatenate(reco_energy[i][j]))
                
                subfolder.create_dataset('ra', data=np.concatenate(ra[i][j]))
                
                subfolder.create_dataset('dec', data=np.concatenate(dec[i][j]))

        f.create_dataset('index_to_sim', data=index_to_sim)
   
        f.create_dataset('dec_to_sim', data=dec_to_sim)

        f.create_dataset('Nevents', data=N_per_core * COMM.size)
            
        f.create_dataset('source_type', data=source.source_type)

        f.create_dataset('normalisation', data=source.flux_model._normalisation)

        f.create_dataset('normalisation_energy', data=source.flux_model._normalisation_energy)

