#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
import h5py
import time

"""
Script to run parallel simulations of simple power law.
Used as input to MarginalisedEnergyLikelihood.
"""

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux, BrokenPowerLawFlux
from icecube_tools.source.source_model import DiffuseSource

from icecube_tools.simulator import Simulator

COMM = MPI.COMM_WORLD

if COMM.rank == 0:

    FILE_STEM = "/cfs/klemming/nobackup/c/capel/neutrino_model/"

    # Setup
    N = int(1e6)
    Emin = 1e3  # GeV
    pl_index = 1.4
    output_file = FILE_STEM + "output/pl_1e6.h5"

    print("Setting up inputs...")

    # Effective area
    Aeff_filename = FILE_STEM + "input/IC86-2012-TabulatedAeff.txt"
    effective_area = EffectiveArea(Aeff_filename)

    # Energy resolution
    eres_file = FILE_STEM + "input/effective_area.h5"
    energy_res = EnergyResolution(eres_file)

    # Angular resolution
    Ares_file = FILE_STEM + "input/IC86-2012-AngRes.txt"
    ang_res = AngularResolution(Ares_file)

    # Detector
    detector = IceCube(effective_area, energy_res, ang_res)

    # Sources
    # Simple power law input
    power_law = PowerLawFlux(
        1.01e-18, 1e5, pl_index, lower_energy=Emin, upper_energy=1e9
    )
    diffuse_pl = DiffuseSource(flux_model=power_law)
    sources = [diffuse_pl]

    # Shared runs
    N_per_core = int(N / COMM.size)
    print(str(N_per_core) + " runs on each core")
    print("Starting...")

    start_time = time.time()

else:

    detector = None
    sources = None
    N_per_core = None

detector = COMM.bcast(detector, root=0)
sources = COMM.bcast(sources, root=0)
N_per_core = COMM.bcast(N_per_core, root=0)

# Parallel simulation of background

simulator = Simulator(sources, detector)
simulator.time = 8.0
simulator.max_cosz = 0.1
simulator.run(N_per_core, show_progress=False)

# Gather results and save

true_energy = MPI.COMM_WORLD.gather(simulator.true_energy, root=0)
reco_energy = MPI.COMM_WORLD.gather(simulator.reco_energy, root=0)
ra = MPI.COMM_WORLD.gather(simulator.ra, root=0)
dec = MPI.COMM_WORLD.gather(simulator.dec, root=0)
source_label = MPI.COMM_WORLD.gather(simulator.source_label, root=0)

if COMM.rank == 0:

    true_energy = np.concatenate(true_energy)
    reco_energy = np.concatenate(reco_energy)
    ra = np.concatenate(ra)
    dec = np.concatenate(dec)
    source_label = np.concatenate(source_label)

    print("Done!")
    print("time:", time.time() - start_time)

    with h5py.File(output_file, "w") as f:

        f.create_dataset("true_energy", data=true_energy)

        f.create_dataset("reco_energy", data=reco_energy)

        f.create_dataset("ra", data=ra)

        f.create_dataset("dec", data=dec)

        f.create_dataset("source_label", data=source_label)

        for i, source in enumerate(sources):

            s = f.create_group("source_" + str(i))

            if isinstance(source.flux_model, PowerLawFlux):

                s.create_dataset("index", data=source.flux_model._index)

                s.create_dataset(
                    "normalisation_energy", data=source.flux_model._normalisation_energy
                )

            elif isinstance(source.flux_model, BrokenPowerLawFlux):

                s.create_dataset("index1", data=source.flux_model._index1)

                s.create_dataset("index2", data=source.flux_model._index2)

                s.create_dataset("break_energy", data=source.flux_model._break_energy)

            s.create_dataset("source_type", data=source.source_type)

            s.create_dataset("normalisation", data=source.flux_model._normalisation)
