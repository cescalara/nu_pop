#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import h5py
import time

from icecube_tools.point_source_likelihood.energy_likelihood import (
    MarginalisedEnergyLikelihoodFromSim,
    MarginalisedEnergyLikelihoodFixed,
)
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.point_source_likelihood.spatial_likelihood import (
    EnergyDependentSpatialGaussianLikelihood,
    SpatialGaussianLikelihood,
)
from icecube_tools.point_source_likelihood.prior import GaussianPrior
from icecube_tools.point_source_likelihood.point_source_likelihood import (
    PointSourceLikelihood,
    EnergyDependentSpatialPointSourceLikelihood,
    SpatialOnlyPointSourceLikelihood,
)

import warnings

warnings.filterwarnings("ignore")

"""
Script to run TS calculation for many trials.

This script just runs a set number of trials with only background (Nsrc=0).
Scans over different test source declinations.
"""

FILE_STEM = "/cfs/klemming/nobackup/c/capel/neutrino_model/"

COMM = MPI.COMM_WORLD

if COMM.rank == 0:

    start_time = time.time()

    # SETUP

    # Parameters
    Nevents = 497000
    dec_to_sim = np.arcsin(np.linspace(-0.08, 0.97, 30))
    band_width_factor = 3.0
    Ntrials = int(1e4)
    output_file = FILE_STEM + "output/TS_dist_bg_30_3_withE_Nevents497.h5"

    # Inputs
    sim_bg_file = FILE_STEM + "output/bg_5e5.h5"
    sim_pl_file = FILE_STEM + "output/pl_1e6.h5"
    Ares_2_file = FILE_STEM + "input/angres_plot_E-2.csv"
    Ares_a_file = FILE_STEM + "input/angres_plot_atmos.csv"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("dec_to_sim", data=dec_to_sim)
        f.create_dataset("Nevents", data=Nevents)
        f.create_dataset("Ntrials", data=Ntrials)
        f.create_dataset("band_width_factor", data=band_width_factor)

    # Likelihood

    # Energy likelihood
    with h5py.File(sim_pl_file, "r") as f:
        sim_reco_energy = f["reco_energy"][()]
        sim_index = f["source_0/index"][()]
        sim_dec = f["dec"][()]

    energy_likelihood = MarginalisedEnergyLikelihoodFromSim(
        sim_reco_energy, sim_dec, sim_index=sim_index, min_E=50, max_E=5e10
    )

    with h5py.File(sim_bg_file, "r") as f:
        sim_reco_energy = f["reco_energy"][()]

    bg_energy_likelihood = MarginalisedEnergyLikelihoodFixed(
        sim_reco_energy, min_E=50, max_E=5e10
    )

    # Direction likelihood
    angres_2 = AngularResolution(Ares_2_file)
    angres_a = AngularResolution(Ares_a_file)
    angres_list = [angres_2, angres_a]
    index_list = [2.0, 3.7]

    direction_likelihood = EnergyDependentSpatialGaussianLikelihood(
        angres_list, index_list
    )

    # Index prior
    index_prior = GaussianPrior(2.19, 0.1)

    # Events
    with h5py.File(sim_bg_file, "r") as f:
        energies = f["reco_energy"][()]
        ras = f["ra"][()]
        decs = f["dec"][()]

    # Sample Nevents randomly
    i_event = np.random.choice(range(len(energies)), replace=False, size=Nevents)
    energies = energies[i_event]
    ras = ras[i_event]
    decs = decs[i_event]
    event_coords = [(ras[i], decs[i]) for i in range(0, len(ras))]

    trials = np.arange(Ntrials)
    trial_segs = np.array_split(trials, COMM.size)

else:

    trial_segs = None
    Ntrials = None
    event_coords = None
    energies = None
    ras = None
    decs = None
    dec_to_sim = None
    direction_likelihood = None
    energy_likelihood = None
    bg_energy_likelihood = None
    index_prior = None
    band_width_factor = None

trial_segs = COMM.scatter(trial_segs, root=0)

Ntrials = COMM.bcast(Ntrials, root=0)
event_coords = COMM.bcast(event_coords, root=0)
energies = COMM.bcast(energies, root=0)
ras = COMM.bcast(ras, root=0)
decs = COMM.bcast(decs, root=0)
dec_to_sim = COMM.bcast(dec_to_sim, root=0)
direction_likelihood = COMM.bcast(direction_likelihood, root=0)
energy_likelihood = COMM.bcast(energy_likelihood, root=0)
bg_energy_likelihood = COMM.bcast(bg_energy_likelihood, root=0)
index_prior = COMM.bcast(index_prior, root=0)
band_width_factor = COMM.bcast(band_width_factor, root=0)

# SIMULATION

TS = np.zeros(len(trial_segs))

for i, src_dec in enumerate(dec_to_sim):

    source_coord = (np.pi, src_dec)
    energy_likelihood.set_src_dec(src_dec)

    for j, _ in enumerate(trial_segs):

        ras = np.random.uniform(0, 2 * np.pi, len(ras))
        decs = decs
        total_energies = energies

        # As there are few events at high dec, we need to use a larger bandwidth.
        # This is accounted for in the likelihood normalisation.
        if src_dec > 0.9:
            bwf = band_width_factor * 3

        else:
            bwf = band_width_factor

        likelihood = PointSourceLikelihood(
            direction_likelihood,
            energy_likelihood,
            ras,
            decs,
            total_energies,
            source_coord,
            index_prior=index_prior,
            bg_energy_likelihood=bg_energy_likelihood,
            band_width_factor=bwf,
        )

        # likelihood = EnergyDependentSpatialPointSourceLikelihood(direction_likelihood, ras, decs,
        #                                                         total_energies, source_coord,
        #                                                         band_width_factor=bwf)

        ts = likelihood.get_test_statistic()
        TS[j] = ts

    if COMM.rank == 0:
        print("dec = %.2f completed..." % src_dec)

    results = MPI.COMM_WORLD.gather(TS, root=0)

    if COMM.rank == 0:

        with h5py.File(output_file, "r+") as f:
            f.create_dataset("TS_%.2f" % src_dec, data=np.concatenate(results))

if COMM.rank == 0:

    print("Done!")
    print("time:", time.time() - start_time)
