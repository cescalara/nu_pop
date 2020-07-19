#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import h5py
import time

import gc

# gc.disable()

from icecube_tools.point_source_likelihood.energy_likelihood import (
    MarginalisedEnergyLikelihoodFromSim,
    MarginalisedEnergyLikelihoodFixed,
)
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.point_source_likelihood.spatial_likelihood import (
    EnergyDependentSpatialGaussianLikelihood,
)
from icecube_tools.point_source_likelihood.prior import GaussianPrior
from icecube_tools.point_source_likelihood.point_source_likelihood import (
    PointSourceLikelihood,
)

# import warnings
# warnings.filterwarnings("ignore")

"""
Script to run TS calculation for many trials for 
different declinations, spectral indices and source counts.
"""


def get_TS(
    ras,
    decs,
    energies,
    Nsrc,
    source_ras,
    source_decs,
    source_energies,
    min_dec,
    direction_likelihood,
    energy_likelihood,
    source_coord,
    index_prior,
    bg_energy_likelihood,
    band_width_factor,
):

    new_ras = np.random.uniform(0, 2 * np.pi, len(ras))
    new_decs = decs

    if Nsrc > 0:
        # add Nsrc source events
        i_src = np.random.choice(range(len(source_energies)), replace=False, size=Nsrc)
        src_ras = source_ras[i_src]
        src_decs = source_decs[i_src]
        src_energies = source_energies[i_src]

        # handle min_dec
        src_ras = src_ras[src_decs > min_dec]
        src_decs = src_decs[src_decs > min_dec]

        # total input
        total_ras = np.concatenate((new_ras, src_ras), axis=0)
        total_decs = np.concatenate((new_decs, src_decs), axis=0)
        total_energies = np.concatenate((energies, src_energies), axis=0)

    else:

        total_ras = new_ras
        total_decs = new_decs
        total_energies = energies

    likelihood = PointSourceLikelihood(
        direction_likelihood,
        energy_likelihood,
        total_ras,
        total_decs,
        total_energies,
        source_coord,
        index_prior=index_prior,
        bg_energy_likelihood=bg_energy_likelihood,
        band_width_factor=band_width_factor,
    )

    ts = likelihood.get_test_statistic()

    return ts


FILE_STEM = "/cfs/klemming/nobackup/c/capel/neutrino_model/"

COMM = MPI.COMM_WORLD

if COMM.rank == 0:

    start_time = time.time()

    # SETUP

    # Parameters
    Nevents = 497000
    band_width_factor = 3.0
    Ntrials = int(2 * 48)  # 2*48
    Nsrc_list = np.array(range(100))  # 100
    min_dec = np.arcsin(-0.1)  # max cosz = 0.1
    output_file = FILE_STEM + "output/TS_dist_dec_30_3_6_Nevents497.h5"

    # Inputs
    sim_bg_file = FILE_STEM + "output/bg_5e5.h5"
    sim_pl_file = FILE_STEM + "output/pl_1e6.h5"
    Ares_2_file = FILE_STEM + "input/angres_plot_E-2.csv"
    Ares_a_file = FILE_STEM + "input/angres_plot_atmos.csv"
    sim_src_file = FILE_STEM + "output/source_1e3_6_30.h5"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("Nevents", data=Nevents)
        f.create_dataset("Ntrials", data=Ntrials)
        f.create_dataset("Nsrc_list", data=Nsrc_list)
        f.create_dataset("band_width_factor", data=band_width_factor)
        f.create_dataset("min_dec", data=min_dec)

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
    angres_index_list = [2.0, 3.7]

    direction_likelihood = EnergyDependentSpatialGaussianLikelihood(
        angres_list, angres_index_list
    )

    # Index prior
    index_prior = GaussianPrior(2.19, 0.1)

    # Events
    with h5py.File(sim_bg_file, "r") as f:
        file_energies = f["reco_energy"][()]
        ras = f["ra"][()]
        decs = f["dec"][()]

    # Sample Nevents randomly
    i_event = np.random.choice(range(len(file_energies)), replace=False, size=Nevents)
    energies = file_energies[i_event]
    ras = ras[i_event]
    decs = decs[i_event]

    # Point source
    with h5py.File(sim_src_file, "r") as f:
        dec_to_sim = f["dec_to_sim"][()]
        index_to_sim = f["index_to_sim"][()]
        Nsim = f["Nevents"][()]

        source_energies = np.zeros((len(dec_to_sim), len(index_to_sim), Nsim))
        source_ras = np.zeros((len(dec_to_sim), len(index_to_sim), Nsim))
        source_decs = np.zeros((len(dec_to_sim), len(index_to_sim), Nsim))

        for i, src_dec in enumerate(dec_to_sim):
            folder = f["dec_%.2f" % src_dec]

            for j, index in enumerate(index_to_sim):
                subfolder = folder["index_%.2f" % index]
                source_energies[i][j] = subfolder["reco_energy"][()]
                source_ras[i][j] = subfolder["ra"][()]
                source_decs[i][j] = subfolder["dec"][()]

    with h5py.File(output_file, "r+") as f:
        f.create_dataset("index_to_sim", data=index_to_sim)
        f.create_dataset("dec_to_sim", data=dec_to_sim)

    trials = np.arange(Ntrials)
    trial_segs = np.array_split(trials, COMM.size)

else:

    trial_segs = None
    energies = None
    ras = None
    decs = None
    source_energies = None
    source_ras = None
    source_decs = None
    direction_likelihood = None
    energy_likelihood = None
    bg_energy_likelihood = None
    index_prior = None
    band_width_factor = None
    Nsrc_list = None
    index_to_sim = None
    dec_to_sim = None
    min_dec = None

trial_segs = COMM.scatter(trial_segs, root=0)

energies = COMM.bcast(energies, root=0)
ras = COMM.bcast(ras, root=0)
decs = COMM.bcast(decs, root=0)
source_energies = COMM.bcast(source_energies, root=0)
source_ras = COMM.bcast(source_ras, root=0)
source_decs = COMM.bcast(source_decs, root=0)
direction_likelihood = COMM.bcast(direction_likelihood, root=0)
energy_likelihood = COMM.bcast(energy_likelihood, root=0)
bg_energy_likelihood = COMM.bcast(bg_energy_likelihood, root=0)
index_prior = COMM.bcast(index_prior, root=0)
band_width_factor = COMM.bcast(band_width_factor, root=0)
Nsrc_list = COMM.bcast(Nsrc_list, root=0)
index_to_sim = COMM.bcast(index_to_sim, root=0)
dec_to_sim = COMM.bcast(dec_to_sim, root=0)
min_dec = COMM.bcast(min_dec, root=0)

# SIMULATION

TS = np.zeros((len(Nsrc_list), len(trial_segs)))

for i, src_dec in enumerate(dec_to_sim):

    source_coord = (np.pi, src_dec)
    energy_likelihood.set_src_dec(src_dec)

    if src_dec > 0.9:
        bwf = band_width_factor * 2
    else:
        bwf = band_width_factor

    if COMM.rank == 0:
        with h5py.File(output_file, "r+") as f:
            folder = f.create_group("dec_%.2f" % src_dec)

    for j, index in enumerate(index_to_sim):

        if COMM.rank == 0:
            with h5py.File(output_file, "r+") as f:
                folder = f["dec_%.2f" % src_dec]
                subfolder = folder.create_group("index_%.2f" % index)

        for k, Nsrc in enumerate(Nsrc_list):

            for l, _ in enumerate(trial_segs):

                TS[k][l] = get_TS(
                    ras,
                    decs,
                    energies,
                    Nsrc,
                    source_ras[i][j],
                    source_decs[i][j],
                    source_energies[i][j],
                    min_dec,
                    direction_likelihood,
                    energy_likelihood,
                    source_coord,
                    index_prior,
                    bg_energy_likelihood,
                    bwf,
                )

        results = MPI.COMM_WORLD.gather(TS, root=0)

        gc.collect()

        if COMM.rank == 0:

            print("\t index = %.2f completed" % index)

            results = np.transpose(results, (1, 0, 2))

            with h5py.File(output_file, "r+") as f:
                folder = f["dec_%.2f" % src_dec]
                subfolder = folder["index_%.2f" % index]
                for k, Nsrc in enumerate(Nsrc_list):
                    subfolder.create_dataset(
                        "TS_" + str(Nsrc), data=np.concatenate(results[k])
                    )

    if COMM.rank == 0:

        print("dec = %.2f completed..." % src_dec)

if COMM.rank == 0:

    print("time:", time.time() - start_time)
