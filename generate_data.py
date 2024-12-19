# make energies of fluxonium under different external fluxes
# and save them in a file

import os

import h5py as h5
import numpy as np
from scqubits import Fluxonium  # type: ignore
from tqdm.auto import trange

# parameters
data_path = "data/dev.h5"
data_num = 300
EJb = (4.0, 6.0)
ECb = (0.5, 1.0)
ELb = (1.0, 2.0)

level_num = 4
cutoff = 50
flxs = np.linspace(0.0, 1.0, 121)


def calculate_spectrum(flxs, EJ, EC, EL, evals_count=4, cutoff=50):
    fluxonium = Fluxonium(EJ, EC, EL, flux=0.0, cutoff=cutoff)
    spectrumData = fluxonium.get_spectrum_vs_paramvals(
        "flux", flxs, evals_count=evals_count
    )

    return spectrumData.energy_table


def dump_data(filepath, flxs, params, energies):
    with h5.File(filepath, "w") as f:
        f.create_dataset("flxs", data=flxs)
        f.create_dataset("params", data=params)
        f.create_dataset("energies", data=energies)


params = []
energies = []
for _ in trange(data_num):
    EJ = np.random.uniform(*EJb)
    EC = np.random.uniform(*ECb)
    EL = np.random.uniform(*ELb)
    params.append((EJ, EC, EL))

    energies.append(calculate_spectrum(flxs, EJ, EC, EL, level_num, cutoff))
params = np.array(params)
energies = np.array(energies)

os.makedirs(os.path.dirname(data_path), exist_ok=True)
dump_data(data_path, flxs, params, energies)
