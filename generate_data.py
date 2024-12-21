# make energies of fluxonium under different external fluxes
# and save them in a file

import os

import h5py as h5
import numpy as np
from scqubits import Fluxonium  # type: ignore
from tqdm.auto import tqdm

# parameters
data_path = "data/dev.h5"
num_per = 7
EJb = (4.0, 6.0)
ECb = (0.5, 1.0)
ELb = (1.0, 2.0)

level_num = 5
cutoff = 50
flxs = np.linspace(0.0, 1.0, 240)


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
print("Calculating on EC-EL plane")
ECEL_num = 0
for EC in tqdm(np.linspace(*ECb, num_per)):
    for EL in tqdm(np.linspace(*ELb, num_per)):
        energy = calculate_spectrum(flxs, EJb[0], EC, EL, level_num, cutoff)

        # since energy is proportional to EJ, we can just use the energy
        for EJ in np.linspace(EJb[0] + 0.01, EJb[1], num_per):
            ratio = EJ / EJb[0]
            tEC = EC * ratio
            tEL = EL * ratio

            if ECb[0] <= tEC <= ECb[1] and ELb[0] <= tEL <= ELb[1]:
                ECEL_num += 1
                params.append((EJ, tEC, tEL))
                energies.append(energy * ratio)
print("EC-EL plane data points:", ECEL_num)

print("Calculating on EJ-EL plane")
EJEL_num = 0
for EJ in tqdm(np.linspace(*EJb, num_per)):
    for EL in tqdm(np.linspace(*ECb, num_per)):
        energy = calculate_spectrum(flxs, EJ, ECb[0], EL, level_num, cutoff)

        for EC in np.linspace(ECb[0] + 0.01, ECb[1], num_per):
            ratio = EC / ECb[0]
            tEJ = EJ * ratio
            tEL = EL * ratio
            if EJb[0] <= tEJ <= EJb[1] and ELb[0] <= tEL <= ELb[1]:
                EJEL_num += 1
                params.append((tEJ, EC, tEL))
                energies.append(energy * ratio)
print("EJ-EL plane data points:", EJEL_num)

print("Calculating on EJ-EC plane")
EJEC_num = 0
for EJ in tqdm(np.linspace(*EJb, num_per)):
    for EC in tqdm(np.linspace(*ECb, num_per)):
        energy = calculate_spectrum(flxs, EJ, EC, ELb[0], level_num, cutoff)

        for EL in np.linspace(ELb[0] + 0.01, ELb[1], num_per):
            ratio = EL / ELb[0]
            tEJ = EJ * ratio
            tEC = EC * ratio
            if EJb[0] <= tEJ <= EJb[1] and ECb[0] <= tEC <= ECb[1]:
                EJEC_num += 1
                params.append((tEJ, tEC, EL))
                energies.append(energy * ratio)
print("EJ-EC plane data points:", EJEC_num)

print("Total data points:", len(params))
params = np.array(params)
energies = np.array(energies)

os.makedirs(os.path.dirname(data_path), exist_ok=True)
dump_data(data_path, flxs, params, energies)
