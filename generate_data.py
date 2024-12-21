# make energies of fluxonium under different external fluxes
# and save them in a file

import os

import h5py as h5
import numpy as np
from scqubits import Fluxonium  # type: ignore
from tqdm.auto import tqdm

# parameters
data_path = "data/fluxonium_2.h5"
num_per = 25
EJb = (3.5, 6.5)
ECb = (0.3, 1.5)
ELb = (0.7, 2.5)

DRY_RUN = False

level_num = 10
cutoff = 50
flxs = np.linspace(0.0, 0.5, 120)


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
for EC in tqdm(np.linspace(ECb[0] + 1e-3, ECb[1], num_per + 1)):
    EL_num = 0
    for EL in tqdm(np.linspace(ELb[0] + 1e-3, ELb[1], num_per + 1)):
        if DRY_RUN:
            energy = np.random.randn(len(flxs), level_num)
        else:
            energy = calculate_spectrum(flxs, EJb[1], EC, EL, level_num, cutoff)

        # since energy is proportional to EJ, we can just use the energy
        for EJ in np.linspace(EJb[0] + 1e-3, EJb[1], num_per + 1):
            ratio = EJ / EJb[1]
            tEC = EC * ratio
            tEL = EL * ratio

            if ECb[0] <= tEC <= ECb[1] and ELb[0] <= tEL <= ELb[1]:
                EL_num += 1
                params.append((EJ, tEC, tEL))
                energies.append(energy * ratio)
    print("EL line data points:", EL_num)
    ECEL_num += EL_num
print("EC-EL plane data points:", ECEL_num)

print("Calculating on EJ-EL plane")
EJEL_num = 0
for EJ in tqdm(np.linspace(EJb[0] + 1e-3, EJb[1], num_per + 1)):
    EL_num = 0
    for EL in tqdm(np.linspace(ELb[0] + 1e-3, ELb[1], num_per + 1)):
        if DRY_RUN:
            energy = np.random.randn(len(flxs), level_num)
        else:
            energy = calculate_spectrum(flxs, EJ, ECb[1], EL, level_num, cutoff)

        for EC in np.linspace(ECb[0] + 1e-3, ECb[1], num_per + 1):
            ratio = EC / ECb[1]
            tEJ = EJ * ratio
            tEL = EL * ratio
            if EJb[0] <= tEJ <= EJb[1] and ELb[0] <= tEL <= ELb[1]:
                EL_num += 1
                params.append((tEJ, EC, tEL))
                energies.append(energy * ratio)
    print("EL line data points:", EL_num)
    EJEL_num += EL_num
print("EJ-EL plane data points:", EJEL_num)

print("Calculating on EJ-EC plane")
EJEC_num = 0
for EJ in tqdm(np.linspace(EJb[0] + 1e-3, EJb[1], num_per + 1)):
    EC_num = 0
    for EC in tqdm(np.linspace(ECb[0] + 1e-3, ECb[1], num_per + 1)):
        if DRY_RUN:
            energy = np.random.randn(len(flxs), level_num)
        else:
            energy = calculate_spectrum(flxs, EJ, EC, ELb[1], level_num, cutoff)

        for EL in np.linspace(ELb[0] + 1e-3, ELb[1], num_per + 1):
            ratio = EL / ELb[1]
            tEJ = EJ * ratio
            tEC = EC * ratio
            if EJb[0] <= tEJ <= EJb[1] and ECb[0] <= tEC <= ECb[1]:
                EC_num += 1
                params.append((tEJ, tEC, EL))
                energies.append(energy * ratio)
    print("EC line data points:", EC_num)
    EJEC_num += EC_num
print("EJ-EC plane data points:", EJEC_num)

print("Total data points:", len(params))

# we can flip the data around 0.5 to make the other half
# since the fluxonium is symmetric
flxs = np.concatenate([flxs, 1.0 - flxs[::-1]])
for i in range(len(params)):
    energies[i] = np.concatenate([energies[i], energies[i][::-1]])

params = np.array(params)
energies = np.array(energies)

os.makedirs(os.path.dirname(data_path), exist_ok=True)
dump_data(data_path, flxs, params, energies)
