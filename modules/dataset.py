# define a dataset for training/testing

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset

EJb = (4.0, 6.0)
ECb = (0.5, 1.0)
ELb = (1.0, 2.0)

FreqB = (4.5, 6.0)  # GHz
peakB = (0.5, 2.0)
widthB = (0.005, 0.01)  # GHz
noice = 0.25


fpts = np.linspace(FreqB[0], FreqB[1], 300)  # (h,)


def lorfunc(x, yscale, x0, gamma):
    return yscale / (1 + ((x - x0) / gamma) ** 2)


def normalize(x, xb):
    return (float(x) - xb[0]) / (xb[1] - xb[0])


class SpectrumDataset(Dataset):
    def __init__(self, filepath):
        super(SpectrumDataset, self).__init__()
        with h5.File(filepath, "r") as file:
            self.flxs = file["flxs"][:]  # type: ignore
            self.params = file["params"][:]  # type: ignore
            self.energies = file["energies"][:]  # type: ignore

    def __len__(self):
        return len(self.params)  # type: ignore

    def make_spectrum(self, energies):
        # energies: (n, m')
        fs = []
        for i in [0, 1]:
            for j in range(i + 1, energies.shape[1]):
                fs.append(energies[:, j] - energies[:, i])
        fs = np.stack(fs, axis=1)  # (n, m)
        _, m = fs.shape

        # calculate the spectrum
        yscales = np.random.uniform(*peakB, (1, 1, m))
        yscales *= np.random.choice([-1, 1], (1, 1, m))  # random sign
        gammas = np.random.uniform(*widthB, (1, 1, m))
        xs = fpts[None, :, None]  # (1, h, 1)
        x0s = fs[:, None, :]  # (n, 1, m)
        ys = lorfunc(xs, yscales, x0s, gammas)  # (n, h, m)
        spectrum = np.sum(ys, axis=-1)  # (n, h)

        # add noise
        spectrum += np.random.normal(0, noice, spectrum.shape)

        return spectrum

    def __getitem__(self, idx):
        energies = self.energies[idx]  # (n, m') # type: ignore
        spectrum = self.make_spectrum(energies)  # (n, h)
        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        params = self.params[idx]  # type: ignore
        params = (
            normalize(params[0], EJb),  # type: ignore
            normalize(params[1], ECb),  # type: ignore
            normalize(params[2], ELb),  # type: ignore
        )
        return spectrum, torch.tensor(params, dtype=torch.float32)


if __name__ == "__main__":
    dataset = SpectrumDataset("data/train.h5")
    spectrum, params = dataset[np.random.randint(len(dataset))]

    import matplotlib.pyplot as plt

    print(params)
    plt.pcolormesh(dataset.flxs, fpts, spectrum.T)  # type: ignore
    # plt.plot(dataset.flxs, curve, color="r")  # type: ignore
    plt.show()
