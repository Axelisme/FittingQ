# define a dataset for training/testing

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset

EJb = (4.0, 6.0)
ECb = (0.5, 1.0)
ELb = (1.0, 2.0)

FreqB = (4.5, 6.0)  # GHz
widthB = (0.002, 0.005)  # GHz


transitions = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]


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
        # torch version
        fs = []
        for i, j in transitions:
            fs.append(energies[:, j] - energies[:, i])
        fs = np.stack(fs, axis=1)  # (n, m)
        _, m = fs.shape

        fs = torch.tensor(fs, dtype=torch.float32)
        weights = torch.tensor([1, 1, 0.2, 1, 0.2], dtype=torch.float32)
        fpts = torch.linspace(FreqB[0], FreqB[1], 1001, dtype=torch.float32)  # (h,)
        fs = fs.cuda()
        weights = weights.cuda()
        fpts = fpts.cuda()
        with torch.no_grad():
            yscales = weights * (torch.rand(m, device=fs.device) + 0.5)
            yscales *= torch.randint(0, 2, (m,), device=fs.device) * 2 - 1
            yscales = yscales[None, None, :]
            gammas = torch.rand(m, device=fs.device) * 0.003 + 0.002
            xs = fpts[None, :, None]
            x0s = fs[:, None, :]
            ys = lorfunc(xs, yscales, x0s, gammas)
            spectrum = torch.sum(ys, dim=-1)

            spectrum += torch.normal(0, 0.35, spectrum.shape, device=fs.device)

        return spectrum.T

    def __getitem__(self, idx):
        energies = self.energies[idx]  # (n, m') # type: ignore
        spectrum = self.make_spectrum(energies)  # (n, h)
        params = self.params[idx]  # type: ignore
        params = (
            normalize(params[0], EJb),  # type: ignore
            normalize(params[1], ECb),  # type: ignore
            normalize(params[2], ELb),  # type: ignore
        )
        return spectrum, torch.tensor(
            params, dtype=torch.float32, device=spectrum.device
        )


if __name__ == "__main__":
    dataset = SpectrumDataset("data/train.h5")
    spectrum, params = dataset[np.random.randint(len(dataset))]

    import matplotlib.pyplot as plt

    spectrum -= torch.median(spectrum, dim=1, keepdim=True).values

    from modules.model import PredictNet

    curve = PredictNet.prepare_input(spectrum[None])[0, 0]

    spectrum = spectrum.cpu().numpy()
    curve = curve.cpu().numpy()

    plt.imshow(spectrum, aspect="auto", origin="lower", extent=(0, 240, 4.5, 6.0))
    plt.plot(curve * 1.5 + 4.5, "r")
    plt.ylim(4.5, 6.0)
    plt.xlim(0, 240)
    plt.show()
    plt.savefig("spectrum.png")
