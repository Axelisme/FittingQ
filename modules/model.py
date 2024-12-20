# define a convolutional neural network


import torch

FreqB = (4.5, 6.0)  # GHz


class PredictNet(torch.nn.Module):
    def __init__(self, backbone):
        super(PredictNet, self).__init__()
        self.backbone = backbone()
        kernel = torch.arange(-10, 11, dtype=torch.float32)
        kernel = torch.exp(-(kernel**2) / 2)
        self.kernel = kernel / kernel.sum()

        self.fpts = torch.linspace(FreqB[0], FreqB[1], 300, dtype=torch.float32)  # (h,)

    def analyze_spectrum(self, spectrum):
        # fpts: (h,)
        # spectrum: (B, n, h)

        # use guassian filter to smooth the spectrum
        B, n, h = spectrum.shape
        spectrum = spectrum.view(B * n, h)  # (B*n, h)
        self.kernel = self.kernel.to(spectrum.device)
        spectrum = torch.nn.functional.conv1d(
            spectrum[:, None, :], self.kernel[None, None, :], padding=10
        ).squeeze(1)  # (B*n, h)
        spectrum = spectrum.view(B, n, h)  # (B, n, h)

        # normalize the spectrum
        contrasts = (
            spectrum.max(axis=-1).values - spectrum.min(axis=-1).values
        )  # (B, n)
        spectrum = spectrum - spectrum.median(axis=-1).values[..., None]  # (B, n, h)
        spectrum = spectrum / contrasts[..., None]  # (B, n, h)
        spectrum = torch.where(
            spectrum.abs() > 0.2, spectrum, torch.zeros_like(spectrum)
        )
        spectrum = spectrum * spectrum  # (B, n, h)
        spectrum = spectrum / spectrum.sum(axis=-1)[..., None]  # (B, n, h)

        # caluculate the avg, std, skewness, kurtosis
        self.fpts = self.fpts.to(spectrum.device)
        fpts = self.fpts[None, None, :]  # (1, 1, h)
        avg = (fpts * spectrum).sum(axis=-1)  # (B, n)
        std = torch.sqrt(torch.sum((fpts - avg[..., None]) ** 2 * spectrum, axis=-1))
        skewness = torch.sum((fpts - avg[..., None]) ** 3 * spectrum, axis=-1) / std**3
        spectrum = torch.stack([avg, std, skewness], axis=1)  # (B, 3, n)

        return spectrum

    def prepare_input(self, x):
        # x.shape: (batch, n)
        x = self.analyze_spectrum(x)
        return x

    def forward(self, x):
        x = self.prepare_input(x)
        out = self.backbone(x)  # type: ignore
        return torch.sigmoid(out)
