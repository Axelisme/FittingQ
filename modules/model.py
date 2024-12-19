# define a convolutional neural network


import torch
from torch import Tensor, nn


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """3 convolution"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    # define a residual net like ResNet18
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(512, 3)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or planes != inplanes:
            downsample = conv1(inplanes, planes, stride)
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # x = x.unsqueeze(1)  # (B, 1, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


FreqB = (4.5, 6.0)  # GHz


class PredictNet(torch.nn.Module):
    def __init__(self):
        super(PredictNet, self).__init__()
        self.resnet = ResNet()
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
        spectrum = torch.stack([avg, std], axis=1)  # (B, 2, n)

        return spectrum

    def prepare_input(self, x):
        # x.shape: (batch, n)
        x = self.analyze_spectrum(x)
        return x

    def forward(self, x):
        x = self.prepare_input(x)
        out = self.resnet(x)  # type: ignore
        return torch.sigmoid(out)
