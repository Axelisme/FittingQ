# define a convolutional neural network


import torch


class PredictNet(torch.nn.Module):
    kernel = torch.arange(-4, 5, dtype=torch.float32)
    kernel = torch.exp(-0.5 * kernel**2 / 2.0**2)
    kernel = kernel / kernel.sum()
    kernel = kernel[None, None, :, None]  # (1, 11, 1)

    def __init__(self, backbone):
        super(PredictNet, self).__init__()
        self.backbone = backbone()

    @classmethod
    def apply_guassian_filter(cls, x):
        cls.kernel = cls.kernel.to(x.device)
        return torch.nn.functional.conv2d(
            x[:, None, :, :], cls.kernel, padding=(4, 0)
        ).squeeze(1)  # (B, H, W)

    @classmethod
    def remove_noise(cls, x):
        x -= x.median(dim=1, keepdim=True).values
        maxs = x.max(dim=1, keepdim=True).values
        mins = x.min(dim=1, keepdim=True).values
        contrasts = maxs - mins  # (B, 1, W)

        # convert to positive values
        x = x.abs()  # (B, H, W)

        # remove small values per row
        x[x < 0.4 * contrasts] = 0.0

        # normalize
        x = x / x.sum(dim=1, keepdim=True)

        return x

    @classmethod
    def find_peak(cls, x):
        B, H, W = x.shape

        # first integral along H axis
        x = x.cumsum(dim=1)  # (B, H, W)
        idxs = (x - 0.5).abs().argmin(dim=1, keepdim=True)  # (B, 1, W)

        return idxs.to(dtype=torch.float32) / H

    @classmethod
    def prepare_input(cls, x):
        # x.shape = (B, H, W)
        with torch.no_grad():
            x = cls.apply_guassian_filter(x)
            x = cls.remove_noise(x)
            x = cls.find_peak(x)

        return x

    @classmethod
    def postpare_output(cls, x):
        return torch.sigmoid(x)

    def forward(self, x):
        inp = self.prepare_input(x)
        out = self.backbone(inp)  # type: ignore
        return self.postpare_output(out)
