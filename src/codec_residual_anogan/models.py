"""Convolutional generator and spectrally normalized discriminator."""
import torch.nn as nn


class ConvGenerator(nn.Module):
    def __init__(self, output_dim, latent_dim=100, base_channels=16):
        super().__init__()
        base = int(base_channels)
        self.latent_dim = int(latent_dim)
        self.input_linear = nn.Linear(self.latent_dim, base * 16 * 16)
        self.convs = nn.Sequential(
            nn.ConvTranspose1d(base * 16, base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(base * 8), nn.ReLU(True),
            nn.ConvTranspose1d(base * 8, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(base * 4), nn.ReLU(True),
            nn.ConvTranspose1d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(base * 2), nn.ReLU(True),
            nn.ConvTranspose1d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(True),
            nn.ConvTranspose1d(base, 1, 4, 2, 1, bias=False),
        )
        self.output_linear = nn.Linear(512, int(output_dim))

    def forward(self, latent):
        values = self.input_linear(latent).view(latent.shape[0], -1, 16)
        return self.output_linear(self.convs(values).squeeze(1))


class ConvDiscriminator(nn.Module):
    def __init__(self, input_dim, base_channels=16):
        super().__init__()
        base = int(base_channels)
        self.input_linear = nn.Linear(int(input_dim), 512)
        self.convs = nn.Sequential(
            nn.Conv1d(1, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(base * 2), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(base * 4), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 4, base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(base * 8), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 8, base * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(base * 16), nn.LeakyReLU(0.2, True),
        )
        self.output_linear = nn.Linear(base * 16 * 16, 1)

    def feature(self, values):
        projected = self.input_linear(values).unsqueeze(1)
        return self.convs(projected).flatten(1)

    def forward(self, values):
        return self.output_linear(self.feature(values)).squeeze(1)


def _initialize(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _spectral_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Conv1d, nn.Linear)):
            setattr(module, name, nn.utils.spectral_norm(child))
        else:
            _spectral_norm(child)


def build_models(input_dim, config, device):
    if config.get("architecture", "conv_baseline") != "conv_baseline":
        raise ValueError("this repository supports the best conv_baseline only")
    generator = ConvGenerator(
        input_dim, config.get("latent_dim", 100),
        config.get("base_channels", 16))
    discriminator = ConvDiscriminator(
        input_dim, config.get("base_channels", 16))
    generator.apply(_initialize)
    discriminator.apply(_initialize)
    if bool(config.get("spectral_norm", True)):
        _spectral_norm(discriminator)
    return generator.to(device), discriminator.to(device)
