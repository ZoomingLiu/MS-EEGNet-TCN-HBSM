"""Window EEGNet encoder and minute-level causal TCN, supervised only at the endpoint."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ForecastConfig


class WindowEEGNet(nn.Module):
    """Preserve the notebook's 8->16 EEGNet topology; replace BatchNorm with per-window GroupNorm.

    Each completed 5-second window is a local observation, not a state-classification target.
    GroupNorm never pools across windows/batch examples and permits encoder micro-batches.
    This deliberate normalization change means old notebook checkpoints are NOT drop-in compatible.
    """

    feature_dim = 16

    def __init__(self, channels=22, dropout=0.25):
        super().__init__()
        self.channels = channels
        self.temporal_block = nn.Sequential(nn.Conv2d(1, 8, (1, 64), bias=False), nn.GroupNorm(1, 8))
        self.spatial_block = nn.Sequential(
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False), nn.GroupNorm(1, 16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable_block_1 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), groups=16, padding=(0, 8), bias=False),
            nn.Conv2d(16, 16, 1, bias=False), nn.GroupNorm(1, 16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable_block_2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), groups=16, padding=(0, 4), bias=False),
            nn.Conv2d(16, 16, 1, bias=False), nn.GroupNorm(1, 16), nn.ELU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(dropout))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError("Window encoder expects [windows,channels,samples]")
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = ((x - mean) / std).unsqueeze(1)
        return self.separable_block_2(self.separable_block_1(self.spatial_block(self.temporal_block(x)))).flatten(1)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Conv1d):
    def __init__(self, channels, dilation):
        super().__init__(channels, channels, kernel_size=3, dilation=dilation)
        self.left_padding = 2 * dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(channels, dilation), ChannelLayerNorm(channels), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, dilation), ChannelLayerNorm(channels), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return F.gelu(x + self.layers(x))


class MinuteTCN(nn.Module):
    def __init__(self, channels=64, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.receptive_field = 1 + 4 * sum(dilations)
        self.blocks = nn.Sequential(*(TemporalBlock(channels, d, dropout) for d in dilations))

    def forward(self, x):
        return self.blocks(x)


class LongContextForecaster(nn.Module):
    def __init__(self, config: ForecastConfig):
        super().__init__()
        self.config = config
        self.encoder = WindowEEGNet(len(config.channel_names), config.encoder_dropout)
        self.morphology_projection = nn.Sequential(nn.Linear(48, 64), nn.GELU())
        self.power_projection = nn.Sequential(nn.Linear(config.power_dim, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(96, config.hidden_dim), nn.GELU())
        self.tcn = MinuteTCN(config.hidden_dim, config.dilations, config.dropout)
        self.head = nn.Sequential(nn.Linear(config.hidden_dim, 32), nn.GELU(), nn.Linear(32, 1))
        if config.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_encoder:
            self.encoder.eval()
        return self

    def minute_tokens(self, windows, power):
        c = self.config
        expected = (c.history_minutes, 60 // c.window_seconds, len(c.channel_names), c.sample_rate * c.window_seconds)
        if windows.ndim != 5 or tuple(windows.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{','.join(map(str, expected))}], got {tuple(windows.shape)}")
        b, m, k, channels, samples = windows.shape
        if tuple(power.shape) != (b, m, c.power_dim):
            raise ValueError("Power features are not aligned to input minutes")
        if c.branches != "power":
            encoded = []
            for chunk in windows.reshape(-1, channels, samples).split(c.encoder_chunk_size):
                if c.checkpoint_encoder and self.training and not c.freeze_encoder:
                    encoded.append(checkpoint(self.encoder, chunk, use_reentrant=False))
                else:
                    encoded.append(self.encoder(chunk))
            features = torch.cat(encoded).reshape(b, m, k, 16)
            summary = torch.cat([features.mean(2), features.std(2, unbiased=False), features[:, :, -1]], dim=-1)
            morphology = self.morphology_projection(summary)
        else:
            morphology = power.new_zeros((b, m, 64))
        amplitudes = self.power_projection(power) if c.branches != "morphology" else power.new_zeros((b, m, 32))
        return self.fusion(torch.cat([morphology, amplitudes], dim=-1)).transpose(1, 2)

    def forward(self, windows, power):
        """Return one raw logit per endpoint; use BCEWithLogitsLoss, not sigmoid + BCE."""
        return self.head(self.tcn(self.minute_tokens(windows, power))[:, :, -1]).squeeze(-1)
