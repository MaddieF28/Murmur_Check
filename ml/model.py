"""Murmur model: 4 kHz heart audio in, murmur probability out.
Filtering and spectrogram are layers in the model so the ONNX file has everything."""
import math

import torch
import torch.nn.functional as F
from torch import nn

SR = 4000             # CirCor's native rate
WIN = 4 * SR          # 4 s windows, about 4-6 heartbeats
FRAME, HOP = 256, 40  # 64 ms frames every 10 ms
BANDS = torch.linspace(25, 400, 31)  # where most murmur energy is


class LogSpectrogram(nn.Module):
    """Log power in BANDS over time, minus each band's average (cancels mic/device coloring)."""

    def __init__(self, lo=25, hi=400, bands=31, frame=FRAME, hop=HOP, cmn=True):
        super().__init__()
        self.hop, self.cmn = hop, cmn
        # high-pass: subtract a 40 ms running average twice, kills the low rumble on phones
        box = torch.full((161,), 1 / 161)
        step = -box
        step[80] += 1
        highpass = F.conv1d(step.view(1, 1, -1), step.view(1, 1, -1), padding=160)
        self.register_buffer("highpass", highpass)

        # windowed sin/cos per band, so one conv gives the spectrogram
        n = torch.arange(frame)
        window = torch.hann_window(frame, periodic=False)
        phase = 2 * math.pi * torch.linspace(lo, hi, bands)[:, None] * n[None, :] / SR
        kernels = torch.cat([window * torch.cos(phase), window * torch.sin(phase)])
        self.register_buffer("kernels", kernels[:, None, :])

    def forward(self, audio):
        x = F.conv1d(audio[:, None, :], self.highpass)
        re, im = F.conv1d(x, self.kernels, stride=self.hop).chunk(2, dim=1)
        power = re ** 2 + im ** 2  # [batch, bands, frames]
        floor = 1e-4 * power.mean(dim=(1, 2), keepdim=True) + 1e-12
        logp = torch.log(power + floor)
        return logp - logp.mean(dim=2 if self.cmn else (1, 2), keepdim=True)


def freq_mixstyle(s, alpha=0.4):
    """Freq-MixStyle: mix band stats between windows so the model can't lean on one device's sound."""
    mu, sd = s.mean(2, keepdim=True), s.std(2, keepdim=True) + 1e-5
    perm = torch.randperm(len(s), device=s.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((len(s), 1, 1)).to(s.device)
    return (
        (s - mu) / sd * (lam * sd + (1 - lam) * sd[perm])
        + lam * mu + (1 - lam) * mu[perm]
    )


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2))


class MurmurNet(nn.Module):
    """CNN over the spectrogram, one logit per window. head is meanmax, attn or gru."""

    def __init__(self, widths=(16, 32, 64, 64), head="meanmax", dropout=0.3, spec=None, fms=0.0):
        super().__init__()
        self.fms = fms  # chance of freq_mixstyle per training batch
        self.spectrogram = LogSpectrogram(**(spec or {}))
        chans = (1, *widths)
        self.cnn = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans, chans[1:])])
        c, self.head_type = widths[-1], head
        if head == "gru":
            self.rnn = nn.GRU(c, c, batch_first=True, bidirectional=True)
            features = 4 * c
        elif head == "attn":
            self.attn = nn.Linear(c, 1)
            features = c
        else:
            features = 2 * c
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 1))

    def forward(self, audio):
        s = self.spectrogram(audio)
        if self.training and self.fms > 0 and torch.rand(1).item() < self.fms:
            s = freq_mixstyle(s)
        h = self.cnn(s[:, None]).mean(dim=2)  # pool frequency: [batch, ch, time]
        if self.head_type == "gru":
            out, _ = self.rnn(h.transpose(1, 2))
            h = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1)
        elif self.head_type == "attn":
            t = h.transpose(1, 2)
            h = (torch.softmax(self.attn(t), dim=1) * t).sum(dim=1)
        else:
            h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(h).squeeze(1)


class ExportedModel(nn.Module):
    """The version that goes in the app: outputs a probability instead of a logit."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, audio):
        return torch.sigmoid(self.net(audio))
