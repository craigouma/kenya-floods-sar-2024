"""U-Net for Sentinel-1 water segmentation, with Sen1Floods11 input scaling."""

import numpy as np
import torch
from torch import nn

# Sen1Floods11 reference code: clip dB to [-50, 1], rescale to [0, 1], then
# standardise with statistics of the hand-labelled training chips (VV, VH).
DB_MIN, DB_MAX = -50.0, 1.0
MEAN = np.array([0.6851, 0.5235], dtype="float32")
STD = np.array([0.0820, 0.1102], dtype="float32")


def scale(db: np.ndarray) -> np.ndarray:
    """(bands, H, W) dB in VV, VH order (repeated for pre-event) to network input."""
    x = (np.clip(np.nan_to_num(db, nan=DB_MIN), DB_MIN, DB_MAX) - DB_MIN) / (DB_MAX - DB_MIN)
    reps = x.shape[0] // 2
    return ((x - np.tile(MEAN, reps)[:, None, None]) / np.tile(STD, reps)[:, None, None]).astype("float32")


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 32, depth: int = 4):
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_channels, widths[0])])
        self.down.extend(_block(widths[i], widths[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(depth)))
        self.merge = nn.ModuleList(_block(widths[i] * 2, widths[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], 2, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, merge in zip(self.up, self.merge):
            x = merge(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


@torch.no_grad()
def predict(model: nn.Module, db: np.ndarray, tile: int = 512, overlap: int = 64,
            valid: np.ndarray | None = None) -> np.ndarray:
    """Water probability for an arbitrarily sized scene by overlapping tiles,
    skipping tiles with no valid pixel."""
    model.eval()
    device = next(model.parameters()).device
    x = scale(db)
    _, h, w = x.shape
    ph, pw = max(h, tile), max(w, tile)
    x = np.pad(x, ((0, 0), (0, ph - h), (0, pw - w)), mode="reflect")
    keep = np.pad(np.ones((h, w), bool) if valid is None else valid, ((0, ph - h), (0, pw - w)))
    prob = np.zeros((ph, pw), "float32")
    weight = np.zeros((ph, pw), "float32")
    step = tile - overlap
    rows = sorted({*range(0, ph - tile + 1, step), ph - tile})
    cols = sorted({*range(0, pw - tile + 1, step), pw - tile})
    for r in rows:
        for c in cols:
            if not keep[r : r + tile, c : c + tile].any():
                continue
            t = torch.from_numpy(x[None, :, r : r + tile, c : c + tile]).to(device)
            p = torch.softmax(model(t), dim=1)[0, 1].cpu().numpy()
            prob[r : r + tile, c : c + tile] += p
            weight[r : r + tile, c : c + tile] += 1
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(prob / weight)[:h, :w]
