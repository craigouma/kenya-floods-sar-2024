"""Train the Sen1Floods11 U-Nets and score them on the Sen1Floods11 test splits.

Follows the training setup of Bonafilia et al. (2020): random 256 x 256 crops
with flips, cross-entropy weighted 1:8 toward water, AdamW (lr 5e-4, weight
decay 1e-2), cosine annealing with warm restarts every 10 epochs doubling, and
the checkpoint with the best mean per-chip validation IoU. Bolivia is held out
completely, including its weakly labelled chips.

Self-contained so it runs as a Kaggle GPU kernel:
    python train_unet.py --data /kaggle/working/s1f --out /kaggle/working/models
No Kenyan data is read here.
"""

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.nn import functional as F

from unet import UNet, scale

BUCKET = "sen1floods11/v1.1"
GCS_FOLDERS = {
    "HandLabeled/S1Hand": "data/flood_events/HandLabeled/S1Hand",
    "HandLabeled/LabelHand": "data/flood_events/HandLabeled/LabelHand",
    "HandLabeled/JRCWaterHand": "data/flood_events/HandLabeled/JRCWaterHand",
    "WeaklyLabeled/S1Weak": "data/flood_events/WeaklyLabeled/S1Weak",
    "WeaklyLabeled/S2IndexLabelWeak": "data/flood_events/WeaklyLabeled/S2IndexLabelWeak",
    "splits/flood_handlabeled": "splits/flood_handlabeled",
}
IGNORE = 255
CROP = 256


def fetch(root: Path) -> None:
    import gcsfs

    fs = gcsfs.GCSFileSystem(token="anon")
    jobs = []
    for local, remote in GCS_FOLDERS.items():
        (root / local).mkdir(parents=True, exist_ok=True)
        for r in fs.ls(f"{BUCKET}/{remote}"):
            target = root / local / r.rsplit("/", 1)[1]
            if not target.exists():
                jobs.append((r, str(target)))
    with ThreadPoolExecutor(32) as pool:
        list(pool.map(lambda j: fs.get(*j), jobs))


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype("float32")


def label(path: Path) -> np.ndarray:
    y = read(path)[0]
    return np.where(y < 0, IGNORE, y).astype("uint8")


def load(pairs: list[tuple[Path, ...]], label_path: callable) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for inputs in pairs:
        xs.append(scale(np.concatenate([read(p) for p in inputs])).astype("float16"))
        ys.append(label(label_path(inputs[0])))
    return np.stack(xs), np.stack(ys)


def split(root: Path, name: str) -> list[str]:
    with open(root / "splits" / "flood_handlabeled" / f"flood_{name}_data.csv") as f:
        return [row[0].removesuffix("_S1Hand.tif") for row in csv.reader(f)]


def hand_inputs(root: Path, chip: str, pre_dir: Path | None) -> tuple[Path, ...]:
    post = root / "HandLabeled" / "S1Hand" / f"{chip}_S1Hand.tif"
    return (post, pre_dir / f"{chip}_S1PreHand.tif") if pre_dir else (post,)


def hand_label(root: Path):
    return lambda p: root / "HandLabeled" / "LabelHand" / p.name.replace("S1Hand", "LabelHand")


def batches(x: np.ndarray, y: np.ndarray, batch: int, rng: np.random.Generator):
    order = rng.permutation(len(x))
    for i in range(0, len(order) - batch + 1, batch):
        idx = order[i : i + batch]
        r, c = rng.integers(0, x.shape[-2] - CROP + 1, 2)
        xb = x[idx, :, r : r + CROP, c : c + CROP].astype("float32")
        yb = y[idx, r : r + CROP, c : c + CROP]
        if rng.random() > 0.5:
            xb, yb = xb[..., ::-1], yb[..., ::-1]
        if rng.random() > 0.5:
            xb, yb = xb[..., ::-1, :], yb[..., ::-1, :]
        yield torch.from_numpy(xb.copy()), torch.from_numpy(yb.astype("int64").copy())


@torch.no_grad()
def predict(model, x: np.ndarray, device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), 8):
        xb = torch.from_numpy(x[i : i + 8].astype("float32")).to(device)
        out.append(model(xb).argmax(1).cpu().numpy().astype("uint8"))
    return np.concatenate(out)


def iou_scores(pred: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Pooled IoU over all pixels, and the mean of per-chip IoUs (the
    Sen1Floods11 paper's convention, chips with an empty union skipped)."""
    valid = y != IGNORE if mask is None else (y != IGNORE) & mask
    p, t = (pred == 1) & valid, (y == 1) & valid
    inter, union = (p & t).sum((1, 2)), (p | t).sum((1, 2))
    per_chip = inter[union > 0] / union[union > 0]
    return {"pooled_iou": float(inter.sum() / union.sum()), "mean_chip_iou": float(per_chip.mean()),
            "n_chips": int(len(y))}


def train(name, x, y, xv, yv, args, device) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = UNet(x.shape[1], args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    steps = len(x) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=steps * 10, T_mult=2)
    weight = torch.tensor([1.0, 8.0], device=device)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in batches(x, y, args.batch, rng):
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb, weight=weight, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        val = iou_scores(predict(model, xv, device), yv)
        history.append({"epoch": epoch, "loss": total / steps, **val, "seconds": time.time() - t0})
        print(name, history[-1], flush=True)
        if val["mean_chip_iou"] > best:
            best = val["mean_chip_iou"]
            torch.save(model.state_dict(), args.out / f"{name}.pt")
    model.load_state_dict(torch.load(args.out / f"{name}.pt", map_location=device))
    return {"history": history, "best_val_mean_chip_iou": best, "model": model}


def evaluate(model, root, pre_dir, device) -> dict:
    out = {}
    for name in ("test", "bolivia"):
        chips = split(root, name)
        x, y = load([hand_inputs(root, c, pre_dir) for c in chips], hand_label(root))
        perm = np.stack([read(root / "HandLabeled" / "JRCWaterHand" / f"{c}_JRCWaterHand.tif")[0] == 1 for c in chips])
        pred = predict(model, x, device)
        out[name] = {"all_water": iou_scores(pred, y), "flood_water": iou_scores(pred, y, ~perm)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pre-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", default=["post_hand_weak", "post_hand", "pre_hand"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fetch(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data
    results = {}
    for run in args.runs:
        pre_dir = args.pre_dir if run.startswith("pre") else None
        x, y = load([hand_inputs(root, c, pre_dir) for c in split(root, "train")], hand_label(root))
        if run.endswith("hand_weak"):
            weak = [p for p in sorted((root / "WeaklyLabeled" / "S1Weak").glob("*_S1Weak.tif"))
                    if not p.name.startswith("Bolivia_")]
            xw, yw = load([(p,) for p in weak],
                          lambda p: root / "WeaklyLabeled" / "S2IndexLabelWeak" / p.name.replace("S1Weak", "S2IndexLabelWeak"))
            x, y = np.concatenate([x, xw]), np.concatenate([y, yw])
        xv, yv = load([hand_inputs(root, c, pre_dir) for c in split(root, "valid")], hand_label(root))
        fit = train(run, x, y, xv, yv, args, device)
        results[run] = {"n_train_chips": int(len(x)), "in_channels": int(x.shape[1]), "base": args.base,
                        "epochs": args.epochs, "best_val_mean_chip_iou": fit["best_val_mean_chip_iou"],
                        **evaluate(fit["model"], root, pre_dir, device)}
        (args.out / f"{run}_history.json").write_text(json.dumps(fit["history"], indent=1))
        (args.out / "sen1floods11_results.json").write_text(json.dumps(results, indent=1))
        print(run, json.dumps({k: v for k, v in results[run].items() if k in ("test", "bolivia")}), flush=True)


if __name__ == "__main__":
    main()
