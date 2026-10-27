"""Small CNN on log-mel spectrograms, with optional SpecAugment. Owner: Bhagyashri.

Trains on the shared features written by preprocess_irmas.py, so the CNN sees exactly
the same clips, splits and subsets as the classical baseline:

    irmas_prep/mels.npy       (6705, 128, 130) float32 log-mels
    irmas_prep/manifest.csv   path, label, recording_id, split, pct_005 ... pct_100
                              (row i of the manifest is row i of mels.npy)

For every (train subset, seed) it trains a fresh CNN, early-stops on val macro-F1 and
appends one row to results/cnn/results.csv. The test split is scored only with
--eval_test (week 7).

Usage, from the repo root:
    python -m src.cnn.train_cnn                          # 5 subsets x 3 seeds, SpecAugment on
    python -m src.cnn.train_cnn --subsets 100 --seeds 0  # one quick run
    python -m src.cnn.train_cnn --no-specaug             # ablation: no SpecAugment
    python -m src.cnn.train_cnn --clip_sec 1             # ablation: 1 s inputs (also 2)
    python -m src.cnn.train_cnn --eval_test              # week 7 only: also score test

Design choices (all in the spirit of Han et al. 2017, scaled down):
  * 4 blocks of (conv3x3-BN-ReLU) x 2, widths 32-64-128-256, 2x2 max-pool between
    blocks, global max pooling, dropout 0.5, 11-way linear output (~1.2 M params).
  * Per-mel-bin standardization with mean/std computed on the *training subset only*.
  * Class-weighted cross-entropy ("balanced" weights), matching class_weight="balanced"
    in the SVM / Random Forest baseline.
  * AdamW, ReduceLROnPlateau on val macro-F1, early stopping on val macro-F1.
  * Small subsets are trained for at least --min_steps optimizer steps per epoch
    (by cycling the data), so the 5% subset isn't stopped after a handful of updates.
  * --clip_sec < 3: train on random crops of that length; at eval time, average the
    softmax over crops that cover the whole 3 s clip.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score

SUBSET_COLS = {5: "pct_005", 10: "pct_010", 25: "pct_025", 50: "pct_050", 100: "pct_100"}
CLIP_SECONDS = 3.0  # IRMAS training clips; preprocess_irmas.py pads/crops to exactly this


# --------------------------------------------------------------------------- data

def load_prep(prep_dir: Path):
    manifest = pd.read_csv(prep_dir / "manifest.csv")
    mels = np.load(prep_dir / "mels.npy", mmap_mode="r")
    if len(manifest) != len(mels):
        raise SystemExit(f"manifest.csv has {len(manifest)} rows but mels.npy has {len(mels)}")
    missing = [c for c in ["path", "label", "split", *SUBSET_COLS.values()] if c not in manifest]
    if missing:
        raise SystemExit(f"manifest.csv is missing columns {missing}; found {list(manifest.columns)}")
    return manifest, mels


def as_bool(col: pd.Series) -> np.ndarray:
    """Subset-membership columns may be saved as True/False, 1/0 or strings."""
    return col.astype(str).str.strip().str.lower().isin({"1", "1.0", "true", "yes"}).to_numpy()


def split_indices(manifest: pd.DataFrame, pct: int):
    split = manifest["split"].astype(str).str.lower().to_numpy()
    train = np.flatnonzero((split == "train") & as_bool(manifest[SUBSET_COLS[pct]]))
    val = np.flatnonzero(split == "val")
    test = np.flatnonzero(split == "test")
    return train, val, test


# --------------------------------------------------------------------------- model

def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, widths=(32, 64, 128, 256), dropout: float = 0.5):
        super().__init__()
        layers, c_in = [], 1
        for i, w in enumerate(widths):
            layers.append(conv_block(c_in, w))
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            c_in = w
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, n_mels, frames)
        h = self.features(x).amax(dim=(2, 3))  # global max pooling over freq and time
        return self.fc(self.dropout(h))


# --------------------------------------------------------------------------- augmentation

def _band_mask(B: int, size: int, max_width: int, device) -> torch.Tensor:
    """(B, size) boolean mask with one random band of width 0..max_width per sample."""
    width = torch.randint(0, max_width + 1, (B,), device=device)
    start = (torch.rand(B, device=device) * (size - width).clamp(min=0)).long()
    pos = torch.arange(size, device=device)[None]
    return (pos >= start[:, None]) & (pos < (start + width)[:, None])


def spec_augment(x: torch.Tensor, n_freq=2, max_f=15, n_time=2, max_t_frac=0.1) -> torch.Tensor:
    """SpecAugment (Park et al. 2019) frequency + time masking, independent per sample.

    Applied after standardization, so a masked value of 0 equals the training mean.
    """
    B, _, n_mels, T = x.shape
    keep = torch.ones(B, n_mels, T, dtype=torch.bool, device=x.device)
    for _ in range(n_freq):
        keep &= ~_band_mask(B, n_mels, max_f, x.device)[:, :, None]
    for _ in range(n_time):
        keep &= ~_band_mask(B, T, max(1, int(T * max_t_frac)), x.device)[:, None, :]
    return x * keep[:, None].to(x.dtype)


# --------------------------------------------------------------------------- helpers

def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_crop(x: torch.Tensor, win: int) -> torch.Tensor:
    """Independent random crop of `win` frames per sample. x: (B, 1, F, T)."""
    B, _, F, T = x.shape
    if win >= T:
        return x
    start = torch.randint(0, T - win + 1, (B,), device=x.device)
    idx = (start[:, None] + torch.arange(win, device=x.device)[None])[:, None, None, :]
    return x.gather(3, idx.expand(B, 1, F, win))


def eval_starts(T: int, win: int) -> list[int]:
    """Crop start frames that together cover the whole clip."""
    if win >= T:
        return [0]
    n = math.ceil(T / win)
    return sorted({int(round(s)) for s in np.linspace(0, T - win, n)})


@torch.no_grad()
def predict_proba(model, X, idx, norm, win, device, batch_size=256) -> np.ndarray:
    model.eval()
    out = []
    starts = eval_starts(X.shape[-1], win)
    for i in range(0, len(idx), batch_size):
        xb = norm(X[idx[i:i + batch_size]])
        probs = sum(torch.softmax(model(xb[..., s:s + win]).float(), dim=1) for s in starts)
        out.append((probs / len(starts)).cpu().numpy())
    return np.concatenate(out)


def cycled_order(n: int, total: int, gen: torch.Generator) -> torch.Tensor:
    """`total` indices into range(n): back-to-back random permutations, truncated."""
    reps = math.ceil(total / n)
    return torch.cat([torch.randperm(n, generator=gen) for _ in range(reps)])[:total]


# --------------------------------------------------------------------------- one run

def train_one(X, y, y_np, tr, va, te, n_classes, args, seed, device):
    set_seed(seed)
    T = X.shape[-1]
    win = max(8, round(T * args.clip_sec / CLIP_SECONDS))

    # Standardize per mel bin using the training subset only (no val/test leakage).
    Xtr = X[torch.as_tensor(tr, device=X.device)]
    mean = Xtr.mean(dim=(0, 1, 3), keepdim=True)
    std = Xtr.std(dim=(0, 1, 3), keepdim=True).clamp(min=1e-5)
    del Xtr

    def norm(xb):
        return ((xb - mean) / std).to(device, non_blocking=True)

    model = SmallCNN(n_classes, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    counts = np.bincount(y_np[tr], minlength=n_classes)
    if (counts == 0).any():
        print(f"    warning: classes with no training clips: {np.flatnonzero(counts == 0).tolist()}")
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    gen = torch.Generator().manual_seed(seed)
    tr_t = torch.as_tensor(tr)
    steps = max(math.ceil(len(tr) / args.batch_size), args.min_steps)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = tr_t[cycled_order(len(tr), steps * args.batch_size, gen)]
        running = 0.0
        for s in range(steps):
            bi = order[s * args.batch_size:(s + 1) * args.batch_size].to(X.device)
            xb = random_crop(norm(X[bi]), win)
            if args.specaug:
                xb = spec_augment(xb)
            yb = y[bi].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        val_pred = predict_proba(model, X, va, norm, win, device).argmax(1)
        val_f1 = f1_score(y_np[va], val_pred, average="macro", zero_division=0)
        scheduler.step(val_f1)
        improved = val_f1 > best["f1"] + 1e-4
        if improved:
            best = {"f1": val_f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if args.verbose:
            print(f"    epoch {epoch:3d}  loss {running / steps:.3f}  val macro-F1 {val_f1:.3f}"
                  f"{'  *' if improved else ''}")
        if bad_epochs >= args.patience:
            break

    train_time = time.time() - t0
    model.load_state_dict(best["state"])
    result = {"n_params": n_params, "best_epoch": best["epoch"], "epochs_run": epoch,
              "train_time_s": round(train_time, 1)}
    preds = {}
    for name, idx in [("val", va)] + ([("test", te)] if args.eval_test else []):
        p = predict_proba(model, X, idx, norm, win, device).argmax(1)
        result[f"{name}_macro_f1"] = round(f1_score(y_np[idx], p, average="macro", zero_division=0), 4)
        result[f"{name}_acc"] = round(accuracy_score(y_np[idx], p), 4)
        preds[name] = p
    return model, result, preds, {"mean": mean.cpu(), "std": std.cpu(), "win": win}


# --------------------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prep_dir", "--prep-dir", default="irmas_prep", type=Path)
    ap.add_argument("--out_dir", "--out-dir", default="results/cnn", type=Path)
    ap.add_argument("--ckpt_dir", "--ckpt-dir", default="checkpoints/cnn", type=Path,
                    help="model weights go here (gitignored: never push *.pt)")
    ap.add_argument("--subsets", type=int, nargs="+", default=[5, 10, 25, 50, 100],
                    choices=sorted(SUBSET_COLS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--specaug", action=argparse.BooleanOptionalAction, default=True,
                    help="SpecAugment on/off (--no-specaug for the ablation)")
    ap.add_argument("--clip_sec", type=float, default=3.0, help="input length in seconds: 1, 2 or 3")
    ap.add_argument("--epochs", type=int, default=60, help="maximum epochs")
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    ap.add_argument("--min_steps", type=int, default=50, help="minimum optimizer steps per epoch")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--eval_test", action="store_true", help="also score test (week 7 only)")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision on CUDA")
    ap.add_argument("--save_ckpt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--verbose", "-v", action="store_true", help="print every epoch")
    args = ap.parse_args()
    if not 0 < args.clip_sec <= CLIP_SECONDS:
        ap.error(f"--clip_sec must be in (0, {CLIP_SECONDS:g}]; IRMAS clips are 3 s long")
    return args


def main():
    args = parse_args()
    device = pick_device(args.device)
    manifest, mels = load_prep(args.prep_dir)

    classes = sorted(manifest["label"].astype(str).unique())
    y_np = manifest["label"].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    n_classes = len(classes)

    # Keep all clips in memory (about 450 MB for 6705 x 128 x 130 float32). On a GPU they
    # are moved there once, so batches never round-trip through the CPU.
    data_device = device if device.type == "cuda" else torch.device("cpu")
    X = torch.from_numpy(np.ascontiguousarray(mels, dtype=np.float32)).unsqueeze(1).to(data_device)
    y = torch.as_tensor(y_np, device=data_device)

    tag = f"specaug{int(args.specaug)}_clip{args.clip_sec:g}s"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "preds").mkdir(exist_ok=True)
    (args.out_dir / "classes.json").write_text(json.dumps(classes, indent=2))
    results_csv = args.out_dir / "results.csv"

    print(f"device={device}  clips={len(manifest)}  classes={n_classes}  input={tuple(X.shape[1:])}  run={tag}")
    for pct in args.subsets:
        tr, va, te = split_indices(manifest, pct)
        if len(tr) == 0:
            raise SystemExit(f"No training clips for the {pct}% subset; check {SUBSET_COLS[pct]} in manifest.csv")
        for seed in args.seeds:
            print(f"[{tag}] subset {pct:3d}%  n_train={len(tr)}  seed={seed}")
            model, res, preds, norm_stats = train_one(X, y, y_np, tr, va, te, n_classes, args, seed, device)

            row = {"model": "cnn", "run": tag, "subset_pct": pct, "seed": seed, "n_train": len(tr),
                   "specaug": int(args.specaug), "clip_sec": args.clip_sec, **res}
            # Merge rather than append: a later --eval_test run adds test columns that
            # earlier rows don't have, and a plain append would break the CSV header.
            old = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame()
            if len(old):  # a re-run of the same (run, subset, seed) replaces its old row
                old = old[~((old["run"] == tag) & (old["subset_pct"] == pct) & (old["seed"] == seed))]
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(results_csv, index=False)
            print(f"    val macro-F1 {res['val_macro_f1']:.3f}  acc {res['val_acc']:.3f}  "
                  f"best epoch {res['best_epoch']}/{res['epochs_run']}  {res['train_time_s']:.0f}s"
                  + (f"  | test macro-F1 {res['test_macro_f1']:.3f}" if args.eval_test else ""))

            stem = f"{tag}_p{pct:03d}_s{seed}"
            for split_name, p in preds.items():
                idx = va if split_name == "val" else te
                pd.DataFrame({"path": manifest["path"].iloc[idx].to_numpy(),
                              "label": [classes[i] for i in y_np[idx]],
                              "pred": [classes[i] for i in p]}
                             ).to_csv(args.out_dir / "preds" / f"{stem}_{split_name}.csv", index=False)
                if pct == 100:
                    report = classification_report(y_np[idx], p, target_names=classes, digits=3, zero_division=0)
                    (args.out_dir / f"report_{stem}_{split_name}.txt").write_text(report)
            if args.save_ckpt:
                args.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "classes": classes, **norm_stats,
                            "args": {k: str(v) for k, v in vars(args).items()}},
                           args.ckpt_dir / f"{stem}.pt")

    # Mean +/- std over seeds for this run configuration (all rows in results.csv with this tag).
    df = pd.read_csv(results_csv)
    df = df[df["run"] == tag]
    metric_cols = ["val_macro_f1"] + (["test_macro_f1"] if "test_macro_f1" in df and df["test_macro_f1"].notna().any() else [])
    summary = df.groupby("subset_pct")[metric_cols].agg(["mean", "std", "count"]).round(4)
    summary.to_csv(args.out_dir / f"summary_{tag}.csv")
    print("\nSummary over seeds:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
