"""Linear probe and fine-tuning of AST / PANNs / CLAP on IRMAS. Owner: Srija.

Two modes, both run on every training subset (5/10/25/50/100%) for the data-scarcity curves:

  probe     Frozen backbone. Embeddings for all clips are extracted once and cached
            (cache/foundation/<model>_feats.npz). A StandardScaler + logistic regression
            (class_weight="balanced") is trained on them; C is chosen on val macro-F1.
            Logistic regression is deterministic, so it runs once per subset (seed 0).

  finetune  New 11-way head on top of the pretrained backbone. The top --unfreeze blocks
            are trained together with the head (0 = head only, -1 = whole model):
              AST    12 transformer layers  (+ final layer norm)
              PANNs  6 conv blocks of CNN14 (+ fc1; CNN14's built-in SpecAugment is on)
              CLAP   4 HTS-AT stages        (+ final norm and audio projection)
            Same recipe as the CNN: class-weighted cross-entropy, AdamW, ReduceLROnPlateau
            and early stopping on val macro-F1, 3 seeds.

Usage, from the repo root:
    python -m src.foundation.finetune.train_finetune --mode probe                        # all models
    python -m src.foundation.finetune.train_finetune --mode finetune --models ast \\
           --subsets 100 --seeds 0                                                      # one quick run
    python -m src.foundation.finetune.train_finetune --mode finetune --models panns --unfreeze -1
    python -m src.foundation.finetune.train_finetune --mode probe --eval_test            # week 7 only

Writes to results/foundation/finetune/ (same layout as results/cnn/):
    results.csv, summary_<model>_<run>.csv, preds/<model>_<run>_pXXX_sN_<split>.csv,
    report_<model>_<run>_p100_sN_<split>.txt
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLASS_CODES, IRMAS_TRAIN_DIR
from src.foundation.common import (DEFAULT_CACHE, MODEL_SR, MODELS, SUBSET_COLS, get_features, labels_to_ids,
                                   load_backbone, load_manifest, load_waveforms, pick_device,
                                   set_seed, split_indices)

DEFAULT_LR = {"ast": 2e-5, "panns": 1e-4, "clap": 2e-5}   # backbone learning rates
N_CLASSES = len(CLASS_CODES)


# --------------------------------------------------------------------------- linear probe

def run_probe(emb, y, tr, va, te, args):
    best = None
    for C in args.probe_C:
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, class_weight="balanced", max_iter=3000))
        clf.fit(emb[tr], y[tr])
        f1 = f1_score(y[va], clf.predict(emb[va]), average="macro", zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    preds = {"val": clf.predict(emb[va])}
    if args.eval_test:
        preds["test"] = clf.predict(emb[te])
    return {"best_C": C}, preds


# --------------------------------------------------------------------------- fine-tuning

class Classifier(nn.Module):
    def __init__(self, backbone, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.embed_dim, N_CLASSES))

    def forward(self, **x):
        emb, _ = self.backbone(**x)
        return self.head(emb.float())


def freeze(model: Classifier, n_unfreeze: int) -> int:
    """Train the head + the top `n_unfreeze` blocks (-1 = everything). Returns #trainable params."""
    bb = model.backbone
    blocks = bb.blocks if n_unfreeze < 0 else bb.blocks[len(bb.blocks) - n_unfreeze:] if n_unfreeze else []
    prefixes = ["backbone." + p for p in blocks + (bb.always_train if n_unfreeze else [])]
    for name, p in model.named_parameters():
        p.requires_grad = (n_unfreeze < 0 and name.startswith("backbone.")) or name.startswith("head.") \
            or any(name.startswith(pre) for pre in prefixes)
    if n_unfreeze < 0 and hasattr(bb.net, "text_model"):   # CLAP: never train the text tower
        for p in bb.net.text_model.parameters():
            p.requires_grad = False
        for p in bb.net.text_projection.parameters():
            p.requires_grad = False
    names = [n for n, _ in model.named_parameters()]
    unmatched = [pre for pre in prefixes if not any(n.startswith(pre) for n in names)]
    if unmatched:
        raise SystemExit(f"{bb.name}: no parameters start with {unmatched}; check Backbone.blocks")
    n_bb = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone."))
    if n_unfreeze and n_bb == 0:
        raise SystemExit(f"--unfreeze {n_unfreeze} matched no {bb.name} parameters; check Backbone.blocks")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_mode(model: nn.Module):
    """train() for the trainable part; frozen leaves (e.g. BatchNorm) stay in eval mode so their
    running statistics are not changed. Parameter-free layers (dropout, SpecAugment) stay in train."""
    model.train()
    for m in model.modules():
        params = list(m.parameters(recurse=False))
        if params and not any(p.requires_grad for p in params):
            m.eval()


def batches(waves, idx, backbone, batch_size, device):
    for i in range(0, len(idx), batch_size):
        x = backbone.prepare(np.asarray(waves[np.sort(idx[i:i + batch_size])], dtype=np.float32))
        yield np.sort(idx[i:i + batch_size]), {k: v.to(device, non_blocking=True) for k, v in x.items()}


@torch.no_grad()
def predict(model, waves, idx, device, batch_size, use_amp):
    model.eval()
    out = np.empty(len(idx), dtype=np.int64)
    pos = {j: k for k, j in enumerate(idx)}
    for bi, x in batches(waves, idx, model.backbone, batch_size, device):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            p = model(**x).argmax(1).cpu().numpy()
        out[[pos[j] for j in bi]] = p
    return out


def run_finetune(name, waves, y, tr, va, te, args, seed, device):
    set_seed(seed)
    model = Classifier(load_backbone(name)).to(device)
    n_trainable = freeze(model, args.unfreeze)
    use_amp = device.type == "cuda" and not args.no_amp

    counts = np.bincount(y[tr], minlength=N_CLASSES)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    lr = args.lr or DEFAULT_LR[name]
    optimizer = torch.optim.AdamW([{"params": head, "lr": args.lr_head}] + ([{"params": body, "lr": lr}] if body else []),
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rng = np.random.default_rng(seed)
    steps = max(math.ceil(len(tr) / args.batch_size), args.min_steps)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    bad, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        train_mode(model)
        order = np.concatenate([rng.permutation(tr) for _ in range(math.ceil(steps * args.batch_size / len(tr)))])
        running = 0.0
        for s in range(steps):
            bi = order[s * args.batch_size:(s + 1) * args.batch_size]
            x = model.backbone.prepare(np.asarray(waves[np.sort(bi)], dtype=np.float32))
            x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
            yb = torch.as_tensor(y[np.sort(bi)], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(**x).float(), yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        val_f1 = f1_score(y[va], predict(model, waves, va, device, args.eval_batch_size, use_amp),
                          average="macro", zero_division=0)
        scheduler.step(val_f1)
        improved = val_f1 > best["f1"] + 1e-4
        if improved:
            best = {"f1": val_f1, "epoch": epoch,
                    "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        print(f"    epoch {epoch:3d}  loss {running / steps:.3f}  val macro-F1 {val_f1:.3f}"
              f"{'  *' if improved else ''}  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= args.patience:
            break

    model.load_state_dict(best["state"])
    res = {"n_trainable": n_trainable, "best_epoch": best["epoch"], "epochs_run": epoch,
           "train_time_s": round(time.time() - t0, 1), "lr": lr}
    preds = {"val": predict(model, waves, va, device, args.eval_batch_size, use_amp)}
    if args.eval_test:
        preds["test"] = predict(model, waves, te, device, args.eval_batch_size, use_amp)
    if args.save_ckpt:  # only the trainable weights, to keep files small (gitignored anyway)
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"state_dict": {k: v for k, v in model.state_dict().items() if k in keep},
                    "model": name, "unfreeze": args.unfreeze, "classes": CLASS_CODES},
                   args.ckpt_dir / f"{name}_{args.tag}_p{args.cur_pct:03d}_s{seed}.pt")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, preds


# --------------------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--subsets", type=int, nargs="+", default=[5, 10, 25, 50, 100], choices=sorted(SUBSET_COLS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prep_dir", default="irmas_prep", type=Path)
    ap.add_argument("--data_dir", default=IRMAS_TRAIN_DIR, type=Path)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE, type=Path)
    ap.add_argument("--out_dir", default="results/foundation/finetune", type=Path)
    ap.add_argument("--ckpt_dir", default="checkpoints/foundation", type=Path)
    ap.add_argument("--save_ckpt", action="store_true", help="save fine-tuned weights (never push them)")
    # probe
    ap.add_argument("--probe_C", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    # fine-tuning
    ap.add_argument("--unfreeze", type=int, default=2, help="top blocks to train: 0 = head only, -1 = all")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min_steps", type=int, default=30, help="minimum optimizer steps per epoch")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="backbone lr (default: ast/clap 2e-5, panns 1e-4)")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--eval_test", action="store_true", help="also score test (week 7 only)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.tag = "probe" if args.mode == "probe" else ("ft_all" if args.unfreeze < 0 else f"ft_top{args.unfreeze}")
    if args.mode == "probe":
        args.seeds = [0]   # logistic regression is deterministic
    return args


def main():
    args = parse_args()
    device = pick_device(args.device)
    manifest = load_manifest(args.prep_dir, args.data_dir)
    y = labels_to_ids(manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "preds").mkdir(exist_ok=True)
    results_csv = args.out_dir / "results.csv"
    print(f"device={device}  clips={len(manifest)}  mode={args.mode}  run={args.tag}")

    for name in args.models:
        if args.mode == "probe":
            emb = get_features(name, manifest, device, args.data_dir, args.cache_dir)["emb"]
        else:
            waves = load_waveforms(manifest, MODEL_SR[name], args.data_dir, args.cache_dir)

        for pct in args.subsets:
            tr, va, te = split_indices(manifest, pct)
            args.cur_pct = pct
            for seed in args.seeds:
                print(f"[{name} {args.tag}] subset {pct:3d}%  n_train={len(tr)}  seed={seed}", flush=True)
                if args.mode == "probe":
                    res, preds = run_probe(emb, y, tr, va, te, args)
                else:
                    res, preds = run_finetune(name, waves, y, tr, va, te, args, seed, device)

                row = {"model": name, "run": args.tag, "subset_pct": pct, "seed": seed, "n_train": len(tr), **res}
                stem = f"{name}_{args.tag}_p{pct:03d}_s{seed}"
                for split, p in preds.items():
                    idx = va if split == "val" else te
                    row[f"{split}_macro_f1"] = round(f1_score(y[idx], p, average="macro", zero_division=0), 4)
                    row[f"{split}_acc"] = round(accuracy_score(y[idx], p), 4)
                    pd.DataFrame({"path": manifest["path"].iloc[idx].to_numpy(),
                                  "label": [CLASS_CODES[i] for i in y[idx]],
                                  "pred": [CLASS_CODES[i] for i in p]}
                                 ).to_csv(args.out_dir / "preds" / f"{stem}_{split}.csv", index=False)
                    if pct == 100:
                        rep = classification_report(y[idx], p, labels=range(N_CLASSES), target_names=CLASS_CODES,
                                                    digits=3, zero_division=0)
                        (args.out_dir / f"report_{stem}_{split}.txt").write_text(rep)
                print(f"    val macro-F1 {row['val_macro_f1']:.3f}  acc {row['val_acc']:.3f}"
                      + (f"  | test macro-F1 {row['test_macro_f1']:.3f}" if args.eval_test else ""), flush=True)

                old = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame()
                if len(old):  # a re-run of the same (model, run, subset, seed) replaces its old row
                    old = old[~((old["model"] == name) & (old["run"] == args.tag)
                                & (old["subset_pct"] == pct) & (old["seed"] == seed))]
                pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(results_csv, index=False)

        df = pd.read_csv(results_csv)
        df = df[(df["model"] == name) & (df["run"] == args.tag)]
        cols = ["val_macro_f1"] + (["test_macro_f1"] if "test_macro_f1" in df and df["test_macro_f1"].notna().any() else [])
        summary = df.groupby("subset_pct")[cols].agg(["mean", "std", "count"]).round(4)
        summary.to_csv(args.out_dir / f"summary_{name}_{args.tag}.csv")
        print(f"\n{name} {args.tag}: mean over seeds\n{summary.to_string()}\n")


if __name__ == "__main__":
    main()
