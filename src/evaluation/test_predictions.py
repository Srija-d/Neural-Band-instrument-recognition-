"""
Make test-set predictions for the CNN and the classical baselines.
Owner: Bhagyashri (test metrics)

Every model's test predictions end up in the same simple format, one CSV per run:

    path, label, pred        (one row per test clip; label/pred are instrument codes)

evaluate_test.py then scores all of them the same way. Zero-shot and fine-tuned
models only need to write their test predictions in this format too.

Two commands
------------
  cnn        Load the CNN models we already trained (checkpoints/cnn/*.pt) and predict
             the test clips. No retraining, so this is exactly the model we chose on val.

  classical  Rebuild Aisha's SVM and Random Forest exactly as in baseline_mfcc.py
             (same features, same grid, picked by val macro-F1), then predict test.
             It prints the val macro-F1 so you can check it matches baseline_results/.

How to run (from the repo root)
-------------------------------
  python -m src.evaluation.test_predictions cnn --ckpt_dir checkpoints/cnn
  python -m src.evaluation.test_predictions classical

Both write into results/test/preds/ by default.

Only run this once everyone has finished tuning on val: the whole point of the
test set is that nobody changes their model after seeing it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.cnn.train_cnn import SmallCNN, load_prep, pick_device, predict_proba, split_indices


def save_preds(out_dir: Path, name: str, manifest, idx, true_codes, pred_codes):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    pd.DataFrame({
        "path": manifest["path"].iloc[idx].to_numpy(),
        "label": true_codes,
        "pred": pred_codes,
    }).to_csv(path, index=False)
    print(f"  wrote {path}")


# =============================================================================
# CNN: score the saved checkpoints on test
# =============================================================================

def run_cnn(args):
    device = pick_device(args.device)
    manifest, mels = load_prep(args.prep_dir)
    _, _, test_idx = split_indices(manifest, 100)
    labels = manifest["label"].astype(str).to_numpy()

    X = torch.from_numpy(np.array(mels, dtype=np.float32)).unsqueeze(1)

    pattern = f"{args.run}_p{args.subset:03d}_s*.pt"
    checkpoints = sorted(args.ckpt_dir.glob(pattern))
    if not checkpoints:
        raise SystemExit(f"No checkpoints matching {pattern} in {args.ckpt_dir}")
    print(f"device={device}  test clips={len(test_idx)}  checkpoints={len(checkpoints)}")

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        classes = ckpt["classes"]
        model = SmallCNN(len(classes))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        # Use the exact normalization and input length the model was trained with
        mean, std, win = ckpt["mean"], ckpt["std"], ckpt["win"]

        def normalize(batch):
            return ((batch - mean) / std).to(device)

        probs = predict_proba(model, X, test_idx, normalize, win, device)
        preds = [classes[i] for i in probs.argmax(axis=1)]
        seed = ckpt_path.stem.split("_s")[-1]
        save_preds(args.out_dir, f"cnn_seed{seed}", manifest, test_idx, labels[test_idx], preds)


# =============================================================================
# Classical: rebuild Aisha's SVM / Random Forest and predict test
# =============================================================================

def candidate_models():
    """The same grids as baseline_mfcc.py."""
    for C in (1, 10, 100):
        for gamma in ("scale", 0.01, 0.001):
            yield "svm", {"C": C, "gamma": gamma}, make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", random_state=42))
    for max_features in ("sqrt", 0.3):
        for leaf in (1, 2):
            yield "rf", {"max_features": max_features, "min_samples_leaf": leaf}, RandomForestClassifier(
                n_estimators=500, max_features=max_features, min_samples_leaf=leaf,
                class_weight="balanced", random_state=42, n_jobs=-1)


def run_classical(args):
    manifest = pd.read_csv(args.prep_dir / "manifest.csv")
    feats = np.load(args.prep_dir / "mfcc_feats.npy")
    labels = manifest["label"].astype(str).to_numpy()
    train_idx, val_idx, test_idx = split_indices(manifest, 100)

    # Pick the best settings for each model family on val, as the baseline did
    best = {}
    for family, params, model in candidate_models():
        model.fit(feats[train_idx], labels[train_idx])
        f1 = f1_score(labels[val_idx], model.predict(feats[val_idx]), average="macro")
        if family not in best or f1 > best[family][0]:
            best[family] = (f1, params, model)

    for family, (val_f1, params, model) in best.items():
        print(f"{family}: best {params}  val macro-F1 {val_f1:.3f}  "
              "(should match baseline_results/results.csv at 100%)")
        preds = model.predict(feats[test_idx])
        save_preds(args.out_dir, f"{family}_seed0", manifest, test_idx, labels[test_idx], preds)


# =============================================================================
# Command-line options
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Make test predictions (cnn / classical).")
    steps = parser.add_subparsers(dest="step", required=True)

    def shared(p):
        p.add_argument("--prep_dir", default="irmas_prep", type=Path)
        p.add_argument("--out_dir", default="results/test/preds", type=Path)

    p = steps.add_parser("cnn", help="score saved CNN checkpoints on test")
    shared(p)
    p.add_argument("--ckpt_dir", default="checkpoints/cnn", type=Path)
    p.add_argument("--run", default="specaug1_clip3s", help="which CNN setting to score")
    p.add_argument("--subset", type=int, default=100, help="training-set size the models were trained on")
    p.add_argument("--device", default="auto")

    p = steps.add_parser("classical", help="rebuild SVM / RF and predict test")
    shared(p)

    args = parser.parse_args()
    {"cnn": run_cnn, "classical": run_classical}[args.step](args)


if __name__ == "__main__":
    main()
