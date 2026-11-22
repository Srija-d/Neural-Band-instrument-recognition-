"""
Classical baseline for Neural Band: MFCC features + SVM / Random Forest.

Uses the shared preprocessed features from preprocess_irmas.py (irmas_prep/manifest.csv
+ mfcc_feats.npy or classical_feats.npy), so it sees exactly the same cleaned clips as
every other model.

  --features mfcc       40-d pooled MFCCs (original baseline)
  --features classical  170-d MFCC + deltas + spectral + chroma (stronger baseline)
  --split random        split ablation: leaky clip-level split (100% train only)
For every training subset (5, 10, 25, 50, 100%):
  1. trains SVM and RF candidates on the subset,
  2. picks the best hyperparameters by macro-F1 on the val split,
  3. reports macro-F1 on val (and on test only when --eval_test is passed).

The test set is touched once, at the end of the project (week 7) -- leave
--eval_test off until then.

Outputs (in --out_dir):
  results.csv             model, subset, best params, val macro-F1 (+ test macro-F1)
  report_<model>.txt      per-class precision/recall/F1 on the eval split (100% subset)
  confusion_<model>.png   eval-split confusion matrix (100% subset)

Usage:
  python baseline_mfcc.py --prep_dir irmas_prep
  python baseline_mfcc.py --prep_dir irmas_prep --eval_test   # week 7 only
  python baseline_mfcc.py --features classical --out_dir baseline_results/classical
  python baseline_mfcc.py --features classical --split random \
      --out_dir baseline_results/classical_random_split
"""

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
SUBSETS = ["005", "010", "025", "050", "100"]
FEATURE_FILES = {"mfcc": "mfcc_feats.npy", "classical": "classical_feats.npy"}

GRIDS = {
    "svm": {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.001]},
    "rf": {"max_features": ["sqrt", 0.3], "min_samples_leaf": [1, 2]},
}


def make_model(name, params):
    if name == "svm":
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", class_weight="balanced", random_state=SEED, **params))
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED, **params)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", default="irmas_prep",
                    help="folder with manifest.csv and feature arrays from preprocess_irmas.py")
    ap.add_argument("--features", choices=FEATURE_FILES, default="mfcc")
    ap.add_argument("--split", choices=["recording", "random"], default="recording",
                    help="random = leaky clip-level split, for the split ablation only")
    ap.add_argument("--out_dir", default="baseline_results")
    ap.add_argument("--eval_test", action="store_true",
                    help="also score the test split (final evaluation only)")
    args = ap.parse_args()

    prep = Path(args.prep_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(prep / "manifest.csv")
    feats = np.load(prep / FEATURE_FILES[args.features])
    assert len(manifest) == len(feats), "manifest.csv and features are not row-aligned"
    y_all = manifest.label.values
    split = manifest.split if args.split == "recording" else manifest.split_random
    # Subsets are drawn from the recording-aware train split, so the random split uses 100% only.
    subsets = SUBSETS if args.split == "recording" else ["100"]
    tag = f"{args.features.upper()}{' (random split)' if args.split == 'random' else ''}"

    va = (split == "val").values
    te = (split == "test").values
    Xva, yva = feats[va], y_all[va]
    Xte, yte = feats[te], y_all[te]
    labels = sorted(manifest.label.unique())
    eval_name = "test" if args.eval_test else "val"

    rows = []
    for model_name, grid in GRIDS.items():
        for s in subsets:
            tr = (split == "train").values
            if args.split == "recording":
                tr = tr & manifest[f"pct_{s}"].values
            Xtr, ytr = feats[tr], y_all[tr]
            best = None
            for combo in itertools.product(*grid.values()):
                params = dict(zip(grid, combo))
                m = make_model(model_name, params).fit(Xtr, ytr)
                pred_va = m.predict(Xva)
                f1 = f1_score(yva, pred_va, average="macro")
                if best is None or f1 > best[0]:
                    best = (f1, params, m, pred_va)
            val_f1, params, m, pred_va = best
            row = dict(model=model_name, features=args.features, split=args.split,
                       subset_pct=int(s), n_train=int(tr.sum()),
                       params=params, val_macro_f1=round(val_f1, 4))
            msg = f"{model_name:>3} | {int(s):>3}% train | val F1 {val_f1:.3f}"
            y_eval, pred = yva, pred_va
            if args.eval_test:
                y_eval, pred = yte, m.predict(Xte)
                row["test_macro_f1"] = round(f1_score(yte, pred, average="macro"), 4)
                msg += f" | test F1 {row['test_macro_f1']:.3f}"
            print(f"{msg} | {params}")
            rows.append(row)

            if s == "100":
                eval_f1 = f1_score(y_eval, pred, average="macro")
                (out / f"report_{model_name}.txt").write_text(
                    f"{tag} + {model_name.upper()}, 100% train, evaluated on {eval_name}\n\n"
                    + classification_report(y_eval, pred, labels=labels, digits=3))
                fig, ax = plt.subplots(figsize=(8, 7))
                ConfusionMatrixDisplay.from_predictions(y_eval, pred, labels=labels, normalize="true",
                                                        values_format=".2f", ax=ax, colorbar=False)
                ax.set_title(f"{tag} + {model_name.upper()} — {eval_name} (macro-F1 {eval_f1:.3f})")
                fig.tight_layout()
                fig.savefig(out / f"confusion_{model_name}.png", dpi=150)
                plt.close(fig)

    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
