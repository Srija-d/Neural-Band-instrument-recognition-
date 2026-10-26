"""
Classical baseline for Neural Band: MFCC features + SVM / Random Forest.

Reads the split CSVs (path, label, group) and, for every training subset
(5, 10, 25, 50, 100%):
  1. trains SVM and RF candidates on the subset,
  2. picks the best hyperparameters by macro-F1 on val.csv,
  3. reports macro-F1 on test.csv.

Outputs (in --out_dir):
  features.npz            cached MFCC features (computed once, reused on reruns)
  results.csv             model, subset, best params, val macro-F1, test macro-F1
  report_<model>.txt      per-class precision/recall/F1 on test (100% subset)
  confusion_<model>.png   test confusion matrix (100% subset)

Usage:
  python baseline_mfcc.py --data_root /path/to/IRMAS-TrainingData --splits_dir /path/to/splits
"""

import argparse
import itertools
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
SUBSETS = ["005", "010", "025", "050", "100"]

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


# ------------------------------------------------------------ features
def mfcc_features(path, sr=22050, seconds=3.0, n_mfcc=40):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = librosa.util.fix_length(y, size=int(sr * seconds))
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    # 40 means + 40 stds + 40 delta means + 40 delta stds = 160 numbers per clip
    return np.concatenate([mfcc.mean(1), mfcc.std(1), d1.mean(1), d1.std(1)]).astype(np.float32)


def load_features(paths, data_root, cache_file, jobs):
    cache = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        cache = dict(zip(z["paths"], z["feats"]))
    missing = [p for p in paths if p not in cache]
    if missing:
        print(f"Extracting MFCCs for {len(missing)} clips...")
        feats = Parallel(n_jobs=jobs, verbose=2)(
            delayed(mfcc_features)(str(data_root / p)) for p in missing)
        cache.update(zip(missing, feats))
        np.savez(cache_file, paths=np.array(list(cache)), feats=np.stack(list(cache.values())))
    return cache


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder the CSV 'path' column is relative to")
    ap.add_argument("--splits_dir", required=True, help="folder with train_XXXpct.csv, val.csv, test.csv")
    ap.add_argument("--out_dir", default="baseline_results")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    data_root, splits = Path(args.data_root), Path(args.splits_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(splits / "val.csv")
    test = pd.read_csv(splits / "test.csv")
    subsets = {s: pd.read_csv(splits / f"train_{s}pct.csv") for s in SUBSETS}

    all_paths = sorted(set(pd.concat([val, test, *subsets.values()]).path))
    feats = load_features(all_paths, data_root, out / "features.npz", args.jobs)
    X = lambda df: np.stack([feats[p] for p in df.path])

    labels = sorted(subsets["100"].label.unique())
    Xva, yva = X(val), val.label.values
    Xte, yte = X(test), test.label.values

    rows = []
    for model_name, grid in GRIDS.items():
        for s in SUBSETS:
            tr = subsets[s]
            Xtr, ytr = X(tr), tr.label.values
            best = None
            for combo in itertools.product(*grid.values()):
                params = dict(zip(grid, combo))
                m = make_model(model_name, params).fit(Xtr, ytr)
                f1 = f1_score(yva, m.predict(Xva), average="macro")
                if best is None or f1 > best[0]:
                    best = (f1, params, m)
            val_f1, params, m = best
            pred = m.predict(Xte)
            test_f1 = f1_score(yte, pred, average="macro")
            print(f"{model_name:>3} | {int(s):>3}% train | val F1 {val_f1:.3f} | test F1 {test_f1:.3f} | {params}")
            rows.append(dict(model=model_name, subset_pct=int(s), n_train=len(tr),
                             params=params, val_macro_f1=round(val_f1, 4), test_macro_f1=round(test_f1, 4)))

            if s == "100":
                (out / f"report_{model_name}.txt").write_text(
                    classification_report(yte, pred, labels=labels, digits=3))
                fig, ax = plt.subplots(figsize=(8, 7))
                ConfusionMatrixDisplay.from_predictions(yte, pred, labels=labels, normalize="true",
                                                        values_format=".2f", ax=ax, colorbar=False)
                ax.set_title(f"MFCC + {model_name.upper()} — test (macro-F1 {test_f1:.3f})")
                fig.tight_layout()
                fig.savefig(out / f"confusion_{model_name}.png", dpi=150)
                plt.close(fig)

    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()