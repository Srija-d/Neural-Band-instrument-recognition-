"""
Final test-set evaluation for every model: the "Test metrics" box.
Owner: Bhagyashri

Takes each model's test predictions (CSV files with columns path, label, pred,
one file per run/seed) and scores them all the same way:

  * macro-F1 and accuracy per run, using Srija's src.evaluation.metrics.evaluate
    (which also saves a JSON and a confusion matrix per run)
  * mean +/- std over seeds, plus a 95% confidence interval from bootstrapping
    the test clips, so we know how much the numbers could move by chance
  * paired comparisons against a reference model (e.g. CNN vs SVM for RQ1),
    with a 95% interval on the difference
  * per-instrument F1, the most common confusions, and row-normalized confusion
    matrices averaged over seeds
  * the clips every model gets wrong, for the error analysis

Usage (from the repo root), one --model per model as NAME=FILES:

  python -m src.evaluation.evaluate_test \
      --model "SVM=results/test/preds/svm_seed0.csv" \
      --model "Random Forest=results/test/preds/rf_seed0.csv" \
      --model "CNN=results/test/preds/cnn_seed*.csv" \
      --model "AST zero-shot=results/test/preds/ast_zeroshot*.csv" \
      --model "AST fine-tuned=results/test/preds/ast_ft*.csv" \
      --reference SVM

Outputs go to results/test/ (summary.csv, comparisons.csv, per_class_f1.csv,
top_confusions.csv, hard_clips.csv, confusion_<model>.png, summary.md).
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASS_CODES
from src.evaluation.metrics import evaluate, majority_baseline


# =============================================================================
# Loading everyone's prediction files
# =============================================================================

def clip_key(path: str) -> str:
    """Match clips across files by file name, since full paths differ between machines."""
    return Path(str(path)).name


def load_runs(spec: str) -> tuple[str, list[pd.DataFrame], list[str]]:
    """Parse "NAME=glob" and load every matching CSV as one run."""
    if "=" not in spec:
        raise SystemExit(f"--model must look like NAME=FILES, got {spec!r}")
    name, pattern = spec.split("=", 1)
    files = sorted(glob.glob(pattern.strip()))
    if not files:
        raise SystemExit(f"No files match {pattern!r} for model {name!r}")

    runs = []
    for f in files:
        df = pd.read_csv(f)
        missing = {"path", "label", "pred"} - set(df.columns)
        if missing:
            raise SystemExit(f"{f} is missing columns {sorted(missing)} (needs path, label, pred)")
        df["key"] = df["path"].map(clip_key)
        if df["key"].duplicated().any():
            raise SystemExit(f"{f}: some clips appear twice")
        unknown = set(df["label"].astype(str)) | set(df["pred"].astype(str))
        unknown -= set(CLASS_CODES)
        if unknown:
            raise SystemExit(f"{f} uses labels not in CLASS_CODES: {sorted(unknown)}")
        runs.append(df.set_index("key"))
    return name.strip(), runs, files


# =============================================================================
# Fast scoring (needed for thousands of bootstrap samples)
# =============================================================================

def confusion_counts(y_true, y_pred, k):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def macro_f1_from_counts(cm):
    tp = np.diag(cm).astype(float)
    precision_den = cm.sum(axis=0)
    recall_den = cm.sum(axis=1)
    f1_den = precision_den + recall_den
    f1 = np.divide(2 * tp, f1_den, out=np.zeros_like(tp), where=f1_den > 0)
    return f1.mean()


def macro_f1(y_true, y_pred, k):
    return macro_f1_from_counts(confusion_counts(y_true, y_pred, k))


def bootstrap_scores(y_true, run_preds, k, samples, rng_seed=0):
    """Macro-F1 (averaged over runs) on `samples` resamplings of the test clips.

    Every model is resampled with the same clip indices (same rng_seed), so the
    samples of two models can be subtracted to compare them fairly.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    out = np.empty(samples)
    for b in range(samples):
        idx = rng.integers(0, n, n)
        out[b] = np.mean([macro_f1(y_true[idx], p[idx], k) for p in run_preds])
    return out


# =============================================================================
# Plots
# =============================================================================

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_confusion(cm_norm, name, f1_mean, out_path):
    """Row-normalized confusion matrix: each row shows where that instrument's clips went."""
    k = len(CLASS_CODES)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), CLASS_CODES, rotation=45, ha="right")
    ax.set_yticks(range(k), CLASS_CODES)
    ax.set_xlabel("Predicted instrument")
    ax.set_ylabel("True instrument")
    for i in range(k):
        for j in range(k):
            v = cm_norm[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    ax.set_title(f"{name} on test: macro-F1 {f1_mean:.3f}\n(rows sum to 1; averaged over seeds)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Score every model on the test set the same way.")
    parser.add_argument("--model", action="append", required=True,
                        help='NAME=FILES, e.g. "CNN=results/test/preds/cnn_seed*.csv" (repeat per model)')
    parser.add_argument("--reference", help="model to compare all others against (e.g. SVM for RQ1)")
    parser.add_argument("--manifest", default="irmas_prep/manifest.csv", type=Path,
                        help="used for the majority-class baseline and to check the test clip list")
    parser.add_argument("--out_dir", default="results/test", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000, help="number of bootstrap samples")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    k = len(CLASS_CODES)
    code_to_id = {c: i for i, c in enumerate(CLASS_CODES)}

    models = [load_runs(spec) for spec in args.model]
    names = [name for name, _, _ in models]
    if args.reference and args.reference not in names:
        raise SystemExit(f"--reference {args.reference!r} is not one of {names}")

    # --- Every model must be scored on exactly the same test clips -----------
    clip_sets = {name: set(runs[0].index) for name, runs, _ in models}
    for name, runs, files in models:
        for df, f in zip(runs, files):
            if set(df.index) != clip_sets[name]:
                raise SystemExit(f"{f} covers different clips than the other runs of {name}")
    first_name = names[0]
    for name in names[1:]:
        if clip_sets[name] != clip_sets[first_name]:
            only_a = len(clip_sets[first_name] - clip_sets[name])
            only_b = len(clip_sets[name] - clip_sets[first_name])
            raise SystemExit(f"{name} and {first_name} cover different test clips "
                             f"({only_b} only in {name}, {only_a} only in {first_name}). "
                             "Everyone must predict the same test split.")
    keys = sorted(clip_sets[first_name])

    manifest = pd.read_csv(args.manifest) if args.manifest.exists() else None
    if manifest is not None:
        test_keys = set(manifest.loc[manifest["split"].astype(str).str.lower() == "test", "path"].map(clip_key))
        if test_keys and set(keys) != test_keys:
            raise SystemExit(f"The prediction files cover {len(keys)} clips, but manifest.csv has "
                             f"{len(test_keys)} test clips. Are these really test predictions?")
    print(f"Scoring {len(names)} models on {len(keys)} test clips\n")

    # Ground truth, checked to agree across all files
    y_true_codes = models[0][1][0].loc[keys, "label"].astype(str).to_numpy()
    for name, runs, files in models:
        for df, f in zip(runs, files):
            if (df.loc[keys, "label"].astype(str).to_numpy() != y_true_codes).any():
                raise SystemExit(f"{f} has different true labels for some clips")
    y_true = np.array([code_to_id[c] for c in y_true_codes])

    # --- Add the majority-class baseline ("always guess the most common") ---
    if manifest is not None:
        train_labels = manifest.loc[manifest["split"].astype(str).str.lower() == "train", "label"].astype(str)
        majority = majority_baseline(list(train_labels), list(y_true_codes))
        maj_df = pd.DataFrame({"label": y_true_codes, "pred": majority}, index=keys)
        models.insert(0, ("Majority class", [maj_df], ["(computed from train labels)"]))
        names.insert(0, "Majority class")

    # --- Score each model ----------------------------------------------------
    summary_rows, per_class_rows, confusion_rows = [], [], []
    boot = {}
    run_preds_by_model = {}

    for name, runs, files in models:
        run_preds = [np.array([code_to_id[c] for c in df.loc[keys, "pred"].astype(str)]) for df in runs]
        run_preds_by_model[name] = run_preds

        f1s, accs = [], []
        for i, (df, p) in enumerate(zip(runs, run_preds)):
            # Srija's shared evaluate(): saves a per-run JSON + confusion matrix too
            scores = evaluate(list(y_true_codes), list(df.loc[keys, "pred"].astype(str)),
                              model_name=f"test_{slug(name)}", run_name=f"run{i}",
                              save=name != "Majority class")
            f1s.append(scores["macro_f1"])
            accs.append(scores["accuracy"])

        boot[name] = bootstrap_scores(y_true, run_preds, k, args.bootstrap)
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        summary_rows.append({
            "model": name, "runs": len(runs),
            "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s, ddof=1) if len(f1s) > 1 else np.nan,
            "macro_f1_ci_low": lo, "macro_f1_ci_high": hi,
            "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs, ddof=1) if len(accs) > 1 else np.nan,
        })

        # Per-instrument F1 and confusions, averaged over runs
        cms = [confusion_counts(y_true, p, k) for p in run_preds]
        cm_mean = np.mean(cms, axis=0)
        per_class = {"model": name}
        for c, code in enumerate(CLASS_CODES):
            f1_c = []
            for cm in cms:
                denom = cm[:, c].sum() + cm[c, :].sum()
                f1_c.append(2 * cm[c, c] / denom if denom else 0.0)
            per_class[code] = np.mean(f1_c)
        per_class_rows.append(per_class)

        row_totals = cm_mean.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_mean, row_totals, out=np.zeros_like(cm_mean), where=row_totals > 0)
        for i in range(k):
            for j in range(k):
                if i != j and cm_norm[i, j] > 0:
                    confusion_rows.append({"model": name, "true": CLASS_CODES[i], "predicted": CLASS_CODES[j],
                                           "share_of_true_class": cm_norm[i, j]})
        if name != "Majority class":
            plot_confusion(cm_norm, name, np.mean(f1s), args.out_dir / f"confusion_{slug(name)}.png")

    summary = pd.DataFrame(summary_rows).round(4)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    per_class = pd.DataFrame(per_class_rows).round(3)
    per_class.to_csv(args.out_dir / "per_class_f1.csv", index=False)
    top_conf = (pd.DataFrame(confusion_rows)
                .sort_values(["model", "share_of_true_class"], ascending=[True, False])
                .groupby("model").head(5).round(3))
    top_conf.to_csv(args.out_dir / "top_confusions.csv", index=False)

    # --- Paired comparisons against the reference model ----------------------
    comparisons = pd.DataFrame()
    if args.reference:
        rows = []
        ref = boot[args.reference]
        ref_mean = summary.loc[summary["model"] == args.reference, "macro_f1_mean"].item()
        for name in names:
            if name in (args.reference, "Majority class"):
                continue
            diff = boot[name] - ref
            lo, hi = np.percentile(diff, [2.5, 97.5])
            mean = summary.loc[summary["model"] == name, "macro_f1_mean"].item()
            rows.append({
                "model": name, "vs": args.reference,
                "difference_points": round((mean - ref_mean) * 100, 1),
                "ci_low_points": round(lo * 100, 1), "ci_high_points": round(hi * 100, 1),
                "clears_5_point_bar": bool(lo * 100 >= 5),
                "share_of_samples_where_worse": round(float((diff <= 0).mean()), 4),
            })
        comparisons = pd.DataFrame(rows)
        comparisons.to_csv(args.out_dir / "comparisons.csv", index=False)

    # --- Clips that every model gets wrong (for the error analysis) ----------
    real_models = [n for n in names if n != "Majority class"]
    wrong_everywhere = np.ones(len(keys), dtype=bool)
    for name in real_models:
        # A clip counts as wrong for a model if most of its runs get it wrong
        wrong_votes = np.mean([p != y_true for p in run_preds_by_model[name]], axis=0)
        wrong_everywhere &= wrong_votes > 0.5
    hard = pd.DataFrame({"clip": np.array(keys)[wrong_everywhere], "label": y_true_codes[wrong_everywhere]})
    for name in real_models:
        hard[f"pred_{slug(name)}"] = [CLASS_CODES[i] for i in run_preds_by_model[name][0][wrong_everywhere]]
    hard.to_csv(args.out_dir / "hard_clips.csv", index=False)

    # --- Readable summary ----------------------------------------------------
    lines = ["# Test-set results", "",
             f"{len(keys)} test clips, {k} instruments. Macro-F1 is the main metric; "
             f"95% intervals come from {args.bootstrap} bootstrap resamples of the test clips.", "",
             "| Model | Runs | Macro-F1 (mean ± std) | 95% CI | Accuracy |", "|---|---|---|---|---|"]
    for r in summary.itertuples():
        std = f" ± {r.macro_f1_std:.3f}" if not np.isnan(r.macro_f1_std) else ""
        lines.append(f"| {r.model} | {r.runs} | {r.macro_f1_mean:.3f}{std} | "
                     f"{r.macro_f1_ci_low:.3f}–{r.macro_f1_ci_high:.3f} | {r.accuracy_mean:.3f} |")
    if len(comparisons):
        lines += ["", f"## Differences vs. {args.reference} (macro-F1 points)", "",
                  "| Model | Difference | 95% CI | Clears 5-point bar? |", "|---|---|---|---|"]
        for r in comparisons.itertuples():
            lines.append(f"| {r.model} | {r.difference_points:+.1f} | {r.ci_low_points:+.1f} to "
                         f"{r.ci_high_points:+.1f} | {'yes' if r.clears_5_point_bar else 'no'} |")
    lines += ["", f"{int(wrong_everywhere.sum())} test clips are wrong for every model (hard_clips.csv)."]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nAll outputs are in {args.out_dir}/")


if __name__ == "__main__":
    main()
