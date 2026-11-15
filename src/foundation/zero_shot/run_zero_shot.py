"""Zero-shot evaluation of AST, PANNs and CLAP: no training on IRMAS at all. Owner: Srija.

  * AST / PANNs were trained on AudioSet (527 labels). Each IRMAS class is mapped to one or
    more AudioSet labels (AUDIOSET_MAP below); the class score is the max probability over its
    mapped labels and the prediction is the highest-scoring of the 11 classes.
  * CLAP compares the clip's audio embedding with a text embedding per class
    ("the sound of a cello", ...). Several prompt templates are tried, plus their average
    ("ensemble"); the template is chosen on val only, then reported.

Nothing is fitted, so there is one number per model (no seeds, no training subsets);
on the data-scarcity plot it is a flat line.

Usage, from the repo root:
    python -m src.foundation.zero_shot.run_zero_shot                  # all three models, val
    python -m src.foundation.zero_shot.run_zero_shot --models clap    # one model
    python -m src.foundation.zero_shot.run_zero_shot --eval_test      # week 7 only: also test

Writes to results/foundation/zero_shot/:
    results.csv                       one row per model (val, + test with --eval_test)
    clap_prompts.csv                  val macro-F1 of every CLAP prompt template
    preds/<model>_<split>.csv         path, label, pred
    report_<model>_<split>.txt        per-class precision / recall / F1
    <model>/<split>_confusion.png     via src.evaluation.metrics.evaluate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import CLASS_CODES, CLASSES, IRMAS_TRAIN_DIR
from src.evaluation.metrics import evaluate
from src.foundation.common import (DEFAULT_CACHE, MODELS, CLAPBackbone, get_features, load_manifest,
                                   pick_device, split_indices)

# IRMAS class -> AudioSet labels (exact names from the AudioSet ontology).
# Kept to labels that clearly mean that instrument; generic parents such as "Guitar",
# "Bowed string instrument" or "Keyboard (musical)" are left out because they would
# fire equally for two IRMAS classes. Document this table in the report.
AUDIOSET_MAP = {
    "cel": ["Cello", "Double bass"],
    "cla": ["Clarinet"],
    "flu": ["Flute"],
    "gac": ["Acoustic guitar", "Strum"],
    "gel": ["Electric guitar"],
    "org": ["Organ", "Electronic organ", "Hammond organ"],
    "pia": ["Piano", "Electric piano"],
    "sax": ["Saxophone"],
    "tru": ["Trumpet"],
    "vio": ["Violin, fiddle", "Pizzicato"],
    "voi": ["Singing", "Male singing", "Female singing", "Choir"],
}

# Text for each class in CLAP prompts.
CLAP_NAMES = {**CLASSES, "voi": "singing voice"}
CLAP_TEMPLATES = [
    "the sound of a {}",
    "a recording of a {}",
    "music played on the {}",
    "a {} solo",
    "{}",
]


def audioset_scores(probs: np.ndarray, labels: list[str], model: str) -> np.ndarray:
    """(N, 527) AudioSet probabilities -> (N, 11) IRMAS class scores (max over mapped labels)."""
    index = {name.lower(): i for i, name in enumerate(labels)}
    scores = np.zeros((len(probs), len(CLASS_CODES)), dtype=np.float32)
    for c, code in enumerate(CLASS_CODES):
        cols = [index[n.lower()] for n in AUDIOSET_MAP[code] if n.lower() in index]
        skipped = [n for n in AUDIOSET_MAP[code] if n.lower() not in index]
        if skipped:
            print(f"  [{model}] note: {skipped} not in this model's label set, skipped")
        if not cols:
            raise SystemExit(f"[{model}] no AudioSet label found for class {code}: {AUDIOSET_MAP[code]}")
        scores[:, c] = probs[:, cols].max(axis=1)
    return scores


def clap_scores(audio_emb: np.ndarray, device, va: np.ndarray, y: np.ndarray, out_dir: Path):
    """Pick the CLAP prompt template on val; return (N, 11) cosine similarities and its name."""
    backbone = CLAPBackbone().to(device).eval()
    audio = torch.as_tensor(audio_emb, dtype=torch.float32)
    audio = torch.nn.functional.normalize(audio, dim=-1)
    text = {}
    for t in CLAP_TEMPLATES:
        text[t] = backbone.text_embeddings([t.format(CLAP_NAMES[c]) for c in CLASS_CODES], device).float().cpu()
    ens = torch.stack(list(text.values())).mean(0)
    text["ensemble (mean of all templates)"] = torch.nn.functional.normalize(ens, dim=-1)

    rows, sims = [], {}
    for t, emb in text.items():
        sims[t] = (audio @ emb.T).numpy()
        pred = sims[t][va].argmax(1)
        rows.append({"template": t, "val_macro_f1": round(f1_score(y[va], pred, average="macro"), 4),
                     "val_acc": round(accuracy_score(y[va], pred), 4)})
    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    table.to_csv(out_dir / "clap_prompts.csv", index=False)
    print(table.to_string(index=False))
    best = table.iloc[0]["template"]
    return sims[best], best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--prep_dir", default="irmas_prep", type=Path)
    ap.add_argument("--data_dir", default=IRMAS_TRAIN_DIR, type=Path)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE, type=Path)
    ap.add_argument("--out_dir", default="results/foundation/zero_shot", type=Path)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_test", action="store_true", help="also score test (week 7 only)")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    args = ap.parse_args()

    device = pick_device(args.device)
    manifest = load_manifest(args.prep_dir, args.data_dir)
    y_str = manifest["label"].astype(str).to_numpy()
    y = np.array([CLASS_CODES.index(c) for c in y_str])
    _, va, te = split_indices(manifest, 100)
    splits = [("val", va)] + ([("test", te)] if args.eval_test else [])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "preds").mkdir(exist_ok=True)
    results_csv = args.out_dir / "results.csv"
    print(f"device={device}  clips={len(manifest)}  val={len(va)}  test={len(te)}")

    for name in args.models:
        print(f"\n=== {name} (zero-shot)")
        feats = get_features(name, manifest, device, args.data_dir, args.cache_dir, args.batch_size)
        if name == "clap":
            scores, prompt = clap_scores(feats["emb"], device, va, y, args.out_dir)
        else:
            scores, prompt = audioset_scores(feats["probs"], feats["labels"], name), ""
        pred = scores.argmax(1)

        row = {"model": name, "run": "zero_shot", "prompt": prompt, "n_train": 0}
        for split, idx in splits:
            s = evaluate(y_str[idx], [CLASS_CODES[i] for i in pred[idx]],
                         model_name=f"foundation/zero_shot/{name}", run_name=split)
            row[f"{split}_macro_f1"] = round(s["macro_f1"], 4)
            row[f"{split}_acc"] = round(s["accuracy"], 4)
            pd.DataFrame({"path": manifest["path"].iloc[idx].to_numpy(), "label": y_str[idx],
                          "pred": [CLASS_CODES[i] for i in pred[idx]]}
                         ).to_csv(args.out_dir / "preds" / f"{name}_{split}.csv", index=False)
            report = classification_report(y[idx], pred[idx], labels=range(len(CLASS_CODES)),
                                           target_names=CLASS_CODES, digits=3, zero_division=0)
            (args.out_dir / f"report_{name}_{split}.txt").write_text(report)
            print(f"  {split}: macro-F1 {row[f'{split}_macro_f1']:.3f}  acc {row[f'{split}_acc']:.3f}")

        old = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame()
        if len(old):
            old = old[old["model"] != name]
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(results_csv, index=False)

    print("\n" + pd.read_csv(results_csv).to_string(index=False))


if __name__ == "__main__":
    main()
