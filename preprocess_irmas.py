"""Preprocess IRMAS clips into a shared feature manifest for every downstream model.

For each clip assigned to a split in results/splits/ (built by
src.splits.make_splits, recording-aware, seed-42), this cleans the audio with
src.preprocessing.audio.load_clip (resample, trim silence, normalize, fix
length) and extracts:

  - mels.npy       log-mel spectrograms, for the CNN and foundation models  (N, n_mels, frames)
  - mfcc_feats.npy mean+std pooled MFCCs, for the classical SVM/RF baseline  (N, 2 * n_mfcc)
  - manifest.csv   one row per clip, in the same order as the two arrays

Recording IDs and the train/val/test/subset assignment are NOT re-derived
here. They come straight from results/splits/*.csv, which make_splits.py
already builds once (recording-aware grouping, seed 42) and the whole team
shares. Reading them back in keeps this manifest and the committed splits
from ever disagreeing.

Usage:
    python preprocess_irmas.py --out-dir irmas_prep
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import IRMAS_TRAIN_DIR, N_MFCC, SAMPLE_RATE, SPLITS_DIR
from src.preprocessing.audio import load_clip, mel_spectrogram

SUBSET_PERCENTS = [5, 10, 25, 50, 100]


def build_manifest(data_dir: Path, splits_dir: Path) -> pd.DataFrame:
    """One row per clip: path, label, recording_id, split, and which
    data-scarcity training subsets (5/10/25/50/100%) it belongs to.
    """
    parts = []
    for name in ("train", "val", "test"):
        df = pd.read_csv(splits_dir / f"{name}.csv")
        df["split"] = name
        parts.append(df)
    manifest = pd.concat(parts, ignore_index=True)
    manifest = manifest.rename(columns={"group": "recording_id"})

    for pct in SUBSET_PERCENTS:
        sub = pd.read_csv(splits_dir / f"train_{pct:03d}pct.csv")
        manifest[f"pct_{pct:03d}"] = manifest["path"].isin(set(sub["path"]))

    # Sanity: the manifest should cover every wav under data_dir exactly once.
    on_disk = {str(p.relative_to(data_dir)) for p in data_dir.rglob("*.wav")}
    in_manifest = set(manifest["path"])
    missing, extra = on_disk - in_manifest, in_manifest - on_disk
    if missing or extra:
        raise ValueError(
            f"manifest/disk mismatch: {len(missing)} clips on disk are missing from "
            f"the splits, {len(extra)} rows in the splits don't exist on disk"
        )

    return manifest.sort_values("path").reset_index(drop=True)


def mfcc_features(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Fixed-length feature vector for the classical SVM/RF baseline:
    mean and std of each MFCC coefficient across time -> (2 * n_mfcc,).
    """
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=IRMAS_TRAIN_DIR,
                     help="IRMAS-TrainingData folder (read-only, never modified)")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                     help="folder with train/val/test/train_*pct.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("irmas_prep"),
                     help="where to write manifest.csv, mels.npy, mfcc_feats.npy")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.data_dir, args.splits_dir)

    mels, mfccs = [], []
    silent_clips, nan_clips = [], []
    for path in tqdm(manifest["path"], desc="preprocessing"):
        y = load_clip(args.data_dir / path)

        if not np.any(y):
            silent_clips.append(path)

        mel = mel_spectrogram(y)
        mfcc = mfcc_features(y)
        if np.isnan(mel).any() or np.isnan(mfcc).any():
            nan_clips.append(path)

        mels.append(mel)
        mfccs.append(mfcc)

    mels = np.stack(mels)
    mfccs = np.stack(mfccs)

    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    np.save(args.out_dir / "mels.npy", mels)
    np.save(args.out_dir / "mfcc_feats.npy", mfccs)

    print(f"\nwrote {len(manifest)} rows to {args.out_dir}")
    print(f"  manifest.csv   {manifest.shape}")
    print(f"  mels.npy       {mels.shape}")
    print(f"  mfcc_feats.npy {mfccs.shape}")
    if silent_clips:
        print(f"  WARNING: {len(silent_clips)} clips are entirely silent after cleaning")
    if nan_clips:
        print(f"  WARNING: {len(nan_clips)} clips produced NaNs in mel/mfcc features")


if __name__ == "__main__":
    main()
