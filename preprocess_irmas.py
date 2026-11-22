"""Preprocess IRMAS clips into a shared feature manifest for every downstream model.

For each clip assigned to a split in results/splits/ (built by
src.splits.make_splits, recording-aware, seed-42), this cleans the audio with
src.preprocessing.audio.load_clip (resample, trim silence, normalize, fix
length) and extracts:

  - mels.npy       log-mel spectrograms, for the CNN and foundation models  (N, n_mels, frames)
  - mfcc_feats.npy      mean+std pooled MFCCs, for the classical SVM/RF baseline  (N, 2 * n_mfcc)
  - classical_feats.npy mean+std pooled MFCCs + deltas + spectral/chroma features  (N, 170)
  - classical_feats.txt names of the 170 classical_feats columns, one per line
  - manifest.csv        one row per clip, in the same order as the arrays

Recording IDs and the train/val/test/subset assignment are NOT re-derived
here. They come straight from results/splits/*.csv, which make_splits.py
already builds once (recording-aware grouping, seed 42) and the whole team
shares. Reading them back in keeps this manifest and the committed splits
from ever disagreeing. The split_random column comes from results/splits/random/
and is only for the random-vs-recording-aware split ablation.

Usage:
    python preprocess_irmas.py --out-dir irmas_prep
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import IRMAS_TRAIN_DIR, N_MFCC, RANDOM_SPLITS_DIR, SAMPLE_RATE, SPLITS_DIR
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

    # Split ablation: where each clip lands in the leaky clip-level random split.
    random_split = {}
    for name in ("train", "val", "test"):
        for path in pd.read_csv(splits_dir / "random" / f"{name}.csv")["path"]:
            random_split[path] = name
    manifest["split_random"] = manifest["path"].map(random_split)
    if manifest["split_random"].isna().any():
        raise ValueError("random split CSVs don't cover every clip; re-run make_splits")

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


def classical_features(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """Stronger fixed-length vector for the classical baseline: mean and std over
    time of MFCCs, their deltas and delta-deltas, spectral shape (centroid,
    bandwidth, rolloff, flatness, 7-band contrast), zero-crossing rate, RMS
    energy and 12 chroma bins -> (170,). Column names in CLASSICAL_FEATURE_NAMES.
    """
    S = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    frames = np.vstack([
        mfcc,
        librosa.feature.delta(mfcc),
        librosa.feature.delta(mfcc, order=2),
        librosa.feature.spectral_centroid(S=S, sr=sr),
        librosa.feature.spectral_bandwidth(S=S, sr=sr),
        librosa.feature.spectral_rolloff(S=S, sr=sr),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.spectral_contrast(S=S, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(S=S),
        librosa.feature.chroma_stft(S=S ** 2, sr=sr),
    ])
    return np.concatenate([frames.mean(axis=1), frames.std(axis=1)]).astype(np.float32)


_FRAME_NAMES = ([f"mfcc{i}" for i in range(N_MFCC)]
                + [f"mfcc{i}_d" for i in range(N_MFCC)]
                + [f"mfcc{i}_dd" for i in range(N_MFCC)]
                + ["centroid", "bandwidth", "rolloff", "flatness"]
                + [f"contrast{i}" for i in range(7)]
                + ["zcr", "rms"]
                + [f"chroma{i}" for i in range(12)])
CLASSICAL_FEATURE_NAMES = [f"{n}_{stat}" for stat in ("mean", "std") for n in _FRAME_NAMES]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=IRMAS_TRAIN_DIR,
                     help="IRMAS-TrainingData folder (read-only, never modified)")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                     help="folder with train/val/test/train_*pct.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("irmas_prep"),
                     help="where to write manifest.csv, mels.npy, mfcc_feats.npy, classical_feats.npy")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.data_dir, args.splits_dir)

    mels, mfccs, classical = [], [], []
    silent_clips, nan_clips = [], []
    for path in tqdm(manifest["path"], desc="preprocessing"):
        y = load_clip(args.data_dir / path)

        if not np.any(y):
            silent_clips.append(path)

        mel = mel_spectrogram(y)
        mfcc = mfcc_features(y)
        clas = classical_features(y)
        if not (np.isfinite(mel).all() and np.isfinite(mfcc).all() and np.isfinite(clas).all()):
            nan_clips.append(path)

        mels.append(mel)
        mfccs.append(mfcc)
        classical.append(clas)

    mels = np.stack(mels)
    mfccs = np.stack(mfccs)
    classical = np.stack(classical)
    assert classical.shape[1] == len(CLASSICAL_FEATURE_NAMES)

    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    np.save(args.out_dir / "mels.npy", mels)
    np.save(args.out_dir / "mfcc_feats.npy", mfccs)
    np.save(args.out_dir / "classical_feats.npy", classical)
    (args.out_dir / "classical_feats.txt").write_text("\n".join(CLASSICAL_FEATURE_NAMES) + "\n")

    print(f"\nwrote {len(manifest)} rows to {args.out_dir}")
    print(f"  manifest.csv   {manifest.shape}")
    print(f"  mels.npy       {mels.shape}")
    print(f"  mfcc_feats.npy {mfccs.shape}")
    print(f"  classical_feats.npy {classical.shape}")
    if silent_clips:
        print(f"  WARNING: {len(silent_clips)} clips are entirely silent after cleaning")
    if nan_clips:
        print(f"  WARNING: {len(nan_clips)} clips produced NaN/Inf features")


if __name__ == "__main__":
    main()
