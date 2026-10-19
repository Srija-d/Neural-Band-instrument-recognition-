"""Recording-aware train/val/test split + data-scarcity subsets. Owner: Aisha.

Clips cut from the same recording must stay in the same split, or results leak.
Run once, commit the CSVs in results/splits/, and every model uses the same files.

Usage:
    python -m src.splits.make_splits
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (DATA_FRACTIONS, IRMAS_TRAIN_DIR, SPLITS_DIR, TEST_SIZE,
                        VAL_SIZE)


def recording_id(filename):
    """Group key for clips from the same recording.

    TODO (week 1): check this against the real IRMAS file names while exploring the data.
    Training files look like "[cel][cla]0001__1.wav"; we assume the part before "__"
    identifies the source recording. Adjust here if that turns out to be wrong.
    """
    stem = filename.rsplit(".", 1)[0]
    return stem.split("__")[0]


def build_index():
    rows = []
    for wav in sorted(IRMAS_TRAIN_DIR.rglob("*.wav")):
        rows.append({
            "path": str(wav.relative_to(IRMAS_TRAIN_DIR)),
            "label": wav.parent.name,                       # folder name = instrument code
            "group": wav.parent.name + "/" + recording_id(wav.name),
        })
    return pd.DataFrame(rows)


def split(df, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=df["group"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=seed)
    train_idx, val_idx = next(gss.split(trainval, groups=trainval["group"]))
    return trainval.iloc[train_idx], trainval.iloc[val_idx], test


def scarce_subsets(train, seed=0):
    """Class-balanced fractions of the training set, chosen by whole recordings."""
    out = {}
    for frac in DATA_FRACTIONS:
        parts = []
        for _, g in train.groupby("label"):
            groups = g["group"].drop_duplicates().sample(frac=frac, random_state=seed)
            parts.append(g[g["group"].isin(groups)])
        out[frac] = pd.concat(parts)
    return out


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_index()
    train, val, test = split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):5d} clips, {part['group'].nunique():4d} recordings")
    for frac, sub in scarce_subsets(train).items():
        sub.to_csv(SPLITS_DIR / f"train_{int(frac * 100):03d}pct.csv", index=False)
    assert not set(train["group"]) & set(test["group"]), "Leak: recording in train and test"
    assert not set(val["group"]) & set(test["group"]), "Leak: recording in val and test"


if __name__ == "__main__":
    main()
