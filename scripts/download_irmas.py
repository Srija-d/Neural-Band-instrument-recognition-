"""Download and unzip the IRMAS training data (about 3 GB) into data/.

Usage:
    python scripts/download_irmas.py

Source: Bosch et al. (2012), hosted on Zenodo, record 1290750.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, IRMAS_TRAIN_DIR  # noqa: E402

URL = "https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1"


def progress(blocks, block_size, total):
    done = blocks * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        print(f"\r  {pct:3d}%  ({done / 1e9:.2f} / {total / 1e9:.2f} GB)", end="", flush=True)


def main():
    if IRMAS_TRAIN_DIR.exists():
        print(f"Already downloaded: {IRMAS_TRAIN_DIR}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "IRMAS-TrainingData.zip"
    if not zip_path.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path, progress)
        print()
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    n = len(list(IRMAS_TRAIN_DIR.rglob("*.wav")))
    print(f"Done: {n} wav files in {IRMAS_TRAIN_DIR}")


if __name__ == "__main__":
    main()
