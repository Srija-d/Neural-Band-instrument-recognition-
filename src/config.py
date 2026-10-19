"""Shared settings. Every model reads from here so results stay comparable."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IRMAS_TRAIN_DIR = DATA_DIR / "IRMAS-TrainingData"
SPLITS_DIR = ROOT / "results" / "splits"
RESULTS_DIR = ROOT / "results"

# IRMAS instrument codes (folder names) -> readable names
CLASSES = {
    "cel": "cello",
    "cla": "clarinet",
    "flu": "flute",
    "gac": "acoustic guitar",
    "gel": "electric guitar",
    "org": "organ",
    "pia": "piano",
    "sax": "saxophone",
    "tru": "trumpet",
    "vio": "violin",
    "voi": "voice",
}
CLASS_CODES = sorted(CLASSES)

# Audio settings for the classical model and CNN.
# Foundation models resample to their own rate (AST 16 kHz, PANNs 32 kHz, CLAP 48 kHz).
SAMPLE_RATE = 22050
CLIP_SECONDS = 3.0
N_MELS = 128
N_MFCC = 20

# Experiment settings
SEEDS = [0, 1, 2]                                   # 3 runs per model
DATA_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]     # data-scarcity study
VAL_SIZE = 0.15
TEST_SIZE = 0.15
