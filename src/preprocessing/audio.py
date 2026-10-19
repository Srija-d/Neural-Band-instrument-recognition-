"""Audio loading and cleaning. Owner: Aisha.

Same steps for every model: resample, trim silence, normalize volume, fix length.
"""
import librosa
import numpy as np

from src.config import CLIP_SECONDS, SAMPLE_RATE


def load_clip(path, sr=SAMPLE_RATE, seconds=CLIP_SECONDS, top_db=40):
    """Load a wav file as a mono float array of exactly `seconds` seconds."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)        # remove leading/trailing silence
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak                                      # equal volume
    target = int(sr * seconds)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=128):
    """Log-mel spectrogram (n_mels x frames) for the CNN."""
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)


# TODO (week 2): add a function that cuts clips into 1 s / 3 s windows for the clip-length ablation.
