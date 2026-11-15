"""Shared code for the foundation-model experiments (zero-shot, linear probe, fine-tuning). Owner: Srija.

Uses the same clips, splits and cleaning as the rest of the project:
  * clip list and train/val/test/subset membership come from irmas_prep/manifest.csv
    (written by preprocess_irmas.py); if it doesn't exist yet, the same manifest is
    rebuilt from results/splits/*.csv with preprocess_irmas.build_manifest.
  * every clip is cleaned with src.preprocessing.audio.load_clip (trim silence,
    peak-normalize, exactly 3 s), only resampled to each model's own rate
    (AST 16 kHz, PANNs 32 kHz, CLAP 48 kHz).

Expensive things are cached under cache/foundation/ (gitignored):
  * wave_<sr>.npy   cleaned waveforms, float16, one row per manifest row
  * <model>_feats.npz  frozen embeddings (+ AudioSet probabilities for AST / PANNs)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import CLASS_CODES, CLIP_SECONDS, IRMAS_TRAIN_DIR, SPLITS_DIR
from src.cnn.train_cnn import SUBSET_COLS, pick_device, set_seed, split_indices  # noqa: F401  (re-exported)
from src.preprocessing.audio import load_clip

MODELS = ["ast", "panns", "clap"]
MODEL_SR = {"ast": 16000, "panns": 32000, "clap": 48000}
CHECKPOINTS = {
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",   # AST fine-tuned on AudioSet (527 labels)
    "panns": "Cnn14_mAP=0.431.pth",                      # PANNs CNN14, downloaded by panns_inference
    "clap": "laion/clap-htsat-unfused",                  # LAION CLAP, HTS-AT audio encoder
}
DEFAULT_CACHE = Path("cache/foundation")


# --------------------------------------------------------------------------- data

def load_manifest(prep_dir: Path = Path("irmas_prep"), data_dir: Path = IRMAS_TRAIN_DIR) -> pd.DataFrame:
    """The shared manifest: path, label, recording_id, split, pct_005 ... pct_100."""
    path = Path(prep_dir) / "manifest.csv"
    if path.exists():
        return pd.read_csv(path)
    print(f"{path} not found; rebuilding the same manifest from {SPLITS_DIR}")
    from preprocess_irmas import build_manifest  # repo-root script, same logic as preprocessing
    return build_manifest(Path(data_dir), SPLITS_DIR)


def labels_to_ids(manifest: pd.DataFrame) -> np.ndarray:
    return manifest["label"].astype(str).map({c: i for i, c in enumerate(CLASS_CODES)}).to_numpy()


def _load_one(args):
    path, sr = args
    return load_clip(path, sr=sr).astype(np.float16)


def load_waveforms(manifest: pd.DataFrame, sr: int, data_dir: Path = IRMAS_TRAIN_DIR,
                   cache_dir: Path = DEFAULT_CACHE, workers: int | None = None) -> np.ndarray:
    """Cleaned 3 s waveforms at `sr`, shape (N, sr * 3), float16, row-aligned with the manifest.

    Built once (a few minutes) and memory-mapped afterwards.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"wave_{sr}.npy"
    keys = cache_dir / f"wave_{sr}_paths.csv"
    if out.exists() and keys.exists() and pd.read_csv(keys)["path"].tolist() == manifest["path"].tolist():
        return np.load(out, mmap_mode="r")

    from concurrent.futures import ProcessPoolExecutor
    n, length = len(manifest), int(sr * CLIP_SECONDS)
    tmp = cache_dir / f"wave_{sr}.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, length))
    jobs = [(str(Path(data_dir) / p), sr) for p in manifest["path"]]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        for i, y in enumerate(tqdm(ex.map(_load_one, jobs, chunksize=16), total=n,
                                   desc=f"loading clips at {sr} Hz")):
            arr[i] = y
    arr.flush()
    del arr
    os.replace(tmp, out)
    manifest[["path"]].to_csv(keys, index=False)
    return np.load(out, mmap_mode="r")


# --------------------------------------------------------------------------- backbones

def _tensor(out, *names):
    """transformers 4.x returns tensors from get_*_features, 5.x may return output objects."""
    if isinstance(out, torch.Tensor):
        return out
    for name in names + ("pooler_output",):
        val = getattr(out, name, None)
        if isinstance(val, torch.Tensor):
            return val
    raise TypeError(f"can't find an embedding tensor in {type(out).__name__}")


class Backbone(nn.Module):
    """Common interface for AST / PANNs / CLAP.

    prepare(waves)  -> dict of CPU tensors (model input) from a float32 array (B, samples)
    forward(**x)    -> (embedding (B, D), AudioSet probabilities (B, 527) or None)
    blocks          -> parameter-name prefixes of the backbone's blocks, bottom to top
    always_train    -> prefixes unfrozen whenever any block is (final norm / projection)
    """
    name: str
    sr: int
    embed_dim: int
    audioset_labels: list[str] | None = None
    blocks: list[str] = []
    always_train: list[str] = []


class ASTBackbone(Backbone):
    name, sr = "ast", MODEL_SR["ast"]

    def __init__(self, checkpoint: str = CHECKPOINTS["ast"]):
        super().__init__()
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.fe = ASTFeatureExtractor.from_pretrained(checkpoint)
        self.net = ASTForAudioClassification.from_pretrained(checkpoint)
        cfg = self.net.config
        self.embed_dim = cfg.hidden_size
        self.audioset_labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        # Transformer layers are "encoder.layer.N." in transformers 4.x and "layers.N." in 5.x.
        found = {}
        for n, _ in self.named_parameters():
            m = re.match(r"^(net\.audio_spectrogram_transformer\.(?:encoder\.layer|layers)\.(\d+)\.)", n)
            if m:
                found[int(m.group(2))] = m.group(1)
        self.blocks = [found[i] for i in sorted(found)]
        self.always_train = ["net.audio_spectrogram_transformer.layernorm."]

    def prepare(self, waves):
        x = self.fe(list(waves), sampling_rate=self.sr, return_tensors="pt")
        return {"input_values": x["input_values"]}

    def forward(self, input_values):
        emb = self.net.audio_spectrogram_transformer(input_values=input_values).pooler_output
        return emb, torch.sigmoid(self.net.classifier(emb))


class PANNsBackbone(Backbone):
    name, sr, embed_dim = "panns", MODEL_SR["panns"], 2048

    def __init__(self, checkpoint: str | None = None):
        super().__init__()
        from panns_inference import AudioTagging, labels
        # device="cpu" so panns_inference doesn't wrap the model in DataParallel; we move it ourselves.
        # The official CNN14 checkpoint is a full pickle; torch >= 2.6 refuses it unless
        # weights_only=False, which panns_inference doesn't pass. It's a trusted file (Zenodo).
        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{"weights_only": False, **k})
        try:
            self.net = AudioTagging(checkpoint_path=checkpoint, device="cpu").model
        finally:
            torch.load = _load
        self.audioset_labels = list(labels)
        self.blocks = [f"net.conv_block{i}." for i in range(1, 7)]
        self.always_train = ["net.fc1."]

    def prepare(self, waves):
        return {"waveform": torch.as_tensor(np.asarray(waves, dtype=np.float32))}

    def forward(self, waveform):
        # In train mode CNN14 applies its own SpecAugment + dropout; in eval mode it doesn't.
        out = self.net(waveform, None)
        return out["embedding"], out["clipwise_output"]


class CLAPBackbone(Backbone):
    name, sr = "clap", MODEL_SR["clap"]

    def __init__(self, checkpoint: str = CHECKPOINTS["clap"]):
        super().__init__()
        from transformers import ClapModel, ClapProcessor
        self.proc = ClapProcessor.from_pretrained(checkpoint)
        self.net = ClapModel.from_pretrained(checkpoint)
        self.embed_dim = self.net.config.projection_dim
        n_stages = len(self.net.audio_model.audio_encoder.layers)
        self.blocks = [f"net.audio_model.audio_encoder.layers.{i}." for i in range(n_stages)]
        self.always_train = ["net.audio_model.audio_encoder.norm.", "net.audio_projection."]

    def prepare(self, waves):
        f = self.proc.feature_extractor(list(waves), sampling_rate=self.sr, return_tensors="pt")
        x = {"input_features": f["input_features"]}
        if "is_longer" in f:
            x["is_longer"] = f["is_longer"]
        return x

    def forward(self, input_features, is_longer=None):
        out = self.net.get_audio_features(input_features=input_features, is_longer=is_longer)
        return nn.functional.normalize(_tensor(out, "audio_embeds"), dim=-1), None

    @torch.no_grad()
    def text_embeddings(self, texts: list[str], device) -> torch.Tensor:
        tok = self.proc.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        out = self.net.get_text_features(**tok)
        return nn.functional.normalize(_tensor(out, "text_embeds"), dim=-1)


def load_backbone(name: str) -> Backbone:
    return {"ast": ASTBackbone, "panns": PANNsBackbone, "clap": CLAPBackbone}[name]()


# --------------------------------------------------------------------------- frozen features

@torch.no_grad()
def extract(backbone: Backbone, waves: np.ndarray, device, batch_size: int = 32):
    """Frozen embeddings (+ AudioSet probabilities) for every row of `waves`."""
    backbone.eval().to(device)
    use_amp = device.type == "cuda"
    embs, probs = [], []
    for i in tqdm(range(0, len(waves), batch_size), desc=f"{backbone.name} embeddings"):
        x = backbone.prepare(np.asarray(waves[i:i + batch_size], dtype=np.float32))
        x = {k: v.to(device) for k, v in x.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            e, p = backbone(**x)
        embs.append(e.float().cpu().numpy())
        if p is not None:
            probs.append(p.float().cpu().numpy())
    return np.concatenate(embs), (np.concatenate(probs) if probs else None)


def get_features(name: str, manifest: pd.DataFrame, device, data_dir: Path = IRMAS_TRAIN_DIR,
                 cache_dir: Path = DEFAULT_CACHE, batch_size: int = 32, backbone: Backbone | None = None):
    """Cached frozen features for all clips: dict(emb=(N, D), probs=(N, 527) or None, labels=[...])."""
    cache = Path(cache_dir) / f"{name}_feats.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if z["paths"].tolist() == manifest["path"].tolist():
            return {"emb": z["emb"], "probs": z["probs"] if z["probs"].size else None,
                    "labels": z["labels"].tolist()}
    backbone = backbone or load_backbone(name)
    waves = load_waveforms(manifest, backbone.sr, data_dir, cache_dir)
    emb, probs = extract(backbone, waves, device, batch_size)
    labels = backbone.audioset_labels or []
    np.savez(cache, emb=emb, probs=probs if probs is not None else np.zeros(0),
             labels=np.array(labels, dtype=object), paths=manifest["path"].to_numpy(dtype=object))
    return {"emb": emb, "probs": probs, "labels": labels}
