"""Small CNN on log-mel spectrograms, with optional SpecAugment. Owner: Bhagyashri.

TODO (week 4):
1. Dataset class: load_clip -> mel_spectrogram (src.preprocessing.audio).
2. Small CNN (a few conv blocks + global pooling + 11-way output), following Han et al. (2017).
3. SpecAugment flag (time and frequency masking, e.g. torchaudio.transforms) for the ablation.
4. Early stopping on val macro-F1; 3 seeds; evaluate with src.evaluation.metrics.evaluate.
5. Repeat on each train_XXXpct.csv for the data-scarcity study.
"""
