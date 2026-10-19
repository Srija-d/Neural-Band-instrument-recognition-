"""Classical baseline: MFCC + other hand-made features -> SVM / Random Forest. Owner: Aisha.

TODO (week 3):
1. For each clip in results/splits/train.csv, load it with src.preprocessing.audio.load_clip.
2. Extract MFCCs (config.N_MFCC) + spectral features; summarize with mean and std over time.
3. Tune SVM / Random Forest on val.csv only.
4. Score with src.evaluation.metrics.evaluate for each seed in config.SEEDS.
5. Repeat on each train_XXXpct.csv for the data-scarcity study.
"""
