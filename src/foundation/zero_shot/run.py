"""Zero-shot evaluation of AST, PANNs and CLAP (no training on IRMAS). Owner: Srija.

TODO (week 3):
- AST / PANNs: map AudioSet labels to the 11 IRMAS classes (e.g. "Violin, fiddle" -> vio,
  "Organ" + "Hammond organ" -> org). Score = max over mapped labels; predict the top class.
  Write the mapping in a dict here and document it in the report.
- CLAP: one text prompt per class, e.g. "the sound of a cello"; predict the closest prompt.
  Pick between a few prompt templates using val.csv only.
- Resample to each model's rate (AST 16 kHz, PANNs 32 kHz, CLAP 48 kHz).
- Score on test.csv with src.evaluation.metrics.evaluate.
"""
