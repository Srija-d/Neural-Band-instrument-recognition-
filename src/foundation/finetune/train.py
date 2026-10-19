"""Linear probe and fine-tuning of AST / PANNs (and CLAP audio encoder). Owner: Bhagyashri.

TODO (week 4-5):
1. Linear probe: extract frozen embeddings once, cache them (not in git), train logistic regression.
2. Fine-tuning: replace the output layer with 11 classes; fine-tune the top layers
   (or the full model if Colab GPU time allows).
3. 3 seeds; run on every train_XXXpct.csv for the data-scarcity curves.
4. Evaluate with src.evaluation.metrics.evaluate.
"""
