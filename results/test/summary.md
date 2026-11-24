# Test-set results

999 test clips, 11 instruments. Macro-F1 is the main metric; 95% intervals come from 1000 bootstrap resamples of the test clips.

| Model | Runs | Macro-F1 (mean ± std) | 95% CI | Accuracy |
|---|---|---|---|---|
| Majority class | 1 | 0.016 | 0.014–0.019 | 0.099 |
| SVM | 1 | 0.477 | 0.443–0.507 | 0.493 |
| Random Forest | 1 | 0.459 | 0.425–0.489 | 0.476 |
| CNN | 3 | 0.756 ± 0.015 | 0.731–0.778 | 0.767 |
| AST zero-shot | 1 | 0.668 | 0.638–0.695 | 0.683 |
| PANNs zero-shot | 1 | 0.689 | 0.655–0.719 | 0.707 |
| CLAP zero-shot | 1 | 0.546 | 0.512–0.575 | 0.559 |
| AST probe | 1 | 0.816 | 0.789–0.839 | 0.826 |
| PANNs probe | 1 | 0.776 | 0.747–0.800 | 0.787 |
| CLAP probe | 1 | 0.732 | 0.700–0.760 | 0.745 |
| AST fine-tuned | 3 | 0.802 ± 0.016 | 0.778–0.823 | 0.812 |
| PANNs fine-tuned | 3 | 0.825 ± 0.010 | 0.800–0.846 | 0.834 |
| CLAP fine-tuned | 1 | 0.821 | 0.795–0.844 | 0.828 |

## Differences vs. CNN (macro-F1 points)

| Model | Difference | 95% CI | Clears 5-point bar? |
|---|---|---|---|
| SVM | -28.0 | -31.2 to -24.5 | no |
| Random Forest | -29.8 | -33.0 to -26.3 | no |
| AST zero-shot | -8.9 | -11.8 to -5.8 | no |
| PANNs zero-shot | -6.8 | -10.0 to -3.9 | no |
| CLAP zero-shot | -21.0 | -24.5 to -17.5 | no |
| AST probe | +5.9 | +3.4 to +8.5 | no |
| PANNs probe | +1.9 | -0.8 to +4.6 | no |
| CLAP probe | -2.4 | -4.9 to +0.3 | no |
| AST fine-tuned | +4.5 | +2.3 to +6.9 | no |
| PANNs fine-tuned | +6.8 | +4.8 to +9.2 | no |
| CLAP fine-tuned | +6.5 | +4.3 to +8.8 | no |

44 test clips are wrong for every model (hard_clips.csv).
