# Neural Band: Musical Instrument Recognition

**CNN vs. classical machine learning vs. audio foundation models on IRMAS**

Team: Aisha Abouganba (aa2381) · Bhagyashri Farkade (bbf24) · Srija Dwarakacherla (sd2003)

## Research questions

1. **RQ1.** Does a CNN on mel spectrograms beat a classical model (SVM / Random Forest on MFCC features) by at least 5 macro-F1 points?
2. **RQ2.** How do pretrained audio foundation models (AST, PANNs, CLAP) compare with our CNN, both **zero-shot** and **fine-tuned**, especially when training data is scarce?

The full proposal is in [`report/Project_Proposal_Neural_Band_Revised.pdf`](report/Project_Proposal_Neural_Band_Revised.pdf).

## Architecture

![Project architecture](report/architecture.png)

## Repository structure

| Folder | What goes there | Owner |
|---|---|---|
| `scripts/` | Data download and one-off scripts | Aisha |
| `src/preprocessing/` | Resample, trim silence, normalize, 3 s clips | Aisha |
| `src/splits/` | Recording-aware train/val/test split, 5–100% subsets | Aisha |
| `src/classical/` | MFCC features + SVM / Random Forest | Aisha |
| `src/cnn/` | CNN on mel spectrograms + SpecAugment | Bhagyashri |
| `src/foundation/zero_shot/` | AST, PANNs, CLAP without IRMAS training | Srija |
| `src/foundation/finetune/` | Linear probe and fine-tuning | Bhagyashri |
| `src/evaluation/` | Macro-F1, confusion matrix, ablations, data-scarcity curves | Srija |
| `notebooks/` | Exploration and Colab runs | Everyone |
| `results/` | Metric tables and plots (small files only) | Everyone |
| `demo/` | Demo app (week 7) | Everyone |
| `report/` | Proposal, final report, slides | Everyone |

`data/` is created locally by the download script and is **not** pushed to GitHub.

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/neural-band-instrument-recognition.git
cd neural-band-instrument-recognition
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_irmas.py # downloads IRMAS training data into data/
```

On Google Colab: clone the repo in a cell, run `pip install -r requirements.txt`, and keep `data/` on Google Drive so you don't re-download it every session.

## Team workflow

1. Before starting work: `git checkout main` then `git pull`.
2. Make a branch for your task: `git checkout -b cnn-baseline`.
3. Commit small, clear steps: `git commit -m "Add mel spectrogram dataset class"`.
4. Push: `git push -u origin cnn-baseline`, then open a **Pull Request** on GitHub.
5. One teammate reviews, then merge into `main`.

Rules:
- Never push audio files, model weights (`*.pt`, `*.pth`, `*.ckpt`) or large outputs. GitHub rejects files over 100 MB.
- Use the **same split files** for every model, so results are comparable.
- Touch the test set only once, at the end (week 7).
- Track tasks on the GitHub **Projects** board (one card per week, assigned to its owner).

## Timeline

| Week | Work |
|---|---|
| 1 | Setup, download data, read papers, explore data |
| 2 | Clean audio, extract features, split data |
| 3 | Train simple models; run zero-shot foundation models |
| 4 | Train CNN; extract pretrained embeddings |
| 5 | Fine-tune foundation models; linear probes |
| 6 | Ablations and data-scarcity experiments |
| 7 | Final testing, error analysis, demo app |
| 8 | Write report and presentation |

## References

1. Eronen & Klapuri (2000). Musical instrument recognition using cepstral coefficients and temporal features. ICASSP.
2. Bosch et al. (2012). A comparison of sound segregation techniques for predominant instrument recognition in musical audio signals. ISMIR.
3. Han, Kim & Lee (2017). Deep convolutional neural networks for predominant instrument recognition in polyphonic music. IEEE/ACM TASLP.
4. Park et al. (2019). SpecAugment. Interspeech.
5. Gemmeke et al. (2017). Audio Set. ICASSP.
6. Kong et al. (2020). PANNs: Large-scale pretrained audio neural networks. IEEE/ACM TASLP.
7. Gong, Chung & Glass (2021). AST: Audio Spectrogram Transformer. Interspeech.
8. Elizalde et al. (2023). CLAP: Learning audio concepts from natural language supervision. ICASSP.
