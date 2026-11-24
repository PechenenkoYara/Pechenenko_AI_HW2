# UCU AI 2026 — HW3: human / ai / fusion text classification

Kaggle notebooks with saved outputs. Run in this order; each one reads the competition data and the outputs of earlier notebooks via **+ Add Input**. All random steps use `SEED = 42`.

| # | Notebook | Accelerator | Internet | Extra inputs | Main outputs | Run time |
|---|----------|-------------|----------|--------------|--------------|----------|
| 01 | `01_eda_split_baseline` | CPU | Off | — | `folds.csv`, `val/test_baseline.csv`, `submission_baseline.csv`, `figs/` | ~50 min |
| 02 | `02_transformer_mdeberta` | GPU T4 x2 | On | output of 01 | `val/test_mdeberta.csv`, `submission_mdeberta.csv` | ~3.6 h |
| 02b | `02b_transformer_xlmr` (same code, `MODEL_NAME="xlm-roberta-base"`, `TAG="xlmr"`) | GPU T4 x2 | On | output of 01 | `val/test_xlmr.csv`, `submission_xlmr.csv` | ~2.6 h |
| 03 | `03_perplexity_stylometry` | GPU T4 x2 | On | output of 01 | `val/test_ppl.csv`, `submission_ppl.csv`, `figs/shap_*.png` | ~1.5 h |
| 04 | `04_ensemble` | CPU | Off | outputs of 01, 02, 02b, 03 | `ensemble_results.csv`, `submission_ensemble.csv` | ~5 min |

02, 02b and 03 are independent and can run in parallel.

## Results (macro-F1)

| Model | Validation (fold 0) | Public LB |
|---|---|---|
| 03 LM features + stylometry → LightGBM | 0.7885 | 0.79401 |
| 01 TF-IDF word+char → LogReg | 0.8794 | 0.86767 |
| 02b XLM-R base (head+tail ⊕ chunk mean) | 0.9169 | — |
| 02 mDeBERTa-v3 base (head+tail ⊕ chunk mean) | 0.9267 | 0.92450 |
| 04 stacking, 3 models (01, 02, 03) | 0.9419 (CV) | 0.93935 |
| **04 stacking, 4 models (01, 02, 02b, 03)** | **0.9450 (CV)** | **0.94343** |

## Protocol
- Fixed working sample: 60 001 rows of `train.csv`, stratified by label × lang × domain.
- Validation = fold 0 of `StratifiedGroupKFold(5)`. Groups are connected components of a near-duplicate graph (TF-IDF cosine ≥ 0.5 within lang+domain), so a human text and its rewrites stay on one side, as in the official split.
- Vectorizers and models are fit on folds 1–4 only; validation and test are only transformed.
- The ensemble is tuned with 5-fold group CV inside fold 0.

## References
GhostWriter dataset (Schaaf, Bönisch, Mehler, LREC 2026) · mDeBERTa-v3 (He et al.) · XLM-R (Conneau et al., 2020) · Qwen2.5 (Qwen Team, 2024) · Fast-DetectGPT (Bao et al., ICLR 2024) · LightGBM (Ke et al., 2017) · SHAP (Lundberg & Lee, 2017). Code was written with the help of an AI coding assistant.
