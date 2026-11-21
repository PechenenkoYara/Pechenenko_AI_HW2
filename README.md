# UCU AI 2026 — HW3: human / ai / fusion text classification

Four Kaggle notebooks, run in this order. Each one reads the competition data and the outputs of the previous ones via **+ Add Input**.

| # | Notebook | Accelerator | Internet | Extra inputs | Main outputs | Est. time |
|---|----------|-------------|----------|--------------|--------------|-----------|
| 01 | `01_eda_split_baseline` | None (CPU) | Off | — | `folds.csv`, `val/test_baseline.csv`, `submission_baseline.csv`, `figs/` | 30–60 min |
| 02 | `02_transformer` | GPU T4 x2 | **On** | output of 01 | `val/test_mdeberta.csv`, `submission_mdeberta.csv`, `long_text_results_mdeberta.csv` | 2–3 h |
| 03 | `03_perplexity_stylometry` | GPU T4 x2 | **On** | output of 01 | `val/test_ppl.csv`, `submission_ppl.csv`, `ppl_ablation.csv`, `figs/shap_*.png` | ~1 h |
| 04 | `04_ensemble` | None (CPU) | Off | outputs of 01, 02, 03 | `ensemble_results.csv`, `submission_ensemble.csv` | ~5 min |

02 and 03 are independent and can run at the same time.

## Fixed protocol
- Working sample: `N_SAMPLE=60000` rows of `train.csv`, stratified by label × lang × domain, `SEED=42`.
- Validation: fold 0 of a 5-fold `StratifiedGroupKFold`. Groups are recovered as connected components of a near-duplicate graph (TF-IDF cosine ≥ 0.5 within lang+domain), so that a human text and its rewrites stay on the same side, as in the official train/test split.
- Every model is compared by macro-F1 on this same validation fold. Vectorizers and models are fit on the training folds only.
- The ensemble is tuned with 5-fold group CV *inside* the validation fold.

## Models
- **Baseline:** TF-IDF word 1–2-grams + char_wb 3–5-grams; LinearSVC / LogReg / ComplementNB; C grid; word-only vs char-only ablation.
- **Encoder:** `microsoft/mdeberta-v3-base` (switch to `xlm-roberta-base` if the loss becomes NaN in fp16), max_len 512, head+tail truncation for training; inference compares head+tail vs mean over sliding chunks vs both.
- **LM features:** `Qwen/Qwen2.5-0.5B` token log-probs and entropies, analytic Fast-DetectGPT score (Bao et al., 2024), 64-token window spread features + stylometry → LightGBM; SHAP.
- **Ensemble:** weighted average of log-probs, per-class bias for macro-F1, stacking LogReg.
