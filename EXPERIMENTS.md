# Experiment Log & Results Registry

This document records all formal experiments executed in the project.

> **RULE:** Every numerical value recorded in this log must come from an executed script on a verified dataset. No fabricated metrics are permitted.

---

## Review 1 Experiments

### Experiment A-1: In-Domain Evaluation on HOLD-Telugu Dataset (DravidianLangTech 2024)
- **Dataset:** HOLD-Telugu (`training_data_telugu-hate.xlsx`, 4,000 raw → 3,939 clean after conflict-removal + dedup)
- **Deduplication:** 10 conflicting-label duplicate groups removed (20 rows); 39 consistent-duplicate groups dedup'd (41 rows). Raw file unchanged.
- **Training Data:** 80% (3,151 samples: 1,622 non-hate, 1,529 hate)
- **Validation Data:** 10% (394 samples: 203 non-hate, 191 hate)
- **Test Data:** 10% Internal Held-Out (394 samples: 203 non-hate, 191 hate)
- **Target Task:** Binary Classification (`hate` [1] vs `non-hate` [0])
- **Random Seed:** `42`
- **Models Evaluated:**
  1. TF-IDF + Logistic Regression
  2. TF-IDF + Linear SVM
  3. XLM-RoBERTa Base (`xlm-roberta-base`) (Tesla T4 GPU, best val Macro F1 = 0.7507)

#### ✅ CLEAN CORPUS RESULTS (authoritative — completed 2026-08-18)

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | AUC-ROC | Metrics File |
|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | **0.8198** | **0.8218** | **0.8185** | **0.8189** | **0.8193** | **0.8762** | `results/metrics/HOLD_TELUGU_tfidf_lr_hold_telugu_indomain_clean_metrics.json` |
| TF-IDF + Linear SVM | **0.7893** | **0.7928** | **0.7875** | **0.7879** | **0.7884** | **0.8556** | `results/metrics/HOLD_TELUGU_tfidf_svm_hold_telugu_indomain_clean_metrics.json` |
| XLM-RoBERTa Base | **0.7792** | **0.7857** | **0.7814** | **0.7787** | **0.7783** | **0.8377** | `results/metrics/HOLD_TELUGU_xlmr_indomain_clean_metrics.json` |

#### 🗄 PRE-CLEANING RESULTS (archived, do not cite — computed before conflict removal)

| Model | Test Accuracy | Macro F1 | Metrics File |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.7722 | 0.7715 | `results/metrics/HOLD_TELUGU_tfidf_lr_PRE_CLEANING_indomain_metrics.json` |
| TF-IDF + Linear SVM | 0.7646 | 0.7638 | `results/metrics/HOLD_TELUGU_tfidf_svm_PRE_CLEANING_indomain_metrics.json` |

---

### Experiment A-2: In-Domain Evaluation on HASOC 2021 (Pending Dataset Delivery)
- **Training Data:** HASOC 2021 Hindi-English Code-Mix Train (80%)
- **Validation Data:** HASOC 2021 Validation (10%)
- **Test Data:** HASOC 2021 Held-Out Test (10%)
- **Target Task:** Binary Classification (`HOF` vs `NOT`)
- **Random Seed:** `42`

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Metrics File |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | `results/metrics/tfidf_lr_hasoc2021_indomain_metrics.json` |
| TF-IDF + Linear SVM | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | `results/metrics/tfidf_svm_hasoc2021_indomain_metrics.json` |
| XLM-RoBERTa Base | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | *Awaiting Dataset* | `results/metrics/xlmr_hasoc2021_indomain_metrics.json` |

---

## Review 2 Experiments (Hooks Ready)

- **Experiment B:** Secondary Transformer Architecture Comparison (mBERT / MuRIL)
- **Experiment C:** Cross-Dataset Generalization
- **Experiment D:** Cross-Language Transfer (Hindi-English → Telugu-English, if data obtained)
- **Experiment E:** Representation Robustness under Spelling & Script Perturbations
- **Experiment F:** Error Analysis & Token Attribution (XAI)
