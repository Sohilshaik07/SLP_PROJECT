# Dataset Status & Provenance Record

This document records the exact provenance, access status, schema, and experimental role of every dataset evaluated or considered for this project.

---

## 1. ACTIVE DATASET: HOLD-Telugu (DravidianLangTech 2024)

| Property | Details |
|---|---|
| **Dataset Title** | HOLD-Telugu: Hate and Offensive Language Detection in Telugu-English Code-Mixed Text |
| **Paper Citation** | Sai et al. (2024). *"HOLD-Telugu: A Corpus of Hate and Offensive Language Detection in Telugu-English Code-Mixed Text"*. In Proceedings of the 4th Workshop on Speech and Language Processing for Dravidian Languages (DravidianLangTech 2024 @ EACL 2024), pages 50–57. ([ACL Anthology 2024.dravidianlangtech-1.8](https://aclanthology.org/2024.dravidianlangtech-1.8/)) |
| **Data Source Repository** | [Salman1804102/DravidianLangTech-EACL-2024-HOLD](https://github.com/Salman1804102/DravidianLangTech-EACL-2024-HOLD) (Repository from which dataset files were obtained) |
| **Language & Format** | Telugu-English Code-Mixed (YouTube comments) |
| **Raw Dataset Files** | `data/raw/hold_telugu/training_data_telugu-hate.xlsx` (4,000 samples)  <br> `data/raw/hold_telugu/telugu-hate-speech-test.xlsx` (500 samples, unlabeled) |
| **Label Scheme (Train)** | `hate` (1,939 samples, 48.48%) vs `non-hate` (2,061 samples, 51.52%) |
| **Internal Eval Split** | Stratified 80% Train (3,151) / 10% Val (394) / 10% Test (394) (seed=42) created from 3,939-sample clean supervised corpus. See deduplication policy below. |
| **Official Test File** | Unlabeled (500 samples). Used for inference only; excluded from supervised metric calculation. |
| **Git Exclusion** | `data/raw/` and `*.xlsx` are ignored via `.gitignore` to prevent committing raw data. |
| **Current Status** | **ACTIVE, VERIFIED & CLEANED.** Clean experimental corpus established 2026-08-18. |

### Deduplication & Conflict Resolution Policy (finalized 2026-08-18)

A systematic audit of the raw training data identified 49 unique comment texts that appeared more than once. Of these, 10 groups contained contradictory hate/non-hate annotations — the same comment text was annotated as both `hate` and `non-hate` by the original annotators.

Rather than arbitrarily selecting one label (e.g., keep-first) or manually overriding the original annotations, **all occurrences belonging to these conflicting groups were excluded entirely from the supervised experimental corpus.** The original raw dataset was preserved unchanged.

The remaining 39 consistent-duplicate groups were further deduplicated by keeping one copy (the first occurrence).

**This policy is conservative:** it sacrifices 61 rows (20 from conflicting groups + 41 consistent duplicates) to ensure that no contradictory evidence enters the training or evaluation data.

| Audit Finding | Count |
|---|---|
| Raw training rows | 4,000 |
| Unique texts appearing >1 time | 49 |
| Consistent-label duplicate groups (dedup'd, keep first) | 39 → 41 rows removed |
| Conflicting-label duplicate groups (ALL rows removed) | 10 → 20 rows removed |
| Total rows removed | 61 |
| **Final clean corpus** | **3,939** |
| Raw file modified | **No — raw Excel preserved unchanged** |

### Clean Experimental Corpus

| Property | Value |
|---|---|
| Clean corpus size | 3,939 samples |
| non-hate | 2,028 (51.49%) |
| hate | 1,911 (48.51%) |
| Train (80%) | 3,151 — non-hate: 1,622, hate: 1,529 |
| Validation (10%) | 394 — non-hate: 203, hate: 191 |
| Test (10%) | 394 — non-hate: 203, hate: 191 |
| Leakage | None (verified) |
| Conflicting labels in any split | None (verified) |
| Seed | 42 |
| Official Unlabeled Test | 500 rows (excluded from supervised metric calculation) |

### Supervised Model Evaluation Summary

#### ✅ Clean HOLD-Telugu Corpus Results (Authoritative)
- **TF-IDF + Logistic Regression:** Macro F1 = **0.8189**, Accuracy = 0.8198
- **TF-IDF + Linear SVM:** Macro F1 = **0.7879**, Accuracy = 0.7893
- **XLM-RoBERTa Base (Tesla T4 GPU):** Macro F1 = **0.7787**, Accuracy = 0.7792 (Best Val Macro F1 = **0.7507**)

#### 🗄 Pre-Cleaning Corpus Results (Archived)
- **TF-IDF + Logistic Regression:** Macro F1 = 0.7715, Accuracy = 0.7722
- **TF-IDF + Linear SVM:** Macro F1 = 0.7638, Accuracy = 0.7646

---

## 2. PRIMARY DATASET TARGET: HASOC 2021 English-Hindi Code-Mix

| Property | Details |
|---|---|
| **Official Webpage** | [HASOC 2021 Datasets](https://hasocfire.github.io/hasoc/2021/dataset.html) |
| **Task Webpage** | [HASOC 2021 Shared Task](https://hasocfire.github.io/hasoc/2021/call_for_participation.html) |
| **Task / Subtask** | Subtask 2: Hindi-English Code-Mixed Hate Speech & Offensive Content Identification |
| **Data Platform** | Twitter / Social Media |
| **Access Status** | **Access Requested** (Password protected per HASOC terms) |
| **Local File Target** | `data/raw/hasoc2021/` |
| **Annotation Scheme** | `NOT` (Not Hate/Offensive) vs `HOF` (Hate / Offensive Content) |
| **Intended Role** | Primary dataset for Review 1 & Review 2 in-domain and benchmark evaluation |
| **Current Status** | Awaiting access credentials. Generic auto-detecting loader built (`src/data/loaders.py`). |

---

## 2. REFERENCE BENCHMARK: Bohra et al. 2018

| Property | Details |
|---|---|
| **Paper** | "A Dataset of Hindi-English Code-Mixed Social Media Text for Hate Speech Detection" ([ACL W18-1105](https://aclanthology.org/W18-1105/)) |
| **Repository** | [deepanshu1995/HateSpeech-Hindi-English-Code-Mixed-Social-Media-Text](https://github.com/deepanshu1995/HateSpeech-Hindi-English-Code-Mixed-Social-Media-Text) |
| **Local File** | `data/raw/bohra/id_annotated.tsv` (Saved for reference) |
| **File Format** | Tweet IDs + Labels (`Hate Speech` / `Normal Speech`) — 1,193 rows |
| **Access Restriction** | **Tweet IDs only.** Tweet text is omitted by authors due to Twitter ToS. |
| **Hydration Status** | Cannot be hydrated without paid Twitter Academic API (~$5,000/mo in 2026). |
| **Intended Role** | Referenced benchmark in literature survey & research question formulation. |
| **Current Status** | Preserved under `data/raw/bohra/id_annotated.tsv`. Not used for model training. |

---

## 3. INACCESSIBLE / CANDIDATE DATASETS

### HECM 2024 (ICON 2024)
- **Paper:** "A Corpus of Hindi-English Code-Mixed Posts to Hate Speech Detection" ([ACL 2024.icon-1.9](https://aclanthology.org/2024.icon-1.9/))
- **Claimed Repository:** `imprasshant/ICON` (Returns HTTP 404 — repository deleted/private)
- **Status:** **Inaccessible**. Documented as pending author response.

### Hugging Face `sourabh5500/hate-speech-muril`
- **Type:** Fine-tuned Model repository, NOT a dataset repository.
- **Status:** **Rejected**. Unverified provenance; gated API access. Must not be used as training dataset.

### HOLD-Telugu (DravidianLangTech 2024)
- **Paper:** [ACL 2024.dravidianlangtech-1.8](https://aclanthology.org/2024.dravidianlangtech-1.8/)
- **Language:** Telugu-English Code-Mixed
- **Status:** Candidate for Review 2 cross-language evaluation if access is verified.

---

## 4. DEMO DATASET (Software Testing Only)

| Property | Details |
|---|---|
| **Location** | `data/demo/demo_dataset.csv` |
| **Count** | 20 manually constructed Hinglish examples |
| **Labels** | `NOT` vs `HOF` |
| **Purpose** | **Software pipeline verification only.** |
| **Notice** | All outputs generated using demo data are strictly marked `DEMO DATA — NOT PROJECT RESULTS`. |
