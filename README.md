# Robustness and Generalization of Multilingual Transformer Models for Hate Speech Detection in Indian Code-Mixed Text

**Course:** BCSE419P Speech and Language Processing Lab Project  
**Author / Student:** University SLP Project  

---

## 1. Project Title
**"Robustness and Generalization of Multilingual Transformer Models for Hate Speech Detection in Indian Code-Mixed Text"**

---

## 2. Problem Statement
Social media content in South Asia, particularly in India, heavily features **code-mixing**—the fluid blending of two or more languages within a single utterance (e.g., Hindi written in Roman script mixed with English words, commonly known as *Hinglish*). Detecting hate speech and offensive content in code-mixed text is challenging due to non-standard orthography, phonetic transliteration variations, dynamic slang, and lack of standardized resources.

---

## 3. Research Motivation
While pre-trained multilingual Transformer models (e.g., XLM-RoBERTa, mBERT, MuRIL) achieve competitive accuracy on in-domain test sets, social media data distributions constantly drift. Real-world deployment requires understanding:
- How reliably do models generalize across independently constructed code-mixed datasets?
- How sensitive are multilingual encoders to transliteration/spelling variations?
- What linguistic patterns cause systemic model errors?

---

## 4. Central Research Question
> **"How robustly do multilingual Transformer models detect problematic hate/offensive content in Indian code-mixed text when the dataset, language, or linguistic representation changes?"**

### Supporting Questions
1. Which architecture performs best under standard in-domain testing?
2. How much performance drop occurs under cross-dataset testing?
3. Can models fine-tuned on Hindi-English code-mixed text transfer to other Indian code-mixed languages (e.g., Telugu-English)?
4. How sensitive are models to common spelling and transliteration variations?
5. What error modes dominate false positives and false negatives?

---

## 5. Research Gap
Existing literature addresses Indian code-mixed hate speech classification primarily as a static benchmark task. Few studies systematically evaluate **cross-dataset generalization drops** and **representation robustness** using controlled comparative frameworks and explainability-guided error analysis.

---

## 6. Student-Level Contribution
A controlled comparative robustness and generalization evaluation of multilingual Transformer models (starting with XLM-RoBERTa base) across independently constructed Indian code-mixed datasets and linguistic representation variations, supported by explainability-based error analysis.

---

## 7. Dataset Strategy & Provenance

### Primary Dataset Target: HASOC 2021 English-Hindi Code-Mix
- **Source:** [HASOC 2021 Shared Task (Subtask 2)](https://hasocfire.github.io/hasoc/2021/dataset.html)
- **Language:** Hindi-English Code-Mixed (Social Media / Twitter)
- **Task Definition:** Binary classification — `HOF` (Hate/Offensive Content) vs. `NOT` (Not Hate/Offensive)
- **Access Status:** Access requested from organizers. Awaiting file delivery into `data/raw/hasoc2021/`.

### Referenced Benchmark: Bohra et al. 2018
- **Source:** ACL W18-1105 ([GitHub Repo](https://github.com/deepanshu1995/HateSpeech-Hindi-English-Code-Mixed-Social-Media-Text))
- **Status:** Raw repo contains 1,193 tweet IDs (`id_annotated.tsv`). Tweet text is omitted per Twitter ToS. Preserved for literature benchmark context under `data/raw/bohra/`.

### Software Verification (Demo Data)
- **Location:** `data/demo/demo_dataset.csv`
- **Purpose:** Allows full execution and verification of all pipeline scripts prior to receiving raw dataset files.
- **Notice:** All demo outputs are explicitly tagged `DEMO DATA — NOT PROJECT RESULTS`.

For complete details on all audited candidate datasets, see [`DATASET_STATUS.md`](file:///c:/Users/SKSOH/OneDrive/Desktop/SLP_PROJECT/DATASET_STATUS.md).

---

## 8. Repository Structure

```
SLP_PROJECT/
├── configs/
│   └── config.yaml                 # Central hyperparameter & path configuration
├── data/
│   ├── raw/
│   │   ├── hasoc2021/              # Official HASOC 2021 dataset files (place here)
│   │   └── bohra/                  # Bohra reference tweet IDs
│   ├── processed/                  # Preprocessed CSVs & train/val/test splits
│   └── demo/                       # Demo dataset (20 examples for pipeline testing)
├── models/
│   ├── baselines/                  # Saved TF-IDF + LR / SVM joblib models
│   └── xlmr/                       # Fine-tuned XLM-RoBERTa checkpoints & tokenizer
├── results/
│   ├── figures/                    # Class distributions, text length plots, confusion matrices
│   ├── metrics/                    # JSON metrics & comparison CSVs
│   └── predictions/                # CSV predictions with probabilities
├── src/
│   ├── data/
│   │   ├── preprocessing.py        # Code-mix-preserving text cleaning
│   │   └── loaders.py              # Generic auto-detecting dataset loader & splitter
│   ├── models/
│   │   ├── baselines.py            # TF-IDF + Logistic Regression & Linear SVM Pipelines
│   │   └── xlmr_model.py           # XLM-RoBERTa fine-tuning & inference classifier
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, Macro F1, Precision, Recall, Confusion Matrix
│   │   └── robustness.py           # Robustness drop calculation & perturbation hooks
│   └── utils/
│       ├── seed.py                 # Reproducibility seed setter & PyTorch GPU detector
│       └── config.py               # YAML configuration loader
├── scripts/
│   ├── download_data.py            # Data ingestion & verification CLI
│   ├── run_eda.py                  # Exploratory Data Analysis CLI
│   ├── train_baselines.py          # Baseline model training CLI
│   ├── train_xlmr.py               # XLM-R fine-tuning CLI
│   └── compare_models.py           # Metrics aggregation & comparison CLI
├── app/
│   └── streamlit_app.py            # Interactive Streamlit Web Application
├── README.md
├── DATASET_STATUS.md
├── EXPERIMENTS.md
├── requirements.txt
└── .gitignore
```

---

## 9. Installation & Setup

1. **Clone / Open Workspace:**
   ```bash
   cd c:\Users\SKSOH\OneDrive\Desktop\SLP_PROJECT
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Environment:**
   Ensure PyTorch, Transformers, scikit-learn, and Streamlit are accessible.

---

## 10. Preprocessing Methodology
Preprocessing (`src/data/preprocessing.py`) follows code-mix-preserving principles:
- **Preserved:** Devanagari script, Roman script, Hindi words, English words, emojis, hashtag terms.
- **Cleaned:** HTTP/HTTPS URLs, @user mentions, excess whitespace, control characters.
- **Forbidden:** No translation to English, no stopword removal, no stemming/lemmatization (which destroy code-mixed morphological markers).
- Both `raw_text` and `cleaned_text` are stored side-by-side.

---

## 11. How to Run the Project Pipeline

### Step 1: Data Ingestion & Splitting
```bash
# To process official HASOC 2021 data (when placed in data/raw/hasoc2021/):
python scripts/download_data.py

# To run software pipeline verification on DEMO data:
python scripts/download_data.py --demo
```

### Step 2: Exploratory Data Analysis
```bash
python scripts/run_eda.py [--demo]
```

### Step 3: Train Baseline Models (TF-IDF + LR & SVM)
```bash
python scripts/train_baselines.py [--demo]
```

### Step 4: Fine-Tune XLM-RoBERTa Transformer
```bash
python scripts/train_xlmr.py [--demo]
```

### Step 5: Generate Model Comparison Summary
```bash
python scripts/compare_models.py
```

### Step 6: Launch Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```

---

## 12. Review 1 Status & Verification Checklist

- [x] Full project directory structure & Python package layout built.
- [x] Code-mix-preserving preprocessing module implemented.
- [x] Auto-detecting dataset loader & stratified splitting module implemented.
- [x] TF-IDF + Logistic Regression & Linear SVM baselines implemented.
- [x] XLM-RoBERTa fine-tuning classifier implemented with GPU detection & checkpointing.
- [x] Evaluation metrics (Macro F1, Precision, Recall, Confusion Matrix) implemented.
- [x] Streamlit web application implemented.
- [x] Software pipeline verified using isolated demo dataset.
- [ ] **Awaiting HASOC 2021 Dataset Access** to execute formal training and report final metrics.

---

## 13. Review 2 Extension Plan
1. Integration of HASOC 2021 dataset upon credential delivery.
2. Cross-dataset evaluation against secondary Indian code-mixed datasets.
3. Secondary Transformer comparison (mBERT, MuRIL).
4. Representation robustness testing (controlled spelling/transliteration perturbations).
5. Token attribution XAI analysis for error diagnosis.

---

## 14. Citations & References
- **HASOC 2021:** Modha et al. (2021). *Overview of the HASOC track at FIRE 2021: Hate Speech and Offensive Content Identification in Indo-European Languages.*
- **Bohra 2018:** Bohra et al. (2018). *A Dataset of Hindi-English Code-Mixed Social Media Text for Hate Speech Detection.* ACL W18-1105.
- **XLM-RoBERTa:** Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.
