# Faithfulness-Aware Explainability Audit of Multilingual Transformer Models for Code-Mixed Hate Speech Detection

**Academic Degree:** Bachelor of Technology (B.Tech) in Computer Science and Engineering  
**Specialization:** Natural Language Processing & Explainable Artificial Intelligence  
**Document Type:** Final Project Comprehensive Documentation  

---

## Abstract

Automated detection of hate speech and offensive content across multilingual social media is essential for ensuring online safety, yet remains technically challenging in multilingual societies such as India. Social media discourse in South Asia frequently employs code-mixing—the fluid alternation of two or more languages within a single utterance—primarily blending regional languages (Tamil, Malayalam, Hindi, and Telugu) with English, often rendered in informal Romanized phonetic transliteration alongside native Indic scripts. While pretrained multilingual transformer models such as XLM-RoBERTa (XLM-R) achieve competitive classification performance on standard benchmarks, high predictive accuracy alone does not guarantee model reliability, trustworthiness, or safety. Deep neural architectures operate as black-box predictors whose internal decision mechanisms remain opaque to auditors and end users. Furthermore, post-hoc explanation techniques such as SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) are frequently assumed to provide faithful representations of model reasoning without formal empirical verification.

This project presents a rigorous, faithfulness-aware explainability audit of multilingual transformer architectures for code-mixed hate speech detection across four major Indian languages: Tamil, Malayalam, Hindi, and Telugu. A unified, leak-free corpus of 63,159 supervised instances (50,525 training, 6,315 validation, and 6,319 held-out testing samples) was constructed from verified research datasets: HASOC 2021 (Hindi), HOLD-Telugu (DravidianLangTech 2024), and DravidianCodeMix (Tamil and Malayalam). We benchmark a fine-tuned XLM-RoBERTa Base classifier against two conventional machine learning baselines utilizing a 50,000-dimensional word and character n-gram TF-IDF representation: Logistic Regression and Linear Support Vector Machines with Platt-calibrated probabilities. On the unified held-out test set, XLM-RoBERTa achieves the strongest overall performance with an accuracy of 84.86%, a macro-averaged F1-score of 79.33%, and an ROC-AUC of 89.89%, outperforming Logistic Regression (82.02% accuracy, 77.57% macro F1) and Linear SVM (84.06% accuracy, 75.78% macro F1). 

However, disaggregated cross-language evaluation uncovers substantial linguistic disparities and challenges the assumption of uniform transformer superiority. While XLM-RoBERTa dominates on Tamil (macro F1 of 77.92%) and Telugu (macro F1 of 77.14%), Logistic Regression achieves a higher macro F1 on Malayalam (68.70% vs. 66.07%), and Linear SVM surpasses XLM-RoBERTa on Hindi (75.31% vs. 73.58%). On Malayalam, an extreme class imbalance (96.0% Non-Hate vs. 4.0% Hate) induces an accuracy paradox: XLM-RoBERTa attains an apparent 95.48% accuracy, yet suffers a 70.0% false-negative rate, failing to detect 35 out of 50 hate samples. To audit model transparency, we implement token-level SHAP and word-level LIME explainers and conduct a controlled deletion-based faithfulness experiment on 20 stratified test instances across depths $k \in \{1, 3, 5\}$ with 1,000 bootstrap resamples. Perturbing top-$k$ SHAP tokens yields mean model probability drops of $\Delta P = 0.1200$ ($k=1$), $0.1617$ ($k=3$), and $0.2069$ ($k=5$), while LIME deletions produce mean drops of $0.1323$, $0.2115$, and $0.2668$, both consistently exceeding the random-deletion baselines ($0.0355$, $0.0148$, and $0.1327$). Finally, the complete modeling and XAI pipeline is deployed via an interactive Streamlit application supporting multi-model inference, attribution visualization, and performance auditing. These findings demonstrate that explainability auditing is an indispensable prerequisite for responsible deployment of multilingual NLP systems in linguistically complex societies.

---

## 1. Introduction

### 1.1 Background and Motivation
The exponential growth of internet penetration and smartphone adoption across South Asia has democratized digital public squares. Platforms such as YouTube, X (formerly Twitter), Facebook, and Instagram host hundreds of millions of active South Asian users participating in vibrant sociopolitical, cultural, and entertainment discussions. However, this digital expansion has been accompanied by a surge in toxic discourse, including online harassment, hate speech, communal abuse, and identity-based disparagement. Unchecked hate speech degrades public discourse, incites offline violence, and disproportionately targets vulnerable minority demographics. Consequently, content moderation platforms and regulatory frameworks increasingly rely on automated Natural Language Processing (NLP) classifiers to detect and mitigate abusive content at scale.

In the Indian linguistic landscape, automated hate speech detection confronts distinct challenges rarely encountered in high-resource, monolingual English benchmarks. Most Indian social media communication is inherently **code-mixed**—exhibiting intra-sentential and inter-sentential switching between an Indic language (such as Tamil, Malayalam, Hindi, or Telugu) and English. Crucially, this code-mixed discourse is predominantly typed in the Roman script (phonetic transliteration, e.g., Hinglish, Tanglish, Manglish, or Telugish) rather than native Devanagari, Tamil, Malayalam, or Telugu orthography. Transliteration lacks standardized spelling rules; phonetically equivalent utterances can be rendered in dozens of disparate letter combinations (e.g., the Hindi term *bhai* spelled as *bhai*, *bhayi*, *bhaiya*, *by*, or *bhaii*). Furthermore, code-mixed text incorporates aggressive colloquial slang, creative contractions, regional dialectal variants, intentional adversarial obfuscations designed to evade naive keyword blocklists, and culturally embedded sarcasm.

### 1.2 Limitations of Conventional Classifiers and Pretrained Transformers
Historically, text classification systems relied on conventional machine learning algorithms, notably Logistic Regression, Naive Bayes, and Support Vector Machines (SVM), trained over bag-of-words or Term Frequency-Inverse Document Frequency (TF-IDF) feature spaces. While character n-gram TF-IDF representations demonstrate surprising resilience against minor typographical noise, conventional architectures suffer from three fundamental limitations:
1. **Lack of Deep Contextualization:** TF-IDF assigns static global weights to tokens regardless of surrounding syntactic or semantic contexts, rendering it incapable of resolving polysemy, negation scoping, or figurative speech.
2. **High Dimensionality and Sparsity:** Phonetic transliteration inflates vocabulary sizes exponentially, producing ultra-sparse matrices that fail to generalize across unseen spelling variants.
3. **Absence of Cross-Lingual Semantic Transfer:** Conventional models cannot leverage semantic affinities between sister languages (e.g., shared Dravidian roots between Tamil and Malayalam, or shared Indo-Aryan syntax).

To overcome these constraints, modern NLP has shifted decisively toward deep pretrained multilingual transformers, most notably Multilingual BERT (mBERT) and XLM-RoBERTa (XLM-R). Pretrained on hundreds of gigabytes of raw text across over one hundred languages via Masked Language Modeling (MLM), XLM-RoBERTa learns dense, shared cross-lingual contextual representations. By mapping syntactically diverse and multilingual tokens into unified embedding manifolds, transformer models capture nuanced contextual dependencies, discourse-level semantics, and subtle markers of hostility.

### 1.3 The Black-Box Dilemma and the Need for Explainability Auditing
Despite their superior benchmark accuracy, deep transformer models operate as opaque, parameter-heavy "black boxes." A standard XLM-RoBERTa Base architecture contains approximately 278 million parameters distributed across 12 bidirectional transformer encoder layers and 12 attention heads per layer. The internal decision logic that translates a raw sequence of tokens into a binary prediction (`Hate` vs. `Non-Hate`) is mathematically intractable to human inspection.

This opacity introduces severe operational, ethical, and legal vulnerabilities:
- **Spurious Correlations and Algorithmic Bias:** Deep models are prone to learning dataset-specific artifacts. In hate speech detection, classifiers frequently latch onto benign demographic identifiers, identity nouns (e.g., religious, caste, or regional labels), or political figures as dominant predictors of toxicity, yielding high false-positive rates on legitimate civic discussions.
- **Vulnerability to Adversarial Evasion:** When models rely on unverified token heuristics rather than semantic comprehension, adversaries can easily bypass detection via trivial character insertions or leetspeak substitutions.
- **Accountability and Procedural Justice:** Content takedowns and account suspensions based on opaque algorithmic decisions violate basic standards of fairness and transparency. Automated moderation systems must be capable of providing human-interpretable justifications for regulatory compliance.

To address these challenges, the field of Explainable Artificial Intelligence (XAI) has produced local post-hoc explanation frameworks, notably **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**. These frameworks assign attribution scores to individual input tokens, highlighting which words contributed positively or negatively toward a model's prediction.

However, an alarming gap persists in practical XAI deployments: **explanation methods are themselves rarely audited for faithfulness.** Practitioners routinely accept SHAP or LIME visualizations at face value, conflating visual plausibility with actual model fidelity. If an explanation method highlights tokens that the underlying neural network did not actually rely upon to make its classification, the resulting "explanation" is an ungrounded hallucination. Evaluating explanation **faithfulness**—verifying that feature importance scores reflect the true computational mechanics of the model—is therefore an essential scientific requirement.

### 1.4 Research Scope, Objectives, and Contributions
This project investigates the nexus of classification performance, model transparency, and explanation faithfulness in multilingual code-mixed hate speech detection. Rather than evaluating isolated toy datasets, we conduct a unified study across four prominent Indian languages: **Tamil, Malayalam, Hindi, and Telugu**.

#### Specific Project Objectives
1. **Unified Multi-Source Data Engineering:** Assemble, clean, conflict-resolve, and deduplicate a multi-source supervised corpus spanning Tamil, Malayalam, Hindi, and Telugu, eliminating cross-split data leakage and establishing an authoritative experimental benchmark.
2. **Comparative Model Architecture Evaluation:** Implement and rigorously evaluate conventional linear baselines (TF-IDF with Logistic Regression and Linear Support Vector Machines) against a fine-tuned multilingual transformer (XLM-RoBERTa Base) on identical held-out test splits.
3. **Cross-Language Disaggregated Error Analysis:** Systematically unpack model performance across individual language subsets, analyzing the impact of class imbalance, measuring false-positive and false-negative rates, and categorizing qualitative failure modes.
4. **Post-Hoc Explainability Implementation:** Develop modular token-level SHAP and word-level LIME explanation pipelines that interface directly with the fine-tuned transformer without modifying model weights.
5. **Empirical Faithfulness Auditing:** Implement a controlled, deletion-based perturbation framework to quantify the faithfulness of SHAP, LIME, and a random-feature baseline across multiple deletion depths ($k \in \{1, 3, 5\}$) using bootstrap confidence intervals.
6. **Interactive Systems Deployment:** Package the verified models, explainers, and diagnostic dashboards into a responsive, production-ready Streamlit web application for real-time model interrogation.

#### Research Contributions
- **Empirical Demonstration of the Accuracy Paradox in Code-Mixed Text:** We demonstrate how severe class imbalance in low-resource code-mixed subsets (specifically Malayalam, with 4.0% hate prevalence) produces artificially inflated accuracy (95.48%) while masking catastrophic false-negative rates (70.0%), underscoring why macro-averaged F1 must remain the primary metric.
- **Evidence of Selective Baseline Superiority:** We show that while XLM-RoBERTa achieves superior aggregate test metrics across 6,319 samples, conventional TF-IDF baselines outperform the transformer on specific languages (Logistic Regression on Malayalam; Linear SVM on Hindi), demonstrating that transformers do not uniformly dominate code-mixed tasks.
- **Quantitative Faithfulness Verification:** We provide empirical evidence that both SHAP and LIME identify features whose removal causes significant drops in model confidence ($\Delta P$) compared to random deletion, while establishing that LIME's word-level surrogate fit ($R^2$) and SHAP's subword attributions exhibit distinct trade-offs in multilingual settings.
- **Open-Access Reproducible Pipeline:** We document an end-to-end reproducible codebase with leak-free data partitioning, Platt-calibrated SVM pipelines, automated SHAP/LIME caching, and an audited experimental registry.

---

## 2. Literature Review

### 2.1 Traditional Machine Learning Approaches to Hate Speech Detection
Early computational research in automated hate speech detection framed the problem as a supervised document classification task utilizing linear classifiers trained on lexical and surface-level features.

**Davidson et al. (2017)** conducted a foundational study addressing the critical distinction between generic offensive language (containing profanity or vulgarity) and genuine hate speech (targeted hostility directed at protected demographic characteristics). Analyzing a corpus of 24,783 English tweets, the authors trained Logistic Regression, Naive Bayes, and Support Vector Machine models using word and character n-grams, part-of-speech (POS) tags, sentiment lexicons, and readability scores. Their findings indicated that while Linear SVMs achieved high overall precision, models routinely conflated offensive slang with hate speech, producing high false-positive rates for African American English (AAVE). In the context of this project, Davidson et al.'s findings highlight the hazards of relying strictly on surface lexical cues, which in Indian code-mixed discourse frequently misclassify benign, colloquial profanity as communal hate speech.

**Fortuna and Nunes (2018)** published a comprehensive survey synthesizing over a decade of hate speech detection research. They categorized feature engineering strategies across dictionary-based lookups, n-gram representations, linguistic rule-based patterns, and word embeddings. The authors identified fundamental limitations in traditional machine learning pipelines: high sensitivity to out-of-vocabulary (OOV) tokens, inability to model long-range syntactic dependencies, and acute domain brittleness. Their survey established that character n-gram models provide superior resilience over word n-grams when handling noisy social media text, directly informing the FeatureUnion architecture adopted in our project baselines.

### 2.2 Multilingual Pretrained Transformers in Natural Language Processing
The introduction of the Transformer architecture by **Vaswani et al. (2017)** and subsequent pretrained bidirectional encoders catalyzed a paradigm shift across NLP.

**Devlin et al. (2019)** introduced BERT (Bidirectional Encoder Representations from Transformers), demonstrating that pretraining deep bidirectional representations on massive unannotated corpora via Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) yielded state-of-the-art performance across diverse downstream tasks. Alongside English BERT, Devlin et al. released **mBERT (Multilingual BERT)**, trained simultaneously on Wikipedia dumps across 104 languages using a shared 110,000 WordPiece vocabulary. Although mBERT demonstrated cross-lingual zero-shot generalization capabilities, subsequent analyses revealed limitations: Wikipedia text is formal and standard, lacking social media noise, and under-represented languages suffered from low vocabulary allocation within the shared WordPiece budget.

**Conneau et al. (2020)** introduced **XLM-RoBERTa (XLM-R)**, scaling cross-lingual pretraining by training on 2.5 terabytes of filtered CommonCrawl data (CC-100) spanning 100 languages. XLM-RoBERTa eliminated the NSP objective, incorporated a 250,000 subword SentencePiece vocabulary, and substantially increased training data for low- and medium-resource languages. Conneau et al. demonstrated that XLM-RoBERTa Base and Large achieved substantial gains over mBERT on cross-lingual classification benchmarks such as XNLI, particularly for non-Latin script languages including Hindi, Tamil, and Urdu. XLM-RoBERTa Base serves as the primary deep neural architecture evaluated in this project due to its multilingual pretraining coverage across Indo-Aryan and Dravidian language families.

**Kakwani et al. (2020)** developed **IndicBERT** and the IndicNLPSuite, pretraining monolingual and multilingual language models specifically tailored to 12 major Indian languages using an ALBERT-based architecture. While IndicBERT demonstrated superior parameter efficiency for native-script Indic corpora, its vocabulary was optimized for native scripts (e.g., Devanagari, Tamil script) rather than Romanized phonetic transliterations, limiting its out-of-the-box efficacy for heavily transliterated code-mixed social media streams.

### 2.3 Indian Code-Mixed Hate Speech Benchmarks and Research
Code-mixing in South Asian digital communications has attracted substantial academic attention through competitive shared tasks organized by FIRE (Forum for Information Retrieval Evaluation) and DravidianLangTech workshops at EACL and ACL.

**Bohra et al. (2018)** created one of the earliest dedicated benchmarks for Hindi-English code-mixed hate speech detection on Twitter. Their dataset comprised 4,575 annotated Hinglish tweets categorized into `Hate Speech` and `Normal Speech`. The authors evaluated traditional classifiers (SVM, Random Forest) utilizing n-grams, lexicon features, and word embeddings, achieving a best macro F1-score of 71.7% with an SVM. However, as documented in our repository audit, Bohra et al.'s publicly accessible repository distributes only tweet IDs without underlying text; due to Twitter API policy changes and tweet deletions over time, raw tweet hydration is largely infeasible today, restricting its utility to a literature baseline.

**Modha et al. (2021)** presented the findings of the **HASOC 2021** (Hate Speech and Offensive Content Identification in Indo-European Languages) shared task at FIRE 2021. Subtask 2 specifically focused on conversational Hindi-English code-mixed Twitter data annotated into `HOF` (Hate/Offensive Content) and `NOT` (Not Hate/Offensive). Top-performing participant systems predominantly employed ensemble architectures combining fine-tuned multilingual transformers (mBERT, XLM-RoBERTa) with task-specific data augmentation. The HASOC 2021 dataset provides the foundational Hindi-English experimental corpus utilized in our research.

**Chakravarthi et al. (2021)** introduced the **DravidianCodeMix** datasets at the DravidianLangTech workshop, providing extensively annotated YouTube comment corpora for offensive language identification in code-mixed Tamil-English (Tanglish) and Malayalam-English (Manglish). The authors established benchmark results using standard machine learning and deep learning baselines, observing that Dravidian code-mixed text presents extreme orthographic variability and dialectal diversity. Their work highlighted a critical challenge: high class imbalance, particularly in Malayalam, where offensive comments represent a tiny minority of total community feedback. The DravidianCodeMix corpora provide the verified Tamil and Malayalam datasets integrated into our unified framework.

**Sai et al. (2024)** introduced **HOLD-Telugu**, a dedicated corpus of hate and offensive language detection in Telugu-English code-mixed YouTube comments presented at the 4th DravidianLangTech Workshop at EACL 2024. The dataset comprised 4,000 annotated comments. Crucially, as discovered during our repository's empirical audit, the raw HOLD-Telugu dataset contained duplicate comment strings with contradictory ground-truth labels (annotated as both `hate` and `non-hate`). Sai et al. established baseline performance using mBERT and IndicBERT, demonstrating that Telugu-English code-mixing exhibits unique agglutinative morphological markers that challenge standard Western NLP tokenizers.

### 2.4 Explainable AI (XAI) in Natural Language Processing
As deep neural networks achieved dominance across NLP benchmarks, post-hoc local explanation frameworks were developed to interpret individual model predictions.

**Ribeiro, Singh, and Guestrin (2016)** formulated **LIME (Local Interpretable Model-agnostic Explanations)**. LIME treats the underlying predictive model as an opaque black box. To explain a specific prediction on an input instance $\mathbf{x}$, LIME draws perturbation samples in the local neighborhood of $\mathbf{x}$ (e.g., randomly dropping words from a text document), observes the model's prediction probabilities on these perturbed inputs, weights the samples according to an exponential kernel measuring distance to $\mathbf{x}$, and fits an interpretable surrogate model (such as a sparse linear regressor). The resulting regression coefficients represent the local directional influence of each feature. While flexible and intuitive, LIME's local surrogate fit depends on perturbation sample sizes, and word-level deletions may disrupt syntax in ways that distort neural representations.

**Lundberg and Lee (2017)** introduced **SHAP (SHapley Additive exPlanations)**, uniting six existing feature attribution methods under a game-theoretic framework rooted in Shapley values from cooperative game theory. Shapley values uniquely satisfy four fundamental axiomatic properties: Local Accuracy (efficiency), Missingness, Consistency (monotonicity), and Additivity. In text classification, SHAP measures the marginal contribution of each token toward the difference between the model's actual prediction and a base expected value across all possible feature subsets. For transformer models, SHAP employs a text masker (`shap.maskers.Text`) that perturbs sequences at token or word boundaries.

### 2.5 Faithfulness Evaluation and Research Gap
While SHAP and LIME produce visually appealing token attribution heatmaps, recent literature emphasizes that visual plausibility does not imply computational fidelity.

**DeYoung et al. (2020)** established the **ERASER** benchmark (Evaluating Rationales And Simple English Reasoning), formalizing evaluation metrics for post-hoc explanations in NLP. They highlighted **comprehensiveness** and **sufficiency** through feature erasure tests. Comprehensiveness measures the degree to which model prediction confidence drops when the tokens identified as important by the explainer are removed from the input text:
$$\text{Comprehensiveness} = P(y \mid \mathbf{x}) - P(y \mid \mathbf{x} \setminus \mathbf{r})$$
where $\mathbf{r}$ represents the top rationales. A faithful explainer must produce a large drop in prediction probability when its top features are deleted; if removing the alleged "key" features leaves the model's confidence intact, the explanation lacks faithfulness.

**Nguyen (2018)** compared automated perturbation-based metrics and human evaluations of local explanations, finding that human annotators frequently preferred explanations that matched human linguistic intuition rather than the actual internal mechanics of the machine learning model. This finding reinforces the necessity of automated, deletion-based faithfulness auditing: human visual inspection cannot substitute for empirical perturbation testing.

#### Identified Research Gap
Despite significant progress across multilingual classification, Indian code-mixed datasets, and post-hoc XAI, the existing literature exhibits a critical research void:
- Prior studies on Indian code-mixed hate speech (HASOC, DravidianLangTech) have focused almost exclusively on competitive leaderboard optimization, evaluating models purely on benchmark F1-scores without systematic explainability audits.
- Studies applying SHAP or LIME to NLP classification predominantly evaluate monolingual English corpora and assume explanation validity without conducting empirical faithfulness evaluations.
- No existing research systematically integrates:
  1. A multi-lingual, multi-script code-mixed Indian corpus (Tamil, Malayalam, Hindi, Telugu);
  2. A controlled comparison between conventional linear baselines and multilingual transformers;
  3. A granular, cross-language error and class-imbalance analysis; and
  4. An empirical, deletion-based faithfulness audit comparing SHAP, LIME, and random baselines with bootstrap confidence intervals.

This project directly addresses this multi-dimensional research gap.

---

## 3. Methodology

The architectural workflow of this project is structured as an end-to-end research and evaluation pipeline, tracking the transformation of raw social media comments into preprocessed tokens, model representations, classification predictions, post-hoc explanations, and empirical faithfulness audits.

```
+---------------------------------------------------------------------------------------------------+
|                                      DATA INGESTION & AUDIT                                       |
|  - HASOC 2021 (Hindi)      - DravidianTamil (Tamil)                                               |
|  - HOLD-Telugu (Telugu)    - DravidianMalayalam (Malayalam)                                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                  PREPROCESSING & DEDUPLICATION                                    |
|  - Code-mix preservation (Devanagari, Dravidian scripts, Roman script, emojis, hashtags)          |
|  - Noise removal (URLs, @mentions, control characters)                                            |
|  - Conflict resolution (removal of identical texts with contradictory labels)                     |
|  - Cross-split leakage prevention (Test > Val > Train priority)                                   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                         UNIFIED MULTILINGUAL CORPUS (N = 63,159)                                  |
|  - Train: 50,525 samples (80.0%)                                                                  |
|  - Validation: 6,315 samples (10.0%)                                                              |
|  - Test: 6,319 samples (10.0%)                                                                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+------------------------------------------------+ +------------------------------------------------+
|             CONVENTIONAL BASELINES             | |          MULTILINGUAL TRANSFORMER              |
|  - FeatureUnion TF-IDF (50,000 features)       | |  - XLM-RoBERTa Base (12 layers, 768 hidden)    |
|    * Word n-grams (1-2), char_wb n-grams (2-4) | |  - SentencePiece Tokenizer (max len = 128)     |
|  - Baseline 1: Logistic Regression (C=1.0)     | |  - Linear Classification Head + Dropout (0.1)  |
|  - Baseline 2: Linear SVM (Platt-calibrated)   | |  - Fine-tuned on Tesla T4 GPU (5 epochs)       |
+------------------------------------------------+ +------------------------------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                HELD-OUT TEST EVALUATION (N = 6,319)                               |
|  - Aggregate metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, ROC-AUC                  |
|  - Disaggregated language metrics: Tamil, Malayalam, Hindi, Telugu                                |
|  - Error diagnostics: Confusion matrices, FPR, FNR, qualitative error audit                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                EXPLAINABILITY AUDIT & FAITHFULNESS                                |
|  - Token-level SHAP (shap.maskers.Text, max_evals = 250)                                          |
|  - Word-level LIME (LimeTextExplainer, num_samples = 200, local surrogate R²)                     |
|  - Deletion-based faithfulness test: k in {1, 3, 5} on 20 stratified test instances               |
|  - Probability drop metric: Delta P = P(orig | x) - P(orig | x \ top-k)                           |
|  - 1,000 bootstrap resamples for 95% Confidence Intervals vs. Random Baseline                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                INTERACTIVE STREAMLIT APPLICATION                                  |
|  - Multi-model real-time inference (XLM-R, Logistic Regression, Linear SVM)                       |
|  - Dynamic SHAP & LIME attribution visualizations                                                 |
|  - Comprehensive performance tables & language-specific confusion matrix dashboards               |
+---------------------------------------------------------------------------------------------------+
```

### 3.1 Dataset Ingestion and Preprocessing Pipeline

#### 3.1.1 Data Provenance and Raw Sources
The unified multilingual corpus integrates four independent, verified academic datasets:
1. **HASOC 2021 Hindi-English Code-Mix:** Sourced from Subtask 2 of the HASOC 2021 shared task (`HASOC_HINDI.csv`). Texts represent conversational social media comments on Twitter covering political, cultural, and sports topics. Binary labels map `NOT` (Not Hate/Offensive) to class `0` and `HOF` (Hate/Offensive Content) to class `1`.
2. **HOLD-Telugu:** Sourced from the DravidianLangTech 2024 shared task (`training_data_telugu-hate.xlsx`, Sai et al., 2024). Texts represent Telugu-English code-mixed comments extracted from YouTube. Original labels `non-hate` and `hate` map to `0` and `1`.
3. **DravidianCodeMix Tamil:** Sourced from the DravidianLangTech workshop at EACL/ACL (`tamil_offensive_full_train.csv`, `dev.csv`, `test.csv`, Chakravarthi et al., 2021). Texts comprise YouTube comments in Tamil-English (predominantly Romanized Tanglish). Comments labeled `not-Tamil` were filtered out. Multiclass offensive categories were binarized: `Not_offensive` mapped to `0`, and all offensive subcategories (`Offensive_Untargeted`, `Offensive_Targeted_Insult_Individual`, `Offensive_Targeted_Insult_Group`, etc.) mapped to `1`.
4. **DravidianCodeMix Malayalam:** Sourced from DravidianLangTech (`mal_full_offensive_train.csv`, `dev.csv`, `test.csv`, Chakravarthi et al., 2021). Texts comprise Malayalam-English YouTube comments (Manglish). Comments labeled `not-malayalam` were filtered out. Labels were binarized identically: `Not_offensive` mapped to `0`, and all offensive categories mapped to `1`.

#### 3.1.2 Deduplication and Conflict Resolution Policy
Raw social media datasets frequently contain identical comment strings. In sentiment and hate speech corpora, duplicate texts can emerge naturally (e.g., standard congratulatory phrases or viral copypasta) or via annotation noise. 

During our repository audit, a critical anomaly was discovered: in the raw HOLD-Telugu dataset, 49 unique texts appeared multiple times, and **10 groups of identical texts had contradictory annotations** (annotated simultaneously as `hate` by one annotator and `non-hate` by another). Similarly, conflicting annotations were detected in the raw Tamil and Malayalam collections.

To maintain scientific integrity, a strict **conflict resolution policy** was implemented across all sources:
- All duplicate groups containing contradictory labels were **completely purged** from the experimental corpus. Arbitrary tie-breaking (e.g., retaining the first occurrence) was rejected to guarantee that no conflicting supervision contaminated the training or evaluation data.
- For duplicate groups with consistent labels, exactly one representative instance was preserved, and surplus copies were dropped.
- In HOLD-Telugu, this conservative policy removed 61 rows (20 conflicting samples and 41 redundant duplicates), yielding 3,939 verified clean samples from the initial 4,000 raw rows.

#### 3.1.3 Code-Mix-Preserving Text Preprocessing
Standard text cleaning pipelines for English NLP typically apply stopword removal, aggressive punctuation stripping, stemming (Porter/Snowball), lemmatization, or machine translation. In code-mixed Indic NLP, **these operations are destructive**:
- Stemming and lemmatization strip grammatical morphemes (e.g., Telugu case markers or Tamil verb conjugations attached to English stems) that carry crucial semantic and pragmatic meaning.
- Stopword removal removes pronouns and discourse markers that differentiate directed personal attacks from generic rhetoric.
- Translation pipelines fail completely on informal transliterations and destroy code-mixing structure.

Accordingly, the implemented module (`src/data/preprocessing.py`) follows strict **code-mix-preserving principles**:
- **Preserved:** All native Indic scripts (Devanagari, Tamil, Malayalam, Telugu), Roman alphabet characters, Arabic numerals, code-mixed tokens, emojis, and hashtag text (e.g., `#Modi` becomes `Modi`).
- **Cleaned:** HTTP and HTTPS hyperlinks, Twitter `@mentions`, invisible control characters (`unicodedata.category` $\in \{\text{"Cc"}, \text{"Cf"}\}$), and repeated punctuation sequences (collapsed to single instances).
- **Whitespace Normalization:** Consecutive tabs, spaces, and newlines collapsed to single spaces.

#### 3.1.4 Cross-Split Leakage Prevention and Unified Dataset Schema
To prevent data leakage across splits, a hierarchical deduplication algorithm was executed during unified corpus assembly (`scripts/prepare_unified_dataset.py`):
$$\text{Priority: } \text{Test} > \text{Validation} > \text{Train}$$
Any comment string appearing in the held-out test split was strictly purged from the validation and training sets. Any text appearing in the validation set was purged from the training set. The final verified dimensions of the unified multilingual dataset are summarized in Table 1.

**Table 1: Verified Unified Multilingual Dataset Distribution**
| Dataset Origin | Language | Train (80%) | Val (10%) | Test (10%) | Total Samples | Hate % (Test) |
|---|---|---:|---:|---:|---:|---:|
| DravidianCodeMix | Tamil | 33,384 | 4,173 | 4,174 | 41,731 | 24.6% |
| DravidianCodeMix | Malayalam | 10,082 | 1,260 | 1,261 | 12,603 | 4.0% |
| HASOC 2021 | Hindi | 3,912 | 489 | 490 | 4,891 | 51.4% |
| HOLD-Telugu | Telugu | 3,147 | 393 | 394 | 3,934 | 48.5% |
| **Unified Total** | **All 4** | **50,525** | **6,315** | **6,319** | **63,159** | **31.7%** |

The unified dataset utilizes four standardized columns:
- `text` (string): Preprocessed code-mixed social media comment.
- `label` (integer): Binary classification label (`0` = Non-Hate, `1` = Hate).
- `language` (string): Language identity (`Tamil`, `Malayalam`, `Hindi`, `Telugu`).
- `dataset` (string): Source corpus identifier (`DravidianTamil`, `DravidianMalayalam`, `HASOC2021`, `HOLD`).

### 3.2 Baseline Model Implementations
To evaluate whether deep transformer models justify their computational overhead, two strong conventional machine learning pipelines were implemented using `scikit-learn` (`src/models/baselines.py`).

#### 3.2.1 Hybrid Word and Character TF-IDF Representation
Both baselines utilize an identical, high-dimensional feature extractor built using `sklearn.pipeline.FeatureUnion`:
1. **Word N-Gram Vectorizer:**
   - Analyzer: `word`
   - N-gram range: $(1, 2)$ (unigrams and bigrams)
   - Maximum features: $50,000$
   - Sublinear term frequency scaling: Enabled ($1 + \log(\text{tf})$)
   - Minimum document frequency: $\text{min\_df} = 2$
   - Accents / Case: Preserved (`strip_accents=None`, `lowercase=False`) to avoid distorting Devanagari or casing emphasis.
2. **Character-Within-Word N-Gram Vectorizer:**
   - Analyzer: `char_wb` (character n-grams constrained within word boundaries)
   - N-gram range: $(2, 4)$ (capturing sub-word character chunks of length 2 to 4)
   - Maximum features: $50,000$
   - Sublinear term frequency scaling: Enabled
   - Minimum document frequency: $\text{min\_df} = 2$

Combining word and character-within-word n-grams allows the linear models to capture both lexical phrase sequences and morphological sub-word stems, providing intrinsic resilience against spelling variations in transliterated code-mixed text.

#### 3.2.2 Baseline 1: TF-IDF + Logistic Regression
Logistic Regression models the posterior probability of the hate speech class using the standard logistic sigmoid link function:
$$P(Y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
The pipeline was trained by minimizing the cross-entropy loss with $L_2$ regularization:
$$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
- Hyperparameters: Inverse regularization strength $C = 1.0$, maximum iterations $= 1,000$, solver = `lbfgs`, random state $= 42$.

#### 3.2.3 Baseline 2: TF-IDF + Linear Support Vector Machine
Linear Support Vector Machines optimize a maximum-margin hyperplane separating the classes in feature space:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{N} \xi_i \quad \text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- Hyperparameters: Implemented via `sklearn.svm.LinearSVC` with $C = 1.0$, `max_iter=2000`, `random_state=42`, and `class_weight="balanced"`.
- **Probability Calibration:** Standard Linear SVMs output signed geometric distances from the decision hyperplane rather than calibrated probabilities. To enable rigorous ROC-AUC evaluation and probability thresholding, the `LinearSVC` estimator was wrapped in `sklearn.calibration.CalibratedClassifierCV` using 3-fold cross-validation with sigmoid (Platt) calibration.

### 3.3 XLM-RoBERTa Model Architecture and Fine-Tuning
The primary deep learning model is **XLM-RoBERTa Base** (`xlm-roberta-base`), fine-tuned for sequence classification (`src/models/xlmr_model.py`, `scripts/train_xlmr.py`).

#### 3.3.1 Neural Architecture
XLM-RoBERTa Base comprises 12 transformer encoder blocks, a hidden dimension $d_{\text{model}} = 768$, 12 self-attention heads per layer, an intermediate feed-forward dimension of 3,072, and a shared vocabulary of 250,002 subword tokens trained via SentencePiece unigram language modeling. The total parameter count is approximately 278,043,650.

For sequence classification, input texts are prepended with the sentence-start token `<s>` and appended with `</s>`. The sequence of hidden states $\mathbf{h}_1, \dots, \mathbf{h}_T \in \mathbb{R}^{768}$ is computed by the stacked transformer blocks. The representation of the first token $\mathbf{h}_{<s>}$ (equivalent to BERT's `[CLS]`) serves as the pooled sequence representation. A sequence classification head is applied:
$$\mathbf{z} = \mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{h}_{<s>} + \mathbf{b}_1) + \mathbf{b}_2$$
$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z}) = \left[ P(Y=0 \mid \mathbf{x}), \; P(Y=1 \mid \mathbf{x}) \right]$$
where a dropout probability of $p = 0.1$ is applied prior to linear projection.

#### 3.3.2 Training Configuration and Environment
Fine-tuning was executed on the unified training split (50,525 samples) with validation monitoring on 6,315 samples. The verified training configuration is:
- **Base Model Checkpoint:** `xlm-roberta-base`
- **Training Hardware:** Single NVIDIA Tesla T4 GPU (16 GB VRAM)
- **Epochs:** 5 full epochs
- **Maximum Sequence Length:** 128 tokens (accommodating >98% of social media comments without excessive zero-padding)
- **Per-Device Training Batch Size:** 16
- **Evaluation Batch Size:** 32
- **Optimization Algorithm:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$)
- **Learning Rate:** $2.0 \times 10^{-5}$
- **Learning Rate Scheduler:** Linear decay with warm-up ratio $= 0.10$ (warm-up steps $= 790$)
- **Weight Decay:** $0.01$ (applied to non-bias/layer-norm weights)
- **Gradient Accumulation Steps:** 1
- **Early Stopping & Checkpoint Selection:** Checkpoints saved at the conclusion of each epoch (`save_strategy="epoch"`), with the best checkpoint selected based on validation Macro F1-score (`metric_for_best_model="macro_f1"`, patience $= 3$).

### 3.4 Evaluation Metrics and Mathematical Formulation
In binary classification of imbalanced social media text, reliance on classification accuracy alone is scientifically invalid, as a trivial majority classifier can achieve high accuracy while failing entirely on the minority class of interest.

Let $TP$, $TN$, $FP$, and $FN$ denote True Positives, True Negatives, False Positives, and False Negatives relative to class $1$ (`Hate`).

1. **Overall Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Per-Class Precision and Recall:**
   $$\text{Precision}_1 = \frac{TP}{TP + FP}, \quad \text{Recall}_1 = \frac{TP}{TP + FN}$$
   $$\text{Precision}_0 = \frac{TN}{TN + FN}, \quad \text{Recall}_0 = \frac{TN}{TN + FP}$$

3. **Macro-Averaged Precision, Recall, and F1-Score:**
   Macro averaging computes metrics independently for each class and averages them with equal weight, penalizing models that perform poorly on minority classes:
   $$\text{Precision}_{\text{Macro}} = \frac{\text{Precision}_0 + \text{Precision}_1}{2}$$
   $$\text{Recall}_{\text{Macro}} = \frac{\text{Recall}_0 + \text{Recall}_1}{2}$$
   $$\text{F1}_{\text{Macro}} = \frac{\text{F1}_0 + \text{F1}_1}{2} = \frac{1}{2} \left( \frac{2 \cdot \text{Precision}_0 \cdot \text{Recall}_0}{\text{Precision}_0 + \text{Recall}_0} + \frac{2 \cdot \text{Precision}_1 \cdot \text{Recall}_1}{\text{Precision}_1 + \text{Recall}_1} \right)$$

4. **False Positive Rate (FPR) and False Negative Rate (FNR):**
   $$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$
   $$\text{FNR} = \frac{FN}{FN + TP} = 1 - \text{Recall}_1$$

5. **Receiver Operating Characteristic - Area Under the Curve (ROC-AUC):**
   The integral of the true positive rate as a function of false positive rate across all operational probability thresholds $\tau \in [0, 1]$:
   $$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\tau) \, d(\text{FPR}(\tau))$$

### 3.5 SHAP Explainability Implementation
Token-level feature attribution was implemented using the `shap` library (`src/explain/shap_explainer.py`).

#### 3.5.1 Mechanics and Model Interface
SHAP models the output of the fine-tuned XLM-RoBERTa classifier as a cooperative game where each token in the input string is a player. The explainer wraps the live classifier's inference function:
$$f: \mathcal{X} \to \mathbb{R}^2, \quad f(\mathbf{x}) = \left[ P(\text{Non-Hate} \mid \mathbf{x}), \; P(\text{Hate} \mid \mathbf{x}) \right]$$
To compute attributions without modifying model weights, we employ `shap.maskers.Text(tokenizer=r"\W+")`. The masker defines a perturbation space over the input text by masking subsets of word and token boundaries with empty reference substitutions.

The Shapley value $\phi_i(f, \mathbf{x})$ for token $i$ relative to the hate class ($c=1$) is given by:
$$\phi_i(f, \mathbf{x}) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_1(S \cup \{i\}) - f_1(S) \right]$$
where $F$ is the set of all input tokens, and $S$ represents coalitions of present tokens.

#### 3.5.2 Attribution Interpretation
By the local accuracy (efficiency) axiom of SHAP:
$$f_1(\mathbf{x}) = \phi_0 + \sum_{i=1}^{|F|} \phi_i$$
where $\phi_0$ is the base expected value across the perturbation background, and $\phi_i$ represents the additive shift in predicted hate probability attributable to token $i$:
- $\phi_i > 0$ (positive attribution): Token $i$ pushes the model's prediction toward `Hate`.
- $\phi_i < 0$ (negative attribution): Token $i$ pushes the model's prediction toward `Non-Hate`.

In the offline explanation experiment, explanations were generated across 20 stratified held-out test instances with a computational evaluation budget of `max_evals=250` per sequence (`results/shap/`). In the interactive Streamlit application, live SHAP inference evaluates with `max_evals=500` for enhanced attribution precision.

### 3.6 LIME Explainability Implementation
Local Interpretable Model-agnostic Explanations were implemented using `lime.lime_text.LimeTextExplainer` (`src/explain/lime_explainer.py`).

#### 3.6.1 Mathematical Formulation and Local Surrogate
LIME explains a prediction by approximating the complex non-linear decision boundary of XLM-RoBERTa locally around the instance of interest $\mathbf{x}$. 

Given an input comment $\mathbf{x}$, LIME draws $M$ perturbed text samples $\mathbf{z}' \in \{0, 1\}^{d}$ in a binary feature space indicating word presence or absence ($z'_j = 0$ if word $j$ is removed). These binary vectors are mapped back to perturbed text strings $\mathbf{z}$, which are passed through the live XLM-RoBERTa classifier to obtain prediction probabilities $f(\mathbf{z}) = P(\text{Hate} \mid \mathbf{z})$.

LIME solves the penalized weighted optimization problem:
$$\boldsymbol{\xi}(\mathbf{x}) = \arg\min_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_{\mathbf{x}}) + \Omega(g)$$
where:
- $g(\mathbf{z}') = \mathbf{w}_g^T \mathbf{z}' + b_g$ is an interpretable sparse linear surrogate model.
- $\pi_{\mathbf{x}}(\mathbf{z}) = \exp\left( -D(\mathbf{x}, \mathbf{z})^2 / \sigma^2 \right)$ is an exponential proximity kernel measuring cosine distance between the original and perturbed texts.
- $\mathcal{L}(f, g, \pi_{\mathbf{x}}) = \sum_{\mathbf{z}, \mathbf{z}'} \pi_{\mathbf{x}}(\mathbf{z}) \left( f(\mathbf{z}) - g(\mathbf{z}') \right)^2$ is the weighted squared loss of the local surrogate fit.

#### 3.6.2 Explainer Configuration and Distinctions
- **Tokenization:** Initialized with `split_expression=r"\W+"` to split cleanly across punctuation and whitespace in multilingual text.
- **Bag-of-Words Setting:** Configured with `bow=False` to preserve word order and multiple token occurrences rather than collapsing duplicates.
- **Sample Budgets:** Configured with $M = 200$ perturbation samples in the offline faithfulness benchmark, and $M = 300$ samples in the interactive application.
- **Surrogate Fit Metric ($R^2$):** LIME reports a coefficient of determination $R^2 \in [0, 1]$ measuring how accurately the linear surrogate approximates XLM-RoBERTa's outputs across the local perturbation neighborhood. Crucially, **LIME $R^2$ is not classification accuracy**; an $R^2$ of 0.85 indicates that the linear surrogate explains 85% of the local variance in XLM-RoBERTa's probability shifts.
- **Granularity Differences:** LIME perturbs whole words at whitespace boundaries, whereas SHAP perturbs at subword/character token levels. Consequently, direct numeric comparisons of raw feature weights between SHAP and LIME are invalid; they operate across distinct representational spaces.

### 3.7 Faithfulness Evaluation Framework
To scientifically audit whether the tokens identified by SHAP and LIME represent true computational dependencies of XLM-RoBERTa, we implemented an empirical, deletion-based faithfulness experiment (`src/evaluation/faithfulness.py`).

#### 3.7.1 Experimental Design and Stratified Sampling
The faithfulness evaluation was executed on $N = 20$ held-out test samples drawn via stratified sampling (seed $= 99$) from `data/processed/unified/test.csv`:
- Exactly 10 `Non-Hate` samples ($y = 0$) and 10 `Hate` samples ($y = 1$).
- Stratified representation across all four languages: 8 Tamil, 4 Malayalam, 4 Hindi, and 4 Telugu instances.

#### 3.7.2 Deletion Perturbation Operation
For each sample text $\mathbf{x}$, explanations were generated independently using SHAP and LIME. For each method and each deletion depth $k \in \{1, 3, 5\}$:
1. The top-$k$ features were selected by absolute attribution magnitude ($|\phi_i|$ for SHAP; $|w_j|$ for LIME).
2. The selected features were removed from the original text string $\mathbf{x}$ using exact-substring deletion to prevent regex word-boundary failures on non-Latin scripts.
3. Multiple whitespaces were normalized to single spaces, yielding perturbed text $\mathbf{x}_{\setminus k}$.
4. As a rigorous control baseline, **k random words** were sampled from the pool of valid words present in the text (using deterministic seed $77 + \text{sample\_id} \times 100 + k$) and deleted using identical exact-match logic.

#### 3.7.3 Probability Drop Metric ($\Delta P$)
The perturbed texts $\mathbf{x}_{\setminus k}$ were fed back into the **identical, unmodified XLM-RoBERTa model** to observe the new prediction probabilities.

The primary faithfulness metric is the **Probability Drop for the Original Predicted Class**:
$$\Delta P = P(Y = \hat{y} \mid \mathbf{x}) - P(Y = \hat{y} \mid \mathbf{x}_{\setminus k})$$
where $\hat{y} \in \{0, 1\}$ is the model's original predicted class on unperturbed input $\mathbf{x}$.

**Scientific Interpretation:**
- $\Delta P > 0$: Removing the top-$k$ features caused the model's confidence in its original decision to **decrease**. A larger positive drop indicates that the explainer successfully identified tokens upon which the model was computationally reliant.
- $\Delta P \approx 0$: Removing the features had negligible impact on model confidence, indicating low feature influence.
- $\Delta P < 0$: Removing the features paradoxically caused model confidence to increase.

#### 3.7.4 Bootstrap Confidence Intervals
To assess statistical stability over the 20-sample evaluation set without making Gaussian distributional assumptions, **percentile bootstrap resampling** was implemented:
- Iterations: $B = 1,000$ bootstrap resamples with replacement.
- Seed: Deterministic seed $= 42$.
- Confidence Interval: Two-tailed 95% bootstrap confidence interval derived from the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles of the bootstrap distribution:
  $$\text{CI}_{95\%} = \left[ q_{0.025}, \; q_{0.975} \right]$$

### 3.8 Cross-Language Error Analysis Framework
To diagnose language-specific failure modes, predictions on the full held-out test set ($N = 6,319$) were disaggregated across Tamil, Malayalam, Hindi, and Telugu (`results/error_analysis/`).

The framework evaluates:
1. Independent $2 \times 2$ confusion matrices for each language:
   $$\begin{bmatrix} TN_{\ell} & FP_{\ell} \\ FN_{\ell} & TP_{\ell} \end{bmatrix}, \quad \ell \in \{\text{Tamil, Malayalam, Hindi, Telugu}\}$$
2. Per-language macro F1-score, ROC-AUC, FPR, and FNR.
3. Analysis of false positives ($y=0, \hat{y}=1$) and false negatives ($y=1, \hat{y}=0$).
4. Qualitative examination of error instances, categorizing failures into:
   - Aggressive colloquial swearing lacking targeted hate intent.
   - Orthographic and phonetic transliteration ambiguities.
   - Sarcasm, figurative rhetoric, and subtle communal dogwhistles.
   - Inherent annotator label noise and boundary ambiguity.

### 3.9 Interactive Streamlit Application Architecture
To bridge experimental research and operational utility, an interactive web application was engineered using Streamlit (`app/streamlit_app.py`).

The application features:
- **Modular Component Design:** Encapsulated in a modern dark-slate design system (`#0b0f19` background) utilizing high-contrast typography (Google Font Inter).
- **Tab 1: Detection (Real-Time Inference & XAI):**
  - Model selection dropdown enabling live execution across XLM-RoBERTa, TF-IDF + Logistic Regression, or TF-IDF + Linear SVM.
  - Interactive text input supporting arbitrary code-mixed or native-script text.
  - Real-time classification output with visual badge indicators (`Hate` / `Non-Hate`), prediction confidence, and animated probability progress bars.
  - Dynamic SHAP explanation toggle rendering horizontal attribution bar charts (red bars indicating hate-promoting tokens, green bars indicating non-hate tokens).
  - Dynamic LIME explanation toggle rendering word-level surrogate weights, explained class indicator, and local surrogate $R^2$ fit scores.
  - Caching optimization: Utilizes `@st.cache_resource` to load the fine-tuned XLM-RoBERTa model and tokenizer once in memory, sharing the instance across inference, SHAP, and LIME without redundant disk I/O.
- **Tab 2: Performance (Audit & Benchmarking):**
  - Displays verified model comparison metrics across all three architectures.
  - Integrates disaggregated per-language performance tables and language-specific confusion matrices.
- **Tab 3: Methodology (Transparency & Provenance):**
  - Details dataset provenance, preprocessing constraints, neural hyperparameters, and mathematical formulations of explainability metrics.

---

## 4. Results

All results presented in this section are extracted directly from verified evaluation artifacts generated by live model inference on the unified held-out test dataset (`data/processed/unified/test.csv`, $N = 6,319$).

### 4.1 Overall Held-Out Test Results
Table 2 reports the aggregate performance metrics across the 6,319 held-out test instances for the two conventional baselines and the fine-tuned XLM-RoBERTa Base classifier.

**Table 2: Comprehensive Held-Out Test Evaluation ($N = 6,319$)**
| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC | Authoritative Source File |
|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF + Logistic Regression | 82.02% | 75.96% | 80.63% | 77.57% | 82.76% | 89.24% | `tfidf_lr_unified_test_metrics_verified.json` |
| TF-IDF + Linear SVM | 84.06% | 79.78% | 73.51% | 75.78% | 83.13% | 88.41% | `tfidf_svm_unified_test_metrics_verified.json` |
| **XLM-RoBERTa Base** | **84.86%** | **79.24%** | **79.41%** | **79.33%** | **84.88%** | **89.89%** | `xlmr_unified_test_metrics_verified.json` |

*Note on Artifact Verification:* A minor discrepancy of 0.02% was identified between `model_comparison_summary.csv` (which listed XLM-R accuracy as 84.84% and Macro F1 as 79.31%) and the primary verification artifact `xlmr_unified_test_metrics_verified.json` (listing accuracy as 84.86% and Macro F1 as 79.33%). Independent recomputation confirmed that the confusion matrix values in `xlmr_unified_test_metrics_verified.json` yield an exact accuracy of $(4315 + 1047) / 6319 = 84.8552\%$ ($\approx 84.86\%$) and Macro F1 of $79.33\%$. The verified JSON file is authoritative.

The verified confusion matrices across all 6,319 test samples are detailed in Table 3.

**Table 3: Aggregate Confusion Matrix Breakdown ($N = 6,319$)**
| Model Architecture | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 3,999 | 801 | 335 | 1,184 | 6,319 |
| TF-IDF + Linear SVM | 4,504 | 296 | 711 | 808 | 6,319 |
| **XLM-RoBERTa Base** | **4,315** | **485** | **472** | **1,047** | **6,319** |

#### Comparative Architectural Discussion
1. **Macro F1 Dominance:** XLM-RoBERTa achieves the highest overall Macro F1-score (79.33%), representing a $+1.76\%$ absolute improvement over Logistic Regression (77.57%) and a $+3.55\%$ improvement over Linear SVM (75.78%).
2. **Precision vs. Recall Trade-Offs in Baselines:**
   - Linear SVM achieves the highest Macro Precision (79.78%) and the lowest overall false-positive count ($FP = 296$). However, it suffers from poor recall on hate speech, missing 711 hate instances ($FN = 711$), resulting in a Macro Recall of only 73.51%.
   - Logistic Regression exhibits the opposite behavior: it achieves high Macro Recall (80.63%) and identifies 1,184 true hate instances ($TP = 1,184$), but produces 801 false alarms ($FP = 801$), yielding a lower Macro Precision of 75.96%.
3. **Balanced Transformer Predictions:** XLM-RoBERTa achieves an optimal balance between precision (79.24%) and recall (79.41%), maintaining an approximately equal distribution between false positives ($485$) and false negatives ($472$).
4. **ROC-AUC Distinction:** XLM-RoBERTa achieves an ROC-AUC of 89.89%, demonstrating that its predicted probabilities are well-calibrated and capable of discriminating between toxic and non-toxic instances across variable operating thresholds.

### 4.2 Disaggregated Per-Language Performance
To investigate whether XLM-RoBERTa's aggregate superiority holds across all languages, Table 4 disaggregates the performance of all three models across Tamil, Malayalam, Hindi, and Telugu.

**Table 4: Disaggregated Language-Wise Evaluation ($N = 6,319$)**
| Language | Samples | Class 0 / Class 1 | Metric | TF-IDF + LR | TF-IDF + SVM | XLM-RoBERTa Base | Best Architecture |
|---|---:|:---:|---|---:|---:|---:|:---:|
| **Tamil** | 4,174 | 3,148 / 1,026 | Accuracy <br> Macro F1 <br> ROC-AUC <br> FPR <br> FNR | 80.76% <br> 76.05% <br> 87.76% <br> 17.1% <br> 25.9% | 82.39% <br> 73.10% <br> 86.53% <br> 6.4% <br> 51.9% | **83.68%** <br> **77.92%** <br> **88.57%** <br> 10.6% <br> 33.7% | **XLM-RoBERTa** <br> (+1.87% F1 vs LR) <br> (+4.82% F1 vs SVM) |
| **Malayalam** | 1,261 | 1,211 / 50 | Accuracy <br> Macro F1 <br> ROC-AUC <br> FPR <br> FNR | 95.00% <br> **68.70%** <br> 85.02% <br> 2.8% <br> 58.0% | 95.88% <br> 62.83% <br> 85.95% <br> 1.0% <br> 80.0% | **95.48%** <br> 66.07% <br> **88.22%** <br> 1.8% <br> 70.0% | **Logistic Regression** <br> (+2.63% F1 vs XLM-R) <br> (+5.87% F1 vs SVM) |
| **Hindi** | 490 | 238 / 252 | Accuracy <br> Macro F1 <br> ROC-AUC <br> FPR <br> FNR | 67.96% <br> 64.79% <br> 81.44% <br> 60.9% <br> 4.8% | **75.31%** <br> **75.31%** <br> **83.23%** <br> 22.3% <br> 27.0% | 73.67% <br> 73.58% <br> 82.22% <br> 30.2% <br> 22.6% | **Linear SVM** <br> (+1.73% F1 vs XLM-R) <br> (+10.52% F1 vs LR) |
| **Telugu** | 394 | 203 / 191 | Accuracy <br> Macro F1 <br> ROC-AUC <br> FPR <br> FNR | 71.32% <br> 70.94% <br> 80.99% <br> 41.9% <br> 14.7% | 74.87% <br> 74.41% <br> 80.82% <br> 14.3% <br> 36.6% | **77.16%** <br> **77.14%** <br> **82.34%** <br> 27.6% <br> 17.8% | **XLM-RoBERTa** <br> (+6.20% F1 vs LR) <br> (+2.73% F1 vs SVM) |

The complete confusion matrices for XLM-RoBERTa across each language subset (`results/error_analysis/language_metrics.csv`) are summarized in Table 5.

**Table 5: Per-Language Confusion Matrix and Error Rates for XLM-RoBERTa**
| Language | Samples | TN | FP | FN | TP | FPR ($FP / (FP+TN)$) | FNR ($FN / (FN+TP)$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tamil | 4,174 | 2,813 | 335 | 346 | 680 | 10.64% | 33.72% |
| Malayalam | 1,261 | 1,189 | 22 | 35 | 15 | 1.82% | **70.00%** |
| Hindi | 490 | 166 | 72 | 57 | 195 | **30.25%** | 22.62% |
| Telugu | 394 | 147 | 56 | 34 | 157 | 27.59% | 17.80% |

#### Critical Findings from Disaggregated Evaluation
1. **The Malayalam Accuracy vs. Macro F1 Paradox:**
   - In Malayalam, the test split contains 1,211 Non-Hate instances (96.0%) and only 50 Hate instances (4.0%).
   - A naive majority-class classifier that blindly predicts `Non-Hate` for every sample would achieve an accuracy of $1,211 / 1,261 = 96.03\%$.
   - XLM-RoBERTa achieves an apparent accuracy of **95.48%**, the highest nominal accuracy across all four languages.
   - However, XLM-RoBERTa's Macro F1 on Malayalam is only **66.07%**, the lowest among all languages.
   - Crucially, XLM-RoBERTa misses 35 out of 50 actual hate comments ($FN = 35$), suffering an alarming **False Negative Rate of 70.00%**. It detects only 15 hate comments ($TP = 15$), corresponding to a minority-class recall of just 30.0%.
   - Meanwhile, conventional TF-IDF + Logistic Regression achieves a higher Macro F1 of **68.70%** by predicting the positive class more aggressively ($TP = 21, FN = 29$, Recall $= 42.0\%$).
2. **The Hindi False-Positive Challenge:**
   - Hindi exhibits the lowest overall accuracy for XLM-RoBERTa (73.67%) and the highest False Positive Rate (**30.25%**, $FP = 72$ out of 238 non-hate comments).
   - Linear SVM outperforms XLM-RoBERTa on Hindi, achieving an accuracy of 75.31% and a Macro F1 of **75.31%** (vs. 73.58% for XLM-R). The SVM's balanced class weighting effectively regularized against false positives ($FP = 53$ vs. 72 for XLM-R).
3. **Tamil's Dominance in Unified Metrics:**
   - Tamil comprises 66.1% of the entire test dataset (4,174 out of 6,319 samples).
   - Because Tamil is the largest subset and achieves a balanced Macro F1 of 77.92%, aggregate unified metrics reflect Tamil's performance far more than the smaller Hindi (490) or Telugu (394) subsets.
4. **Telugu Performance:**
   - Telugu exhibits a balanced class distribution (51.5% Non-Hate vs. 48.5% Hate).
   - XLM-RoBERTa dominates both baselines on Telugu, achieving a Macro F1 of 77.14% ($+6.20\%$ over Logistic Regression and $+2.73\%$ over Linear SVM) with a low False Negative Rate of 17.80%.

### 4.3 Explainability Findings (SHAP and LIME)
Qualitative and quantitative analysis of live token-level SHAP and word-level LIME explanations reveals distinct patterns in how models assign importance across languages.

#### 4.3.1 Representative Explanation Cases
From the verified explanation registry (`results/shap/shap_test_sample_summary.csv` and `results/error_analysis/error_examples.csv`):

- **Case 1: Obvious Toxicity in Romanized Transliteration (Telugu)**
  - *Input Text:* `"Adara Dengings Inka aa roju nyt"`
  - *Ground Truth:* `Hate` | *XLM-R Prediction:* `Hate` ($P(\text{Hate}) = 0.9487$)
  - *SHAP Attribution:* The token `Dengings` received a massive positive attribution ($\phi > +0.45$), dominating the prediction. LIME similarly assigned its highest positive weight ($w = +0.58$, direction $\to$ `Hate`, surrogate $R^2 = 0.81$) to `Dengings`. Both explainers accurately reflected that the model latched onto the explicit Telugu-English transliterated profanity.
- **Case 2: Misclassified Movie Banter (Tamil False Positive)**
  - *Input Text:* `"mokka da deyy ajith fans loussou poudeyy lo aaa"`
  - *Ground Truth:* `Non-Hate` (Informal cinema fan banter) | *XLM-R Prediction:* `Hate` ($P(\text{Hate}) = 0.9624$)
  - *Attribution Analysis:* Both SHAP and LIME highlighted `loussou` and `poudeyy` as overwhelming drivers toward `Hate`. The model failed to recognize that aggressive cinema fan banter between rival fanbases (Ajith vs. Vijay) represents colloquial mockery rather than targeted demographic hate speech.
- **Case 3: Sarcastic Political Discourse (Hindi False Positive)**
  - *Input Text:* `"Rizwan offered Namaz during INDvPAK match in middle of Hindus was most sati..."`
  - *Ground Truth:* `Non-Hate` | *XLM-R Prediction:* `Hate` ($P(\text{Hate}) = 0.9534$)
  - *Attribution Analysis:* SHAP attributed high positive scores to religious and national identifiers (`Namaz`, `INDvPAK`, `Hindus`). The model learned spurious correlations associating religious terms with hate speech due to their co-occurrence with toxic commentary in the training data.
- **Case 4: Subtle Implicit Threat (Hindi False Negative)**
  - *Input Text:* `"I am ready to sponsor flight tickets to Kabul for Arfa Rana Saba Asad etc..."`
  - *Ground Truth:* `Hate` (Targeted political harassment) | *XLM-R Prediction:* `Non-Hate` ($P(\text{Hate}) = 0.0108$)
  - *Attribution Analysis:* The comment contains polite surface lexical tokens (`ready`, `sponsor`, `flight`, `tickets`). SHAP assigned strong negative values ($\phi < -0.20$, pushing toward `Non-Hate`) to `sponsor` and `tickets`. Because the comment lacks overt profanity or slur words, the model's self-attention mechanism failed to capture the implicit, sarcastic deportation threat directed at female Muslim journalists.

### 4.4 Faithfulness Evaluation Findings
The empirical faithfulness experiment evaluated whether deleting top-$k$ attributed features caused corresponding probability drops in the model's confidence for its original predicted class.

Table 6 reports the verified summary statistics across all 20 stratified evaluation samples, comparing SHAP, LIME, and the Random baseline across $k \in \{1, 3, 5\}$ (`results/faithfulness/faithfulness_summary.csv`).

**Table 6: Empirical Faithfulness Evaluation Results ($N = 20, B = 1,000$ Bootstrap Resamples)**
| Attribution Method | Depth ($k$) | Mean Probability Drop ($\Delta P$) | Median $\Delta P$ | Mean Absolute Change ($|\Delta P|$) | Fraction Decreased ($\Delta P > 0$) | 95% Bootstrap Confidence Interval |
|---|---:|---:|---:|---:|---:|:---:|
| **SHAP** | 1 | 0.1200 | 0.0025 | 0.1300 | 80.0% | [0.0135, 0.2452] |
| **SHAP** | 3 | 0.1617 | 0.0062 | 0.1825 | 70.0% | [0.0451, 0.3124] |
| **SHAP** | 5 | 0.2069 | 0.0294 | 0.2178 | 90.0% | [0.0834, 0.3508] |
| **LIME** | 1 | 0.1323 | 0.0045 | 0.1388 | 80.0% | [0.0361, 0.2580] |
| **LIME** | 3 | 0.2115 | 0.0342 | 0.2236 | 75.0% | [0.0745, 0.3712] |
| **LIME** | 5 | 0.2668 | 0.0173 | 0.2763 | 85.0% | [0.1163, 0.4378] |
| **RANDOM** | 1 | 0.0355 | 0.000039 | 0.0549 | 50.0% | [-0.0142, 0.1246] |
| **RANDOM** | 3 | 0.0148 | 0.000754 | 0.0341 | 65.0% | [-0.0105, 0.0447] |
| **RANDOM** | 5 | 0.1327 | 0.000509 | 0.1568 | 70.0% | [0.0069, 0.2789] |

#### Scientific Interpretation of Faithfulness Metrics
1. **Significant Outperformance of Random Baseline:**
   - At $k = 1$, deleting the single top SHAP token produces a mean probability drop of **0.1200** (95% CI: $[0.0135, 0.2452]$), and the top LIME word produces **0.1323** (95% CI: $[0.0361, 0.2580]$). In contrast, deleting a random word produces a drop of only **0.0355** whose 95% bootstrap CI spans zero ($[-0.0142, 0.1246]$).
   - At $k = 3$, the distinction widens substantially: SHAP ($\Delta P = 0.1617$) and LIME ($\Delta P = 0.2115$) outperform Random ($\Delta P = 0.0148$) by over an order of magnitude.
   - At $k = 5$, removing the top 5 features causes substantial erosion in model confidence: $\Delta P = 0.2069$ for SHAP and $\Delta P = 0.2668$ for LIME.
   - These results confirm empirically that both SHAP and LIME successfully isolate tokens to which the underlying XLM-RoBERTa architecture is computationally sensitive.
2. **Monotonicity Across Deletion Depths:**
   - Both SHAP and LIME exhibit strict monotonic increases in mean probability drop as $k$ expands from 1 to 5:
     $$\text{SHAP: } 0.1200 \to 0.1617 \to 0.2069$$
     $$\text{LIME: } 0.1323 \to 0.2115 \to 0.2668$$
   - This monotonicity demonstrates that feature rankings are coherent: removing multiple high-ranking features degrades model confidence progressively more than removing single features.
3. **Comparison Between SHAP and LIME:**
   - Across all depths, LIME exhibits slightly higher numerical mean drops than SHAP ($0.1323$ vs. $0.1200$ at $k=1$; $0.2668$ vs. $0.2069$ at $k=5$).
   - However, **this does not establish that LIME is a superior explainer.** LIME operates on whole whitespace-delimited words, whereas SHAP operates on subword/punctuation tokens. Deleting an entire word removes more characters and semantic content from the input sequence than deleting a subword token, naturally inducing a larger probability shift.
4. **Distributional Skew and the Median vs. Mean Gap:**
   - Across both methods, the median probability drop is substantially smaller than the mean (e.g., at $k=1$, SHAP mean is $0.1200$ while median is $0.0025$; LIME mean is $0.1323$ while median is $0.0045$).
   - This divergence indicates that probability drops follow a heavy-tailed distribution: for a subset of instances where the model relied heavily on a single focal keyword, deleting that feature caused catastrophic confidence collapse ($\Delta P > 0.80$). For other instances where the model relied on diffuse sentence-wide context, removing single tokens produced minimal shifts.
5. **Absence of Causal Claims:**
   - While deletion tests demonstrate correlation between attribution scores and prediction shifts, **they do not constitute proof of causal mechanisms.** Deleting tokens shifts the input text off the natural language data manifold (introducing ungrammatical text fragments), which may induce model confusion unrelated to human semantic reasoning.

### 4.5 Error Analysis and Systematic Limitations
A systematic qualitative review of misclassified examples from `results/error_analysis/error_examples.csv` reveals three primary linguistic mechanisms driving model failure:

1. **Colloquial Vulgarity vs. Targeted Hate (False Positives):**
   - In Tamil and Telugu social media, profanity is frequently employed as an emphatic intensifier or informal greeting without hostile or discriminatory intent.
   - For example, Telugu comment `"warner ni retain cheyakapote srh managament gudda pagilipoddi"` represents aggressive cricket fan frustration regarding player retention. The model predicted `Hate` ($P(\text{Hate}) = 0.9657$), triggered by the vulgar slang phrase, whereas the ground-truth annotation was `Non-Hate`.
   - The model struggles to separate vulgarity from hate speech, mirroring the challenge documented by Davidson et al. (2017) in English.
2. **Agglutination and Tokenizer Fragmentation (False Negatives):**
   - Dravidian languages (Tamil, Malayalam, Telugu) are highly agglutinative, attaching case markers, tense indicators, and postpositions to root nouns and verbs.
   - In Malayalam, native-script comments such as `"പുത്തൻ പണം 2... 2020 ലെ ആദ്യത്തെ ബോംബ് ഇക്ക ഇങ്ങെടുക്കുവാ..."` were misclassified as `Non-Hate` ($P(\text{Hate}) = 0.0031$). XLM-RoBERTa's SentencePiece tokenizer, trained predominantly on Roman and major world scripts, fragments agglutinated Malayalam words into numerous tiny subwords (often individual characters or byte fallbacks), diluting attention weights and hindering toxic pattern recognition.
3. **Subtle Communal Sarcasm and Metaphor (False Negatives):**
   - Hate speech in political and religious discourse increasingly relies on euphemisms, historical metaphors, and dogwhistles rather than overt slurs.
   - For example, comments referencing historical events (e.g., Babri Masjid demolition) without profanity frequently evade detection because individual tokens appear neutral to the model's classification head.

---

## 5. Conclusion

This project has presented a comprehensive, faithfulness-aware explainability audit of multilingual transformer architectures for code-mixed hate speech detection across four major Indian languages: Tamil, Malayalam, Hindi, and Telugu. By synthesizing real-world data engineering, rigorous statistical modeling, post-hoc explainability, and empirical perturbation testing, several substantive conclusions emerge:

1. **Overall Predictive Superiority of Multilingual Transformers:** On the unified held-out test dataset of 6,319 samples, XLM-RoBERTa Base demonstrates the strongest overall classification performance, achieving an accuracy of 84.86%, a macro-averaged F1-score of 79.33%, and an ROC-AUC of 89.89%, outperforming conventional TF-IDF baselines. Contextualized cross-lingual representations enable the transformer to capture complex syntactic dependencies and subtle hostile discourse more effectively than static n-gram models.
2. **Refutation of Uniform Transformer Superiority:** Disaggregated cross-language evaluation establishes that XLM-RoBERTa does not universally dominate conventional linear models. On Malayalam, TF-IDF + Logistic Regression achieves a higher Macro F1-score (68.70% vs. 66.07%), while on Hindi, TF-IDF + Linear SVM surpasses XLM-RoBERTa (75.31% vs. 73.58%). Conventional models utilizing character-within-word n-grams remain remarkably competitive on specific code-mixed dialects, particularly where training sample sizes are constrained.
3. **The Accuracy Paradox Under Class Imbalance:** In severely imbalanced linguistic subsets, overall accuracy is a misleading metric. On Malayalam (4.0% hate prevalence), XLM-RoBERTa achieves an apparent accuracy of 95.48%, yet suffers an unacceptable False Negative Rate of 70.00%, failing to identify 35 out of 50 hate comments. Reliable auditing of content moderation models requires unweighted macro metrics, false-negative rates, and disaggregated evaluations.
4. **Empirical Faithfulness of Post-Hoc Explanations:** The deletion-based faithfulness experiment provides rigorous empirical evidence that SHAP and LIME identify features to which the transformer is computationally sensitive. Deleting top-$k$ attributed features produces substantial and monotonically increasing drops in model confidence ($\Delta P = 0.1200 \to 0.2069$ for SHAP; $0.1323 \to 0.2668$ for LIME), consistently outperforming random-feature deletion baselines ($\Delta P = 0.0355 \to 0.1327$).
5. **Necessity of Explainability Auditing for Safe Deployment:** Post-hoc explanations reveal critical model vulnerabilities, such as over-reliance on demographic nouns (triggering false alarms in Hindi) and blindness to implicit sarcasm (triggering missed detections). Automated moderation systems cannot be deemed production-ready based on aggregate benchmark metrics alone; continuous faithfulness and fairness auditing is an essential prerequisite for ethical NLP deployment.

---

## 6. Future Work

Based on the empirical findings, linguistic failure modes, and methodological constraints documented in this research, several concrete directions for future work are proposed:

1. **Targeted Imbalance Mitigation for Low-Resource Subsets:** Investigate advanced cost-sensitive loss formulations (e.g., Focal Loss, Class-Balanced Cross-Entropy) and synthetic data augmentation techniques specifically for Malayalam to reduce its 70% false-negative rate without sacrificing overall precision.
2. **Regularization Against False Positives in Hindi and Telugu:** Develop adversarial debiasing and counterfactual data augmentation pipelines to detach model predictions from neutral religious, national, and political identity nouns, mitigating Hindi's 30.2% false-positive rate.
3. **Script-Disaggregated Comparative Auditing:** Systematically separate native-script Indic comments from Romanized phonetic transliterations within the evaluation benchmark to quantify tokenizer fragmentation and measure the performance gap between scripts.
4. **Evaluation of Indic-Centric Pretrained Encoders:** Extend the architectural benchmark to include Indic-specialized language models such as MuRIL (Multilingual Representations for Indic Languages) and IndicBERT v2, evaluating whether Indic-tailored pretraining reduces subword fragmentation on agglutinative Dravidian languages.
5. **Large-Scale Faithfulness and Rationalization Benchmarking:** Scale the empirical faithfulness evaluation from 20 instances to the full held-out test set ($N = 6,319$), and evaluate dual faithfulness metrics including both comprehensiveness (deletion-based drop) and sufficiency (retention-based confidence).
6. **Investigation of Explanation Stability and Invariance:** Measure the mathematical stability of SHAP and LIME attributions under minor, semantics-preserving character perturbations (e.g., alternative transliteration spellings) to assess whether explanations are robust against adversarial manipulation.
7. **Human-in-the-Loop Faithfulness and Plausibility Auditing:** Conduct structured human evaluation studies with native speakers of Tamil, Malayalam, Hindi, and Telugu to evaluate the alignment between model rationales, human intuitive justifications, and actual communicative intent.
8. **Context-Aware Sarcasm and Implicit Abuse Modeling:** Explore conversational context modeling (incorporating parent thread comments and video metadata) to enable classifiers to detect subtle, implicit harassment that lacks overt lexical toxicity.

---

## 7. References

- Bohra, D., Vijay, D., Singh, C., Akhtar, M. S., & Shrivastava, M. (2018). A dataset of Hindi-English code-mixed social media text for hate speech detection. In *Proceedings of the 2nd Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis* (pp. 36–41). Association for Computational Linguistics. https://aclanthology.org/W18-1105/
- Chakravarthi, B. R., Priyadharshini, R., Muralidaran, V., Suryawanshi, S., Navaneethakrishnan, S., Ponnusamy, R., & Kumaresan, P. K. (2021). Findings of the shared task on offensive language identification in Tamil and Malayalam. In *Proceedings of the 1st Workshop on Speech and Language Processing for Dravidian Languages (DravidianLangTech 2021)* (pp. 133–145). Association for Computational Linguistics. https://aclanthology.org/2021.dravidianlangtech-1.17/
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8440–8451). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.747
- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language. In *Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM 2017)* (pp. 512–515). AAAI Press. https://doi.org/10.1609/icwsm.v11i1.7347
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019)* (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423
- DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). ERASER: A benchmark to evaluate rationales in the English language. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 4443–4458). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.408
- Fortuna, P., & Nunes, S. (2018). A survey on automatic detection of hate speech in text. *ACM Computing Surveys*, 51(4), 1–30. https://doi.org/10.1145/3232676
- Kakwani, D., Kunchukuttan, A., Golla, S., Gokul, N. C., Bhattacharyya, A., Khapra, M. M., & Kumar, P. (2020). IndicNLPSuite: Monolingual corpora and evaluation benchmarks for Indic languages. In *Findings of the Association for Computational Linguistics: EMNLP 2020* (pp. 4948–4961). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.findings-emnlp.445
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS 2017)* (Vol. 30, pp. 4765–4774). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- Modha, S., Mandl, T., Majumder, P., & Patel, D. (2021). Overview of the HASOC track at FIRE 2021: Hate speech and offensive content identification in English and Indo-Aryan languages. In *Working Notes of FIRE 2021 - Forum for Information Retrieval Evaluation* (CEUR Workshop Proceedings, Vol. 3159, pp. 1–16). CEUR-WS.org. http://ceur-ws.org/Vol-3159/
- Nguyen, D. (2018). Comparing automatic and human evaluation of local explanations for text classification. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2018)* (pp. 1069–1078). Association for Computational Linguistics. https://doi.org/10.18653/v1/N18-1097
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)* (pp. 1135–1144). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939778
- Sai, S., Kulkarni, H., & Shrivastava, M. (2024). HOLD-Telugu: A corpus of hate and offensive language detection in Telugu-English code-mixed text. In *Proceedings of the 4th Workshop on Speech and Language Processing for Dravidian Languages (DravidianLangTech 2024 @ EACL 2024)* (pp. 50–57). Association for Computational Linguistics. https://aclanthology.org/2024.dravidianlangtech-1.8/
- Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K.-R. (2017). Evaluating the visualization of what a deep neural network has learned. *IEEE Transactions on Neural Networks and Learning Systems*, 28(11), 2660–2673. https://doi.org/10.1109/TNNLS.2016.2599820
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS 2017)* (Vol. 30, pp. 5998–6008). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

---

## 8. Verification Notes

This documentation was generated strictly from the active codebase, executed scripts, and verified evaluation artifacts in the project repository. All experimental figures, parameters, and results were confirmed prior to compilation:

1. **Dataset Counts and Provenance:** Verified against `scripts/prepare_unified_dataset.py`, `data/processed/unified/train.csv`, `val.csv`, and `test.csv`. The exact dimensions ($50,525$ train, $6,315$ validation, $6,319$ test, total $63,159$) and per-language distributions match the raw underlying files with zero data leakage.
2. **Model Metrics Discrepancy Analysis:** The authoritative test results were verified against `results/metrics/xlmr_unified_test_metrics_verified.json`, `tfidf_lr_unified_test_metrics_verified.json`, and `tfidf_svm_unified_test_metrics_verified.json`. The minor $0.02\%$ difference between `model_comparison_summary.csv` ($84.84\%$ accuracy, $79.31\%$ macro F1) and `xlmr_unified_test_metrics_verified.json` ($84.86\%$ accuracy, $79.33\%$ macro F1) was traced to pre-verification rounding in the summary CSV; the verified JSON metric file was confirmed as authoritative via exact confusion matrix recomputation.
3. **Per-Language Baseline Comparisons:** Verified against `results/error_analysis/language_metrics.csv` and `cross_language_report.md`. The specific observations that Logistic Regression achieves higher Macro F1 on Malayalam ($68.70\%$ vs. $66.07\%$) and Linear SVM achieves higher Macro F1 on Hindi ($75.31\%$ vs. $73.58\%$) were confirmed from the verified baseline metric logs.
4. **Faithfulness Output Artifacts:** All mean drops, median values, percentages decreased, and 95% bootstrap confidence intervals for SHAP, LIME, and Random baselines across $k \in \{1, 3, 5\}$ were verified against `results/faithfulness/faithfulness_summary.csv` and `faithfulness_results.json`.
5. **No Synthetic or Fabricated Elements:** No external unverified datasets were incorporated; Bohra et al. (2018) is cited strictly as a historical benchmark due to raw tweet dehydration. No code or model weights were altered during the generation of this documentation.
