# Phase 3F — Cross-Language Error Analysis Report

**Evaluation Split:** Unified Test Set (`data/processed/unified/test.csv`, 6,319 samples)
**Model:** XLM-RoBERTa (`models/xlmr/UNIFIED_XLMR_best`)
**Prediction Source:** `results/predictions/xlmr_unified_test_predictions.csv`

## 1. Overall Performance

- **Total Samples:** 6,319
- **Accuracy:** 0.8486
- **Macro Precision:** 0.7924
- **Macro Recall:** 0.7941
- **Macro F1:** 0.7933
- **ROC-AUC:** 0.8989
- **Confusion Matrix:** TN=4315, FP=485, FN=472, TP=1047

## 2. Per-Language Metrics

| Language | Samples | Non-Hate | Hate | Hate % | Accuracy | Macro Prec | Macro Rec | Macro F1 | ROC-AUC | TN | FP | FN | TP | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tamil** | 4174 | 3148 | 1026 | 24.6% | 0.8368 | 0.7802 | 0.7782 | **0.7792** | 0.8857 | 2813 | 335 | 346 | 680 | 0.1064 | 0.3372 |
| **Malayalam** | 1261 | 1211 | 50 | 4.0% | 0.9548 | 0.6884 | 0.6409 | **0.6607** | 0.8822 | 1189 | 22 | 35 | 15 | 0.0182 | 0.7000 |
| **Hindi** | 490 | 238 | 252 | 51.4% | 0.7367 | 0.7374 | 0.7356 | **0.7358** | 0.8222 | 166 | 72 | 57 | 195 | 0.3025 | 0.2262 |
| **Telugu** | 394 | 203 | 191 | 48.5% | 0.7716 | 0.7746 | 0.7731 | **0.7714** | 0.8234 | 147 | 56 | 34 | 157 | 0.2759 | 0.1780 |

## 3. Performance Extremes

- **Highest Accuracy:** Malayalam (0.9548)
- **Lowest Accuracy:** Hindi (0.7367)
- **Highest Macro F1:** Tamil (0.7792)
- **Lowest Macro F1:** Malayalam (0.6607)
- **Highest False-Negative Rate (Missed Hate):** Malayalam (70.0%, FN=35/50)
- **Highest False-Positive Rate (False Alarms):** Hindi (30.2%, FP=72/238)

## 4. Class Imbalance Analysis & Accuracy vs. Macro F1 Paradox

A critical scientific finding emerges when comparing **Malayalam** to other languages:

- **Malayalam exhibits severe class imbalance:** 1,211 Non-Hate (96.0%) vs. 50 Hate (4.0%).
- Because 96% of samples are Non-Hate, a trivial majority classifier predicting Non-Hate for everything would achieve 96.0% accuracy.
- XLM-R achieves **95.48% accuracy** on Malayalam, the highest among all languages.
- However, XLM-R's **Macro F1 on Malayalam is only 0.6607**, the lowest among all languages.
- Out of 50 actual Hate comments in Malayalam, the model misses 35 (FN = 35, FNR = 70.0%), detecting only 15 (TP = 15). Recall on Hate is only 30.0%.
- **Conclusion:** Accuracy is severely distorted by label distribution. Macro F1 provides an unweighted, balanced assessment of minority-class recognition.

## 5. Baseline Comparison (XLM-R vs. TF-IDF + LR vs. TF-IDF + SVM)

| Language | XLM-R Acc | LR Acc | SVM Acc | XLM-R F1 | LR F1 | SVM F1 | F1 Advantage vs LR | F1 Advantage vs SVM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tamil** | 0.8368 | 0.8076 | 0.8239 | **0.7792** | 0.7605 | 0.7310 | +0.0187 | +0.0482 |
| **Malayalam** | 0.9548 | 0.9500 | 0.9588 | **0.6607** | 0.6870 | 0.6283 | -0.0263 | +0.0324 |
| **Hindi** | 0.7367 | 0.6796 | 0.7531 | **0.7358** | 0.6479 | 0.7531 | +0.0879 | -0.0173 |
| **Telugu** | 0.7716 | 0.7132 | 0.7487 | **0.7714** | 0.7094 | 0.7441 | +0.0620 | +0.0273 |

## 6. Representative Error Examples

Actual unmodified examples from `xlmr_unified_test_predictions.csv`:

| Language | Error Type | True Label | Pred Label | P(Hate) | Text Excerpt |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Tamil | False Positive | Non-Hate | Hate | 0.9624 | `mokka da deyy ajith fans loussou poudeyy lo aaa` |
| Tamil | False Positive | Non-Hate | Hate | 0.9562 | `Mokka teaser.. ‼ ... Ithuku tha IVLO build up ah.. ‼` |
| Tamil | False Negative | Hate | Non-Hate | 0.0012 | `புதுச்சேரி வன்னியர்கள் சார்பாக இத்திரைப்படம் வெற்றிபெற வாழ்த்துக்கள்..........` |
| Tamil | False Negative | Hate | Non-Hate | 0.0015 | `வேற லெவல்ன்னு ஒரு கூட்டம் சொல்வாங்க பாரு` |
| Malayalam | False Positive | Non-Hate | Hate | 0.9517 | `flop kaluda rajakumaran ee padam hit agum pisharadikku` |
| Malayalam | False Positive | Non-Hate | Hate | 0.9414 | `Dae Ajithaaa copy adikathada Mohan Lal` |
| Malayalam | False Negative | Hate | Non-Hate | 0.0016 | `Rajappanod pokan para....njangalkku njangade stephen chettan undallo` |
| Malayalam | False Negative | Hate | Non-Hate | 0.0031 | `പുത്തൻ പണം 2...  2020 ലെ ആദ്യത്തെ ബോംബ് ഇക്ക ഇങ്ങെടുക്കുവാ 🤣🤣 എജ്ജാതി തീട്ട...` |
| Hindi | False Positive | Non-Hate | Hate | 0.9632 | `No nation can claim itself to be safe if the security of its own PM gets co...` |
| Hindi | False Positive | Non-Hate | Hate | 0.9534 | `Rizwan offered Namaz during INDvPAK match in middle of Hindus was most sati...` |
| Hindi | False Negative | Hate | Non-Hate | 0.0108 | `I am ready to sponsor flight tickets to Kabul for Arfa Rana Saba Asad etc. ...` |
| Hindi | False Negative | Hate | Non-Hate | 0.0224 | `Its game over folks. The Alfa-Trump pings were faked spoofed they are witho...` |
| Telugu | False Positive | Non-Hate | Hate | 0.9657 | `warner ni retain cheyakapote srh managament gudda pagilipoddi` |
| Telugu | False Positive | Non-Hate | Hate | 0.9651 | `Maa jaganu attu kudda pikalarru miru motamu kalise vachinnaaalion is jaganu...` |
| Telugu | False Negative | Hate | Non-Hate | 0.0081 | `తెలుగు బాయ్ ఫ్రెండ్ కావాలీ` |
| Telugu | False Negative | Hate | Non-Hate | 0.0107 | `నిత్యానందస్వామి ఆశీర్వాదం అందుకున్నా జబర్దస్ట్ రాణి` |

## 7. Scientific Observations vs. Hypotheses

- **OBSERVATION 1:** Tamil constitutes 66.1% (4,174 / 6,319) of the test dataset and achieves balanced Macro F1 (0.7792), driving most of the unified metric.
- **OBSERVATION 2:** Malayalam has the lowest Macro F1 (0.6607) due to high false-negative rate (70.0% of hate comments missed).
- **OBSERVATION 3:** Hindi has the lowest accuracy (0.7367) and highest false-positive rate (30.25%, FP=72/238).
- **HYPOTHESIS A:** Hindi's high false-positive rate may stem from intense code-mixed political discourse containing aggressive rhetoric or swear words used colloquially without genuine hate intent.
- **HYPOTHESIS B:** Malayalam's high false-negative rate may be influenced by subword tokenization fragmentation in SentencePiece or subtle sarcasm/implicit hate requiring deeper contextual reasoning.
