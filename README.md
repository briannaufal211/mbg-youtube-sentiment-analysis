# MBG YouTube Sentiment Analysis & Text Mining

End-to-end **Text Mining + NLP + Machine Learning + Deep Learning** project for analyzing 12,000 YouTube comments related to the MBG discussion.

## Project Objective

The project turns unstructured YouTube comments into measurable outputs through:

**Data quality → duplicate-leakage control → label validation workflow → text preprocessing → exploratory text mining → binary classification → model benchmarking → imbalance-aware evaluation → Optuna tuning → error analysis → explainability → 3-class classification**

The notebook is intentionally designed to distinguish descriptive findings from model-evaluation evidence and dataset limitations.

## Dataset

Primary portfolio file:

`data/mbg_comments_labeled.csv`

Current dataset snapshot in the repository contains 12,000 labeled comments. The notebook computes the exact post-deduplication modeling counts at runtime instead of hard-coding them.

### Labeling

The current portfolio dataset uses **AI-assisted semantic labeling** for:
- `positive`
- `negative`
- `neutral`

The repository also contains `Scripts/02_prepare_labeling.py`, which prepares a sample for manual annotation.

The notebook now creates:

`results/human_validation_sample_for_annotation.csv`

This is a **blinded** validation file: the AI label is not exposed to the human annotator. Until `human_label` is actually filled, the notebook reports validation status as **PENDING** and does not fabricate agreement metrics.

## Data Quality & Leakage Controls

Before ML splitting, the notebook:

1. normalizes raw text into a deterministic duplicate key;
2. removes empty/invalid texts from the ML corpus;
3. detects duplicate groups with conflicting labels;
4. excludes conflicting duplicate groups rather than choosing an arbitrary label;
5. deduplicates repeated text **before** train/validation/test splitting;
6. asserts there is no duplicate-key overlap between splits.

The raw dataset remains available for descriptive EDA, while the model corpus is protected against repeated-text leakage.

## NLP Preprocessing

The notebook handles Indonesian YouTube-style text through:

- HTML/entity normalization
- URL and mention removal
- hashtag normalization
- Unicode normalization
- lowercasing
- repeated-character normalization
- slang normalization
- Indonesian stopword removal
- explicit preservation of negation words
- optional Sastrawi stemming

## Modeling

### Binary task

**Positive vs Negative** is treated as a focused polarity-classification experiment. Neutral is kept for dataset-level analysis and a separate 3-class benchmark.

Models:

1. **TF-IDF + Logistic Regression** — interpretable classical baseline
2. **BiLSTM** — deep-learning baseline
3. **BiGRU + Optuna** — tuned deep-learning candidate

### Class imbalance

The binary task is expected to be strongly imbalanced, so the notebook reports:

- Accuracy
- Balanced Accuracy
- Positive-class Precision / Recall / F1
- Macro F1
- Weighted F1
- ROC-AUC
- PR-AUC / Average Precision
- confusion matrix
- error analysis

Class weights are learned from the training set only.

### Optuna

Optuna's objective is explicitly:

**maximize validation Macro F1**

The test set is never used during hyperparameter tuning.

Repository Actions run Optuna by default. For a faster local smoke test:

```
RUN_OPTUNA=0
```

You can also control the number of trials with:

```
OPTUNA_TRIALS=5
```

The notebook prints:
- whether Optuna was requested;
- whether it actually ran;
- number of completed trials;
- best parameters.

## Explainability

The notebook includes coefficient-based explainability for the TF-IDF + Logistic Regression model, showing which unigram/bigram features push predictions toward positive or negative sentiment.

This provides a transparent benchmark even when the deep-learning models are less interpretable.

## 3-Class Classification

Neutral is not discarded from the project. A separate **3-class TF-IDF + Logistic Regression** model evaluates:

- positive
- negative
- neutral

This benchmark also uses class weighting and reports Macro F1, Balanced Accuracy, and macro PR-AUC.

## Error Analysis

The notebook exports:

`results/error_analysis.csv`

and surfaces false positives, false negatives, and error rates by comment-length bucket.

## Results Artifacts

The notebook writes reusable outputs to `results/`, including:

- `data_quality_profile.csv`
- `human_validation_sample_for_annotation.csv`
- `human_validation_scoring.csv` (only when human labels exist)
- `model_comparison.csv`
- `tfidf_binary_metrics.csv`
- `bilstm_binary_metrics.csv`
- `bigru_binary_metrics.csv`
- `three_class_metrics.csv`
- `optuna_trials.csv`
- `error_analysis.csv`
- `explainability_top_terms.csv`
- `project_summary.json`

## Repository Structure

```
.
├── data/
│   ├── mbg_comments_raw.csv
│   ├── mbg_comments_clean.csv
│   ├── mbg_comments_to_label.csv
│   └── mbg_comments_labeled.csv
├── Scripts/
│   ├── 01_data_quality_cleaning.py
│   ├── 02_prepare_labeling.py
│   ├── 03_merge_labels.py
│   └── 04_prepare_human_validation.py
├── results/
├── sentiment_analysis_mbg_youtube.ipynb
├── requirements.txt
└── .github/
    └── workflows/
        └── execute-mbg-notebook.yml
```

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

`sentiment_analysis_mbg_youtube.ipynb`

The notebook is designed to run from the repository root. The GitHub Actions workflow also executes it automatically when the notebook or its labeled dataset changes.

## Important Interpretation Boundary

This project describes the collected **YouTube sample** and the model's performance against the repository's labeling scheme.

It does not:
- establish causality from discussion spikes;
- represent the opinion of the entire population;
- treat AI-assisted labels as human gold standard;
- report a fabricated human-validation score.

The main data-quality caveat that remains is the need to complete the blinded human-validation sample. That step is made explicit and reproducible in the repository.

## Portfolio Takeaway

This project goes beyond “train a sentiment model.” It demonstrates a complete analytics workflow with **data quality, leakage prevention, label validation workflow, NLP preprocessing, EDA, imbalanced classification, hyperparameter tuning, PR-AUC, error analysis, model explainability, and 3-class benchmarking**.
