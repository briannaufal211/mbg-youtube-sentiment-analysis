# MBG YouTube Sentiment Analysis & Text Mining

End-to-end **Text Mining + NLP + Machine Learning + Deep Learning** project for analyzing 12,000 YouTube comments related to the MBG discussion.

## Project Objective

The project turns unstructured YouTube comments into measurable outputs through:

**Data quality → duplicate-leakage control → label validation workflow → text preprocessing → exploratory text mining → binary classification → model benchmarking → imbalance-aware evaluation → Optuna tuning → error analysis → explainability → 3-class classification**

The notebook is intentionally designed to distinguish descriptive findings from model-evaluation evidence and dataset limitations.

## Dataset

Primary portfolio file:

`data/mbg_comments_labeled.csv`

Current dataset snapshot in the repository contains **12,000 labeled comments** and preserves the source collection metadata used for downstream analysis.

Key fields include:

- `comment_id`
- `video_id`
- `comment`
- `comment_clean`
- `published_at`
- `comment_like_count`
- `video_title`
- `channel_id`
- `channel_title`
- `sentiment`

### Dataset lineage

The repository intentionally keeps different data stages with different purposes:

- `data/mbg_comments_raw.csv` — 12,000 raw comments collected from YouTube.
- `data/mbg_comments_clean.csv` — 11,557 rows after the cleaning helper removes invalid/empty comments; this is a preprocessing/annotation helper dataset.
- `data/mbg_comments_to_label.csv` — annotation sample prepared by `Scripts/02_prepare_labeling.py`.
- `data/mbg_comments_labeled.csv` — the complete 12,000-row portfolio labeling snapshot, aligned to the raw collection and retaining source metadata.

The different row counts are therefore intentional: the **clean helper dataset is filtered**, while the **portfolio labeled snapshot preserves the full labeled raw-aligned corpus**. The notebook performs its own model-ready validity and duplicate controls before train/validation/test splitting.

### Labeling

The current portfolio dataset uses **AI-assisted semantic labeling** for:
- `positive`
- `negative`
- `neutral`

The repository also contains `Scripts/02_prepare_labeling.py`, which prepares a sample for annotation and now retains stable comment/video metadata.

For validation, the project uses a **binary sentiment framework** with:
- `positive`
- `negative`

The validation sample is stored in:

`results/human_validation_sample_for_annotation.csv`

The completed validation file contains reviewed labels in `human_label` and is treated as **human-reviewed AI-assisted annotation**, rather than an independent gold-standard annotation study.

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
│   ├── 04_prepare_human_validation.py
│   ├── 05_restore_labeled_metadata.py
│   ├── 06_enrich_commenter_metadata.py
│   ├── 07_coordination_pattern_analysis.py
│   └── 08_coordination_multi_signal_review.py
├── results/
├── sentiment_analysis_mbg_youtube.ipynb
├── requirements.txt
└── .github/
    └── workflows/
        └── execute-mbg-notebook.yml
```

### Metadata consistency safeguard

`Scripts/05_restore_labeled_metadata.py` validates the 12,000-row raw/labeled alignment and restores source metadata into the portfolio labeled snapshot before the notebook runs.

The GitHub Actions workflow executes this safeguard automatically, then commits the restored dataset together with the executed notebook/results.


## Coordination Pattern Analysis Extension

After commenter metadata enrichment, the project can be extended with a **descriptive coordination-pattern analysis** using:

- `comment_id`
- `commenter_channel_id`
- `commenter_name`
- `parent_comment_id`
- `video_id`
- `published_at`
- `comment_clean`
- `sentiment`

Run:

```bash
python Scripts/07_coordination_pattern_analysis.py
```

The script produces descriptive outputs under:

`data/coordination_analysis/`

including:

- `coordination_exact_repetitions.csv`
- `coordination_cross_video_repetition.csv`
- `coordination_temporal_patterns.csv`
- `coordination_similar_text_pairs.csv`
- `commenter_activity_summary_anonymized.csv`
- `coordination_analysis_summary.json`

The analysis is intentionally framed as **coordination-like pattern detection**, not definitive “buzzer detection”. Repeated text, high text similarity, synchronized posting, or cross-video repetition are signals for further review; they do not by themselves establish that an account is a bot, buzzer, or coordinated actor.

The exported commenter activity table uses a one-way hash rather than exposing commenter channel IDs in the portfolio analysis output.

### Multi-signal review

After running the first coordination-pattern analysis, run:

```bash
python Scripts/08_coordination_multi_signal_review.py
```

This second-stage script aggregates several descriptive indicators at commenter level:

- exact repeated-text participation
- cross-video repeated-text participation
- highly similar text-pair participation
- temporal-pattern participation

It reports **indicator counts and multi-pattern observations**, not a buzzer/bot classification or risk score. The commenter output is anonymized with a one-way hash. A separate pattern-review table highlights repeated-text clusters that meet the project's descriptive review criteria.

Outputs:

- `data/coordination_analysis/commenter_pattern_indicators_anonymized.csv`
- `data/coordination_analysis/coordination_pattern_review_candidates.csv`
- `data/coordination_analysis/coordination_screening_summary.json`

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

The validation sample has now been reviewed using a binary sentiment scheme (`positive` / `negative`). Because the starting labels were AI-assisted and the human review was based on those proposed labels, the result should be described as human-reviewed annotation rather than an independent gold-standard benchmark.

## Portfolio Takeaway

This project goes beyond “train a sentiment model.” It demonstrates a complete analytics workflow with **data quality, leakage prevention, label validation workflow, NLP preprocessing, EDA, imbalanced classification, hyperparameter tuning, PR-AUC, error analysis, model explainability, and 3-class benchmarking**.