# MBG YouTube Sentiment Analysis & Text Mining

End-to-end **Text Mining + NLP + Machine Learning + Deep Learning** project that analyzes 12,000 YouTube comments related to the Makan Bergizi Gratis (MBG) discussion.

## Project Overview

The project turns unstructured YouTube comments into measurable information through:

**Data validation → text preprocessing → exploratory text mining → sentiment analysis → sentiment trend → binary classification → model benchmarking → evaluation → insight**

### Questions addressed

- What is the sentiment composition of the collected comments?
- When is discussion volume highest?
- Which words dominate the conversation after preprocessing?
- Can positive vs negative comments be classified automatically?
- How does a classical NLP baseline compare with BiLSTM and BiGRU?

## Dataset

Primary file:

`data/mbg_comments_labeled.csv`

Current portfolio dataset:

| Metric | Value |
|---|---:|
| Total comments | 12,000 |
| Positive | 220 (1.83%) |
| Negative | 2,880 (24.00%) |
| Neutral | 8,900 (74.17%) |
| Binary modeling rows | 3,100 |
| Binary negative share | 92.9% |
| Binary positive share | 7.1% |

### Labeling note

Sentiment labels use **AI-assisted semantic labeling**. They are not human gold-standard annotations. Model metrics therefore measure performance against this labeling scheme rather than against an independently verified human benchmark.

## NLP & Modeling

### Text preprocessing

The notebook handles Indonesian YouTube-style text through:

- HTML/entity normalization
- URL and mention removal
- hashtag symbol normalization
- lowercasing and punctuation cleanup
- slang normalization
- Indonesian stopword removal
- explicit preservation of negation words
- optional Sastrawi stemming

### Models

1. **TF-IDF + Logistic Regression** — classical NLP baseline
2. **Bidirectional LSTM** — deep-learning baseline
3. **Bidirectional GRU** — deep-learning candidate
4. **Optuna** — optional hyperparameter tuning for BiGRU

### Evaluation

The notebook reports:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Confusion matrix
- Training / validation curves

Macro F1 is emphasized because the binary dataset is highly imbalanced.

## Portfolio Run Results

Embedded outputs in the notebook come from an executed run on the 12,000-comment dataset.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | **93.39%** | **74.66%** |
| BiLSTM baseline | 88.87% | 64.10% |
| BiGRU candidate | 88.23% | 63.29% |

On the 620-row test set, there are 44 positive and 576 negative examples.

### Key dataset findings

- **Neutral is the dominant label:** 8,900 comments (74.17%).
- **Peak discussion volume:** 21 September 2026 with 895 comments.
- **Top five words:** `mbg`, `gak`, `tidak`, `makan`, `anak`.
- The binary classification task is strongly imbalanced, so accuracy should not be read alone.
- In this run, the classical TF-IDF baseline recorded a higher Macro F1 than the tested deep-learning models.

## Important Interpretation Boundary

The project describes the collected **YouTube dataset**, not the opinion of the entire population. Discussion spikes are descriptive and are not treated as proof of a specific cause.

## Limitations

- Platform-specific sample; not automatically population-representative.
- Online comments can contain slang, sarcasm, typo, emoji, spam, and ambiguity.
- AI-assisted labels are not an independently validated human gold standard.
- Binary modeling removes neutral from training.
- RNN-based models have limitations with long context and sarcasm.

## Repository Structure

```
.
├── data/
│   ├── mbg_comments_raw.csv
│   ├── mbg_comments_clean.csv
│   ├── mbg_comments_to_label.csv
│   └── mbg_comments_labeled.csv
├── Scripts/
├── results/
│   ├── model_comparison.csv
│   └── project_summary.json
├── sentiment_analysis_mbg_youtube.ipynb
├── requirements.txt
└── .github/
    └── workflows/
        └── execute-mbg-notebook.yml
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

`sentiment_analysis_mbg_youtube.ipynb`

The notebook is designed to run from the repository root.

Optuna is disabled by default to keep the normal portfolio run practical. To enable it:

**Windows CMD**
```cmd
set RUN_OPTUNA=1
```

**macOS / Linux**
```bash
export RUN_OPTUNA=1
```

## Portfolio Notes

This repository separates:

- **EDA / text mining** for understanding the collected conversation
- **sentiment analysis** for dataset-level sentiment composition
- **model benchmarking** for positive-vs-negative classification
- **interpretation** for converting outputs into data-driven findings and clearly stated limitations

## Verification

The checked-in notebook is portfolio-ready and includes:
- stratified train/test split before model training;
- TF-IDF + Logistic Regression baseline with embedded test results;
- guarded BiLSTM/BiGRU sections with the required GRU import;
- no duplicate tokenization section;
- reproducibility notes and dataset-specific interpretation boundaries.

The notebook has no embedded error outputs in the committed version.
