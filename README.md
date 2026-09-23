# MBG YouTube Sentiment Analysis & Text Mining

End-to-end text mining and sentiment classification project using **12,000 YouTube comments** related to the Makan Bergizi Gratis (MBG) discussion.

## Project Objective

The project turns unstructured YouTube comments into measurable information through:

1. Data validation and profiling
2. Indonesian text preprocessing
3. Exploratory text mining
4. Positive / negative / neutral sentiment analysis
5. Sentiment trend analysis over time
6. Binary sentiment classification (positive vs negative)
7. Model evaluation and error analysis

## Modeling

The notebook includes:

- **TF-IDF + Logistic Regression** as a lightweight baseline
- **Bidirectional LSTM** as the deep-learning baseline
- **Bidirectional GRU** as the final candidate
- **Optuna** for optional hyperparameter tuning

Evaluation includes accuracy, precision, recall, class-level F1, Macro F1, confusion matrix, and training/validation curves.

## Dataset

Primary file:

`data/mbg_comments_labeled.csv`

Current labeled dataset:
- 12,000 comments
- 220 positive
- 2,880 negative
- 8,900 neutral

For binary modeling, neutral comments are excluded, leaving 3,100 positive/negative comments.

### Labeling note

Sentiment labels are **AI-assisted semantic labels**, not human-annotated gold-standard labels. Model metrics should therefore be interpreted as performance against this labeling scheme.

## Repository Structure

```
.
├── data/
│   ├── mbg_comments_raw.csv
│   └── mbg_comments_labeled.csv
├── Scripts/
├── sentiment_analysis_mbg_youtube.ipynb
├── requirements.txt
└── .github/
    └── workflows/
        └── execute-mbg-notebook.yml
```

## How to Run

Create a Python environment, install dependencies, then open:

`sentiment_analysis_mbg_youtube.ipynb`

For a normal local run, Optuna is disabled by default.

To enable tuning:

```bash
set RUN_OPTUNA=1
```

On macOS/Linux:

```bash
export RUN_OPTUNA=1
```

## Portfolio Notes

The notebook is intentionally data-driven: numerical results and insights are generated from the project dataset during execution rather than copied from another notebook.

YouTube comments should be treated as a platform-specific sample, not as a direct estimate of the views or opinions of the entire population.
