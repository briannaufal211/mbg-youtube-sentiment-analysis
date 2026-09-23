from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "mbg_comments_labeled.csv"
OUTPUT_FILE = RESULTS_DIR / "human_validation_sample_for_annotation.csv"

SEED = 42
SAMPLE_PER_LABEL = 100
LABELS = ["positive", "negative", "neutral"]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"row_id", "comment", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    df["sentiment"] = (
        df["sentiment"].fillna("").astype(str).str.strip().str.lower()
    )
    df = df[df["sentiment"].isin(LABELS)].copy()
    df["dedup_key"] = (
        df["comment"].fillna("").astype(str).str.strip().str.lower()
    )
    df = df[df["dedup_key"].ne("")].drop_duplicates("dedup_key")

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        if {"validation_id", "row_id", "comment", "human_label", "human_notes"}.issubset(existing.columns):
            print(f"Validation sample already exists: {OUTPUT_FILE}")
            return

    parts = []
    for label in LABELS:
        group = df[df["sentiment"] == label]
        n = min(SAMPLE_PER_LABEL, len(group))
        if n:
            parts.append(group.sample(n=n, random_state=SEED))

    sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "validation_id", np.arange(1, len(sample) + 1))

    if "published_at" not in sample.columns:
        sample["published_at"] = ""

    sample = sample[["validation_id", "row_id", "comment", "published_at"]]
    sample["human_label"] = ""
    sample["human_notes"] = ""

    sample.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("BLINDED HUMAN VALIDATION SAMPLE CREATED")
    print("=" * 80)
    print(f"Rows     : {len(sample):,}")
    print(f"Output   : {OUTPUT_FILE}")
    print("The AI label is intentionally not included.")


if __name__ == "__main__":
    main()
