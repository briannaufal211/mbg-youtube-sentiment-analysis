from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "mbg_comments_clean.csv"
SAMPLE_FILE = DATA_DIR / "mbg_comments_to_label.csv"

LABEL_SAMPLE_SIZE = 1500
RANDOM_STATE = 42
ALLOWED_LABELS = {"positive", "negative", "neutral"}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}. Jalankan 01_data_quality_cleaning.py terlebih dahulu."
        )

    df = pd.read_csv(INPUT_FILE)

    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))

    sample_size = min(LABEL_SAMPLE_SIZE, len(df))

    if "video_id" in df.columns and df["video_id"].nunique() > 1:
        parts = []
        for _, group in df.groupby("video_id", sort=False):
            n = min(len(group), max(1, round(sample_size * len(group) / len(df))))
            parts.append(group.sample(n=n, random_state=RANDOM_STATE))
        sample_df = pd.concat(parts, ignore_index=True)
        if len(sample_df) > sample_size:
            sample_df = sample_df.sample(sample_size, random_state=RANDOM_STATE)
    else:
        sample_df = df.sample(sample_size, random_state=RANDOM_STATE)

    label_columns = [
        "row_id", "comment", "comment_clean", "video_title",
        "channel_title", "published_at", "like_count",
        "sentiment", "label_notes"
    ]
    available = [c for c in label_columns if c in sample_df.columns]
    sample_df = sample_df[available].copy()

    if "sentiment" not in sample_df.columns:
        sample_df["sentiment"] = ""
    if "label_notes" not in sample_df.columns:
        sample_df["label_notes"] = ""

    sample_df = sample_df.sort_values("row_id")
    sample_df.to_csv(SAMPLE_FILE, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("LABELING DATASET CREATED")
    print("=" * 80)
    print(f"Clean rows    : {len(df):,}")
    print(f"Rows to label : {len(sample_df):,}")
    print(f"Output        : {SAMPLE_FILE}")
    print("\nAllowed labels: positive / negative / neutral")
    print("POSITIVE = dukungan, manfaat, kepuasan, atau penilaian positif.")
    print("NEGATIVE = kritik, penolakan, keluhan, atau penilaian negatif.")
    print("NEUTRAL  = informatif/deskriptif atau tidak ada polaritas yang jelas.")
    print("Untuk komentar campuran, gunakan SENTIMEN UTAMA dan tulis alasan di label_notes.")


if __name__ == "__main__":
    main()
