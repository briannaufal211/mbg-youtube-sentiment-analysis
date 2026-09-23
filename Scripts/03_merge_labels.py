from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CLEAN_FILE = DATA_DIR / "mbg_comments_clean.csv"
LABEL_FILE = DATA_DIR / "mbg_comments_to_label.csv"
OUTPUT_FILE = DATA_DIR / "mbg_comments_labeled.csv"

ALLOWED_LABELS = {"positive", "negative", "neutral"}


def main():
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(f"Clean file tidak ditemukan: {CLEAN_FILE}")
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Label file tidak ditemukan: {LABEL_FILE}")

    clean_df = pd.read_csv(CLEAN_FILE)
    label_df = pd.read_csv(LABEL_FILE)

    for frame_name, frame in [("clean", clean_df), ("label", label_df)]:
        if "row_id" not in frame.columns:
            raise ValueError(f"File {frame_name} tidak punya row_id.")

    if "sentiment" not in label_df.columns:
        raise ValueError("Kolom sentiment belum ada di file labeling.")

    label_df["sentiment"] = (
        label_df["sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_mask = (
        label_df["sentiment"].ne("")
        & ~label_df["sentiment"].isin(ALLOWED_LABELS)
    )

    if invalid_mask.any():
        bad = label_df.loc[invalid_mask, ["row_id", "sentiment"]]
        raise ValueError(
            "Ditemukan label tidak valid. Gunakan positive / negative / neutral.\n"
            + bad.head(20).to_string(index=False)
        )

    labeled_part = label_df[
        label_df["sentiment"].isin(ALLOWED_LABELS)
    ][["row_id", "sentiment"]].copy()

    if labeled_part["row_id"].duplicated().any():
        raise ValueError("Terdapat duplicate row_id di file labeling.")

    final_df = clean_df.merge(
        labeled_part,
        on="row_id",
        how="left",
        suffixes=("", "_new")
    )

    if "sentiment_new" in final_df.columns:
        final_df["sentiment"] = (
            final_df["sentiment_new"].combine_first(final_df["sentiment"])
        )
        final_df = final_df.drop(columns=["sentiment_new"])

    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    labeled_count = int(final_df["sentiment"].isin(ALLOWED_LABELS).sum())
    print("=" * 80)
    print("LABEL MERGE COMPLETE")
    print("=" * 80)
    print(f"Total clean rows : {len(final_df):,}")
    print(f"Labeled rows     : {labeled_count:,}")
    print(f"Unlabeled rows   : {len(final_df) - labeled_count:,}")
    print(f"Output           : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
