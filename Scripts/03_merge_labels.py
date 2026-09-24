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
        bad = label_df.loc[invalid_mask, ["sentiment"]]
        raise ValueError(
            "Ditemukan label tidak valid. Gunakan positive / negative / neutral.\n"
            + bad.head(20).to_string(index=False)
        )

    labeled_part = label_df[
        label_df["sentiment"].isin(ALLOWED_LABELS)
    ].copy()

    # Prefer a stable source identifier when available.
    # Fall back to row_id for legacy annotation files.
    if "comment_id" in clean_df.columns and "comment_id" in labeled_part.columns:
        join_key = "comment_id"

        if labeled_part[join_key].duplicated().any():
            raise ValueError("Terdapat duplicate comment_id di file labeling.")

        labeled_part = labeled_part[[join_key, "sentiment"]].copy()

    else:
        if "row_id" not in clean_df.columns or "row_id" not in labeled_part.columns:
            raise ValueError(
                "Dibutuhkan pasangan key: comment_id (preferred) atau row_id."
            )

        if labeled_part["row_id"].duplicated().any():
            raise ValueError("Terdapat duplicate row_id di file labeling.")

        join_key = "row_id"
        labeled_part = labeled_part[[join_key, "sentiment"]].copy()

    final_df = clean_df.merge(
        labeled_part,
        on=join_key,
        how="left",
        suffixes=("", "_new"),
        validate="one_to_one",
    )

    if "sentiment_new" in final_df.columns:
        final_df["sentiment"] = (
            final_df["sentiment_new"].combine_first(final_df.get("sentiment"))
        )
        final_df = final_df.drop(columns=["sentiment_new"])

    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    labeled_count = int(final_df["sentiment"].isin(ALLOWED_LABELS).sum())

    print("=" * 80)
    print("LABEL MERGE COMPLETE")
    print("=" * 80)
    print(f"Join key         : {join_key}")
    print(f"Total clean rows : {len(final_df):,}")
    print(f"Labeled rows     : {labeled_count:,}")
    print(f"Unlabeled rows   : {len(final_df) - labeled_count:,}")
    print(f"Output           : {OUTPUT_FILE}")
    print(
        "\nNote: this helper script merges the current annotation sample into "
        "the clean dataset. The portfolio's complete 12,000-row labeled snapshot "
        "is maintained separately and enriched by 05_restore_labeled_metadata.py."
    )


if __name__ == "__main__":
    main()
