import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEAN_FILE = os.path.join(
    BASE_DIR,
    "data",
    "mbg_comments_clean.csv"
)

LABEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "mbg_comments_to_label.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "mbg_comments_labeled.csv"
)

ALLOWED_LABELS = {
    "positive",
    "negative",
    "neutral"
}


def main():
    if not os.path.exists(CLEAN_FILE):
        raise FileNotFoundError(
            f"File clean tidak ditemukan: {CLEAN_FILE}"
        )

    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(
            f"File labeling tidak ditemukan: {LABEL_FILE}"
        )

    clean_df = pd.read_csv(CLEAN_FILE)
    label_df = pd.read_csv(LABEL_FILE)

    if "row_id" not in clean_df.columns:
        raise ValueError("Clean file tidak punya row_id.")

    if "row_id" not in label_df.columns:
        raise ValueError("Label file tidak punya row_id.")

    if "sentiment" not in label_df.columns:
        raise ValueError(
            "Kolom sentiment belum ada di file labeling."
        )

    # Normalize labels
    label_df["sentiment"] = (
        label_df["sentiment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Validate labels
    invalid_mask = (
        label_df["sentiment"] != ""
    ) & ~(
        label_df["sentiment"].isin(
            ALLOWED_LABELS
        )
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count > 0:
        print(
            f"ERROR: Ada {invalid_count} label tidak valid."
        )
        print(
            label_df.loc[
                invalid_mask,
                ["row_id", "sentiment"]
            ].head(20)
        )
        print(
            "\nGunakan hanya: positive / negative / neutral"
        )
        return

    # Hanya label yang sudah diisi
    labeled_part = label_df[
        label_df["sentiment"].isin(
            ALLOWED_LABELS
        )
    ][
        ["row_id", "sentiment", "label_notes"]
    ].copy()

    # Check duplicate row_id in label file
    duplicate_labels = int(
        labeled_part["row_id"].duplicated().sum()
    )

    if duplicate_labels > 0:
        raise ValueError(
            f"Terdapat {duplicate_labels} row_id duplicate "
            "di file labeling."
        )

    # Merge labels onto clean dataset
    final_df = clean_df.merge(
        labeled_part[
            ["row_id", "sentiment"]
        ],
        on="row_id",
        how="left",
        suffixes=("", "_new")
    )

    # If clean_df already had sentiment, prefer the new labeled value
    if "sentiment_new" in final_df.columns:
        final_df["sentiment"] = (
            final_df["sentiment_new"]
            .combine_first(
                final_df["sentiment"]
            )
        )
        final_df = final_df.drop(
            columns=["sentiment_new"]
        )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    labeled_count = int(
        final_df["sentiment"]
        .isin(ALLOWED_LABELS)
        .sum()
    )

    unlabeled_count = len(final_df) - labeled_count

    print("=" * 80)
    print("LABEL MERGE COMPLETE")
    print("=" * 80)

    print(
        f"Total clean rows : {len(final_df):,}"
    )

    print(
        f"Labeled rows     : {labeled_count:,}"
    )

    print(
        f"Unlabeled rows   : {unlabeled_count:,}"
    )

    print(
        f"\nOutput           : {OUTPUT_FILE}"
    )

    if labeled_count > 0:
        print("\nLabel distribution:")
        print(
            final_df[
                final_df["sentiment"]
                .isin(ALLOWED_LABELS)
            ]["sentiment"]
            .value_counts()
        )


if __name__ == "__main__":
    main()
