import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "mbg_comments_clean.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data"
)

SAMPLE_FILE = os.path.join(
    OUTPUT_DIR,
    "mbg_comments_to_label.csv"
)

# Jumlah data yang akan diberi label manual.
# 1500 adalah titik awal yang baik untuk dataset 10k-15k.
LABEL_SAMPLE_SIZE = 1500

RANDOM_STATE = 42

ALLOWED_LABELS = {
    "positive",
    "negative",
    "neutral"
}


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}\n"
            "Jalankan script 01_data_quality_cleaning.py terlebih dahulu."
        )

    df = pd.read_csv(INPUT_FILE)

    # Pastikan ada row_id
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))

    # Jika kolom sentiment sudah ada, jangan menimpa label lama.
    if "sentiment" not in df.columns:
        df["sentiment"] = ""

    # --------------------------------------------------------
    # Sampling:
    # 1) Utamakan variasi video bila video_id tersedia.
    # 2) Sisanya random sample.
    # --------------------------------------------------------
    sample_size = min(
        LABEL_SAMPLE_SIZE,
        len(df)
    )

    if "video_id" in df.columns and df["video_id"].nunique() > 1:

        # Ambil proporsional dari setiap video
        parts = []

        grouped = list(
            df.groupby("video_id", sort=False)
        )

        for _, group in grouped:

            proportion = len(group) / len(df)

            n = max(
                1,
                round(sample_size * proportion)
            )

            n = min(n, len(group))

            parts.append(
                group.sample(
                    n=n,
                    random_state=RANDOM_STATE
                )
            )

        sample_df = pd.concat(
            parts,
            ignore_index=True
        )

        if len(sample_df) > sample_size:
            sample_df = sample_df.sample(
                n=sample_size,
                random_state=RANDOM_STATE
            )

    else:

        sample_df = df.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )


    # --------------------------------------------------------
    # Kolom untuk labeling
    # --------------------------------------------------------

    label_columns = [
        "row_id",
        "comment",
        "comment_clean",
        "video_title",
        "channel_title",
        "published_at",
        "like_count",
        "sentiment",
        "label_notes"
    ]

    available_columns = [
        c for c in label_columns
        if c in sample_df.columns
    ]

    sample_df = sample_df[
        available_columns
    ].copy()

    if "sentiment" not in sample_df.columns:
        sample_df["sentiment"] = ""

    if "label_notes" not in sample_df.columns:
        sample_df["label_notes"] = ""


    sample_df = sample_df.sort_values(
        "row_id"
    )

    sample_df.to_csv(
        SAMPLE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 80)
    print("LABELING DATASET CREATED")
    print("=" * 80)
    print(f"Clean rows       : {len(df):,}")
    print(f"Rows to label    : {len(sample_df):,}")
    print(f"Output           : {SAMPLE_FILE}")

    print("\nIsi kolom 'sentiment' dengan:")
    print("  positive")
    print("  negative")
    print("  neutral")

    print("\nPanduan labeling:")
    print("POSITIVE = komentar menunjukkan dukungan, manfaat, kepuasan, atau penilaian positif.")
    print("NEGATIVE = komentar menunjukkan kritik, penolakan, keluhan, atau penilaian negatif.")
    print("NEUTRAL  = informatif/deskriptif atau tidak menunjukkan sentimen yang jelas.")

    print("\nUntuk komentar campuran, beri label berdasarkan SENTIMEN UTAMA.")
    print("Gunakan label_notes untuk kasus ambigu.")


if __name__ == "__main__":
    main()
