from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RAW_FILE = DATA_DIR / "mbg_comments_raw.csv"
LABELED_FILE = DATA_DIR / "mbg_comments_labeled.csv"
TEMP_FILE = DATA_DIR / "mbg_comments_labeled.tmp.csv"

REQUIRED_RAW = {
    "comment_id",
    "video_id",
    "comment",
    "published_at",
}
REQUIRED_LABEL = {
    "comment",
    "comment_clean",
    "published_at",
    "sentiment",
}


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file tidak ditemukan: {RAW_FILE}")
    if not LABELED_FILE.exists():
        raise FileNotFoundError(f"Labeled file tidak ditemukan: {LABELED_FILE}")

    raw = pd.read_csv(RAW_FILE)
    labeled = pd.read_csv(LABELED_FILE)

    missing_raw = REQUIRED_RAW - set(raw.columns)
    missing_label = REQUIRED_LABEL - set(labeled.columns)

    if missing_raw:
        raise ValueError(f"Raw dataset kehilangan kolom: {sorted(missing_raw)}")
    if missing_label:
        raise ValueError(f"Labeled dataset kehilangan kolom: {sorted(missing_label)}")

    if len(raw) != len(labeled):
        raise ValueError(
            f"Row count mismatch: raw={len(raw):,}, labeled={len(labeled):,}. "
            "Metadata enrichment dihentikan agar tidak melakukan join yang salah."
        )

    # The current portfolio labeling snapshot follows raw-row order.
    # Validate row alignment before adding metadata.
    if "row_id" in labeled.columns:
        expected = pd.Series(range(1, len(labeled) + 1), name="row_id")
        actual = pd.to_numeric(labeled["row_id"], errors="coerce")
        if not actual.reset_index(drop=True).equals(expected):
            raise ValueError("row_id labeled dataset tidak berurutan 1..N.")

    for col in ["comment", "published_at"]:
        if not raw[col].astype(str).reset_index(drop=True).equals(
            labeled[col].astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                f"Alignment check gagal pada kolom '{col}'. "
                "Tidak ada file yang ditulis."
            )

    # Keep the existing sentiment labels and cleaned text exactly as-is,
    # while restoring source metadata from the raw collection.
    output = pd.DataFrame({
        "row_id": range(1, len(raw) + 1),
        "comment_id": raw["comment_id"],
        "video_id": raw["video_id"],
        "comment": labeled["comment"],
        "comment_clean": labeled["comment_clean"],
        "published_at": raw["published_at"],
        "updated_at": raw.get("updated_at", pd.Series([""] * len(raw))),
        "comment_like_count": raw.get("comment_like_count", pd.Series([""] * len(raw))),
        "video_title": raw.get("video_title", pd.Series([""] * len(raw))),
        "channel_id": raw.get("channel_id", pd.Series([""] * len(raw))),
        "channel_title": raw.get("channel_title", pd.Series([""] * len(raw))),
        "video_published_at": raw.get("video_published_at", pd.Series([""] * len(raw))),
        "view_count": raw.get("view_count", pd.Series([""] * len(raw))),
        "video_like_count": raw.get("video_like_count", pd.Series([""] * len(raw))),
        "video_comment_count": raw.get("video_comment_count", pd.Series([""] * len(raw))),
        "sentiment": labeled["sentiment"],
    })

    output.to_csv(TEMP_FILE, index=False, encoding="utf-8-sig")
    TEMP_FILE.replace(LABELED_FILE)

    print("=" * 80)
    print("LABELED DATASET METADATA RESTORED")
    print("=" * 80)
    print(f"Rows                : {len(output):,}")
    print(f"Unique comment_id   : {output['comment_id'].nunique():,}")
    print(f"Unique video_id     : {output['video_id'].nunique():,}")
    print(f"Unique channel_id   : {output['channel_id'].nunique():,}")
    print(f"Output columns      : {list(output.columns)}")
    print(f"Output              : {LABELED_FILE}")


if __name__ == "__main__":
    main()
