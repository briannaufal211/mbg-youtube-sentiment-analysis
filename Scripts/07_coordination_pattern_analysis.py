from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "mbg_comments_coordination_ready.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "coordination_analysis"

REQUIRED_COLUMNS = {
    "comment_id",
    "video_id",
    "comment",
    "comment_clean",
    "published_at",
    "sentiment",
    "commenter_channel_id",
    "commenter_name",
}

MIN_TEXT_CHARS = 20
MIN_TEXT_WORDS = 4
TIME_BUCKET = "10min"


def safe_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def make_pattern_key(text: str) -> str:
    return " ".join(str(text).split())


def is_eligible_text(text: str) -> bool:
    clean = make_pattern_key(text)
    return (
        len(clean) >= MIN_TEXT_CHARS
        and len(clean.split()) >= MIN_TEXT_WORDS
    )


def load_data(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input tidak ditemukan: {input_file}\n"
            "Jalankan 06_enrich_commenter_metadata.py terlebih dahulu."
        )

    df = pd.read_csv(input_file, dtype=str).fillna("")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing)}"
        )

    if df["comment_id"].duplicated().any():
        raise ValueError("comment_id harus unik.")

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True,
    )

    if df["published_at"].isna().any():
        raise ValueError(
            f"Terdapat {int(df['published_at'].isna().sum())} "
            "timestamp yang tidak valid."
        )

    df["pattern_text"] = df["comment_clean"].map(make_pattern_key)
    df["eligible_for_pattern_analysis"] = df["pattern_text"].map(
        is_eligible_text
    )

    return df


def exact_repetition_analysis(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df[df["eligible_for_pattern_analysis"]].copy()

    grouped = (
        eligible.groupby("pattern_text", dropna=False)
        .agg(
            comment_count=("comment_id", "size"),
            distinct_commenters=("commenter_channel_id", "nunique"),
            distinct_videos=("video_id", "nunique"),
            first_published_at=("published_at", "min"),
            last_published_at=("published_at", "max"),
        )
        .reset_index()
    )

    grouped = grouped[
        (grouped["comment_count"] >= 2)
        & (
            (grouped["distinct_commenters"] >= 2)
            | (grouped["distinct_videos"] >= 2)
        )
    ].copy()

    grouped["pattern_type"] = np.where(
        grouped["distinct_commenters"] >= 2,
        "repeated_text_across_commenters",
        "repeated_text_across_videos",
    )

    grouped["pattern_text_length"] = grouped["pattern_text"].str.len()

    return grouped.sort_values(
        ["distinct_commenters", "distinct_videos", "comment_count"],
        ascending=[False, False, False],
    )


def cross_video_repetition_analysis(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df[df["eligible_for_pattern_analysis"]].copy()

    grouped = (
        eligible.groupby("pattern_text", dropna=False)
        .agg(
            comment_count=("comment_id", "size"),
            distinct_videos=("video_id", "nunique"),
            distinct_commenters=("commenter_channel_id", "nunique"),
            sentiment_count=("sentiment", "nunique"),
        )
        .reset_index()
    )

    grouped = grouped[
        (grouped["distinct_videos"] >= 2)
        & (grouped["distinct_commenters"] >= 2)
    ].copy()

    return grouped.sort_values(
        ["distinct_videos", "distinct_commenters", "comment_count"],
        ascending=[False, False, False],
    )


def temporal_pattern_analysis(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df[df["eligible_for_pattern_analysis"]].copy()
    eligible["time_bucket"] = eligible["published_at"].dt.floor(TIME_BUCKET)

    grouped = (
        eligible.groupby(
            ["time_bucket", "pattern_text"],
            dropna=False,
        )
        .agg(
            comment_count=("comment_id", "size"),
            distinct_commenters=("commenter_channel_id", "nunique"),
            distinct_videos=("video_id", "nunique"),
        )
        .reset_index()
    )

    # Keep only patterns repeated by more than one commenter in the same
    # 10-minute window. This is a descriptive temporal signal, not proof
    # of coordination.
    grouped = grouped[
        (grouped["comment_count"] >= 2)
        & (grouped["distinct_commenters"] >= 2)
    ].copy()

    return grouped.sort_values(
        ["time_bucket", "distinct_commenters", "comment_count"],
        ascending=[True, False, False],
    )


def similar_text_pairs(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    eligible = df[df["eligible_for_pattern_analysis"]].copy()

    if len(eligible) < 2:
        return pd.DataFrame(
            columns=[
                "comment_id_a",
                "comment_id_b",
                "similarity",
                "same_commenter",
                "same_video",
                "video_id_a",
                "video_id_b",
                "sentiment_a",
                "sentiment_b",
            ]
        )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(eligible["pattern_text"])

    if matrix.shape[1] == 0:
        return pd.DataFrame()

    n_neighbors = min(6, len(eligible))
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )
    model.fit(matrix)

    distances, indices = model.kneighbors(matrix)

    records = []
    rows = list(eligible.itertuples(index=False))

    for i, neighbors in enumerate(indices):
        for distance, j in zip(distances[i], neighbors):
            if i == j:
                continue

            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue

            a = rows[i]
            b = rows[j]

            # Keep each pair once.
            if str(a.comment_id) >= str(b.comment_id):
                continue

            same_commenter = (
                str(a.commenter_channel_id)
                == str(b.commenter_channel_id)
            )
            same_video = str(a.video_id) == str(b.video_id)

            # Cross-commenter or cross-video similarity is the more useful
            # descriptive signal for this extension.
            if same_commenter and same_video:
                continue

            records.append(
                {
                    "comment_id_a": a.comment_id,
                    "comment_id_b": b.comment_id,
                    "similarity": round(similarity, 4),
                    "same_commenter": same_commenter,
                    "same_video": same_video,
                    "video_id_a": a.video_id,
                    "video_id_b": b.video_id,
                    "sentiment_a": a.sentiment,
                    "sentiment_b": b.sentiment,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "comment_id_a",
                "comment_id_b",
                "similarity",
                "same_commenter",
                "same_video",
                "video_id_a",
                "video_id_b",
                "sentiment_a",
                "sentiment_b",
            ]
        )

    result = pd.DataFrame(records).drop_duplicates(
        ["comment_id_a", "comment_id_b"]
    )

    return result.sort_values(
        "similarity",
        ascending=False,
    )


def commenter_activity_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("commenter_channel_id", dropna=False)
        .agg(
            comment_count=("comment_id", "size"),
            unique_videos=("video_id", "nunique"),
            unique_sentiments=("sentiment", "nunique"),
            first_comment_at=("published_at", "min"),
            last_comment_at=("published_at", "max"),
        )
        .reset_index()
    )

    grouped["commenter_hash"] = grouped["commenter_channel_id"].map(
        safe_hash
    )

    return grouped[
        [
            "commenter_hash",
            "comment_count",
            "unique_videos",
            "unique_sentiments",
            "first_comment_at",
            "last_comment_at",
        ]
    ].sort_values(
        ["comment_count", "unique_videos"],
        ascending=[False, False],
    )


def build_summary(
    df: pd.DataFrame,
    exact_df: pd.DataFrame,
    cross_video_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    threshold: float,
) -> dict:
    return {
        "input_rows": int(len(df)),
        "unique_commenters": int(df["commenter_channel_id"].nunique()),
        "unique_videos": int(df["video_id"].nunique()),
        "eligible_pattern_rows": int(
            df["eligible_for_pattern_analysis"].sum()
        ),
        "exact_repetition_groups": int(len(exact_df)),
        "cross_video_repetition_groups": int(len(cross_video_df)),
        "temporal_pattern_groups_10min": int(len(temporal_df)),
        "similar_text_pairs_above_threshold": int(len(similarity_df)),
        "similarity_threshold": threshold,
        "time_bucket": TIME_BUCKET,
        "interpretation_boundary": (
            "Outputs are descriptive coordination-like pattern signals. "
            "They do not establish that an account is a bot, buzzer, "
            "or coordinated actor."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descriptive coordination-pattern analysis for MBG YouTube comments."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input coordination-ready CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for exported analysis tables.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.88,
        help="Cosine similarity threshold for near-duplicate text pairs.",
    )
    args = parser.parse_args()

    if not 0 < args.similarity_threshold <= 1:
        raise ValueError("--similarity-threshold harus di antara 0 dan 1.")

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_file)

    print("=" * 80)
    print("COORDINATION PATTERN ANALYSIS")
    print("=" * 80)
    print(f"Input rows              : {len(df):,}")
    print(f"Unique commenters       : {df['commenter_channel_id'].nunique():,}")
    print(f"Unique videos           : {df['video_id'].nunique():,}")
    print(
        "Eligible pattern rows   : "
        f"{int(df['eligible_for_pattern_analysis'].sum()):,}"
    )

    exact_df = exact_repetition_analysis(df)
    exact_df.to_csv(
        output_dir / "coordination_exact_repetitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cross_video_df = cross_video_repetition_analysis(df)
    cross_video_df.to_csv(
        output_dir / "coordination_cross_video_repetition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    temporal_df = temporal_pattern_analysis(df)
    temporal_df.to_csv(
        output_dir / "coordination_temporal_patterns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    similarity_df = similar_text_pairs(
        df,
        threshold=args.similarity_threshold,
    )
    similarity_df.to_csv(
        output_dir / "coordination_similar_text_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    commenter_df = commenter_activity_summary(df)
    commenter_df.to_csv(
        output_dir / "commenter_activity_summary_anonymized.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary(
        df,
        exact_df,
        cross_video_df,
        temporal_df,
        similarity_df,
        args.similarity_threshold,
    )

    (output_dir / "coordination_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Exact repetition groups       : {len(exact_df):,}")
    print(f"Cross-video repetition groups : {len(cross_video_df):,}")
    print(f"10-min temporal patterns      : {len(temporal_df):,}")
    print(
        "Similar text pairs             : "
        f"{len(similarity_df):,} "
        f"(threshold={args.similarity_threshold:.2f})"
    )
    print(f"Output directory               : {output_dir}")


if __name__ == "__main__":
    main()
