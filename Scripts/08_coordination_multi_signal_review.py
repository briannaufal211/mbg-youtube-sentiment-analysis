from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "mbg_comments_coordination_ready.csv"
DEFAULT_ANALYSIS_DIR = DATA_DIR / "coordination_analysis"

REQUIRED_COLUMNS = {
    "comment_id",
    "video_id",
    "comment",
    "comment_clean",
    "published_at",
    "sentiment",
    "commenter_channel_id",
}


def hash_id(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input tidak ditemukan: {path}")

    df = pd.read_csv(path, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    if df["comment_id"].duplicated().any():
        raise ValueError("comment_id harus unik.")

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True,
    )

    if df["published_at"].isna().any():
        raise ValueError("Terdapat published_at yang tidak valid.")

    return df


def prepare_indicators(
    df: pd.DataFrame,
    analysis_dir: Path,
) -> pd.DataFrame:
    base = df.copy()
    base["commenter_hash"] = base["commenter_channel_id"].map(hash_id)

    # 1) Exact repeated-text signal from the level-1 analysis.
    exact_file = analysis_dir / "coordination_exact_repetitions.csv"
    exact = pd.read_csv(exact_file, dtype=str).fillna("") if exact_file.exists() else pd.DataFrame()

    exact_group_count = {}
    if not exact.empty:
        # Re-derive participation from the original text so the result stays
        # aligned with the current input dataset.
        eligible = base.copy()
        eligible["pattern_text"] = eligible["comment_clean"].astype(str).str.split().str.join(" ")
        repeated = eligible.groupby("pattern_text").agg(
            repeated_comment_count=("comment_id", "size"),
            distinct_commenters=("commenter_channel_id", "nunique"),
            distinct_videos=("video_id", "nunique"),
        ).reset_index()

        repeated = repeated[
            (repeated["repeated_comment_count"] >= 2)
            & (
                (repeated["distinct_commenters"] >= 2)
                | (repeated["distinct_videos"] >= 2)
            )
        ]

        repeated_texts = set(repeated["pattern_text"])
        base["pattern_text"] = base["comment_clean"].astype(str).str.split().str.join(" ")
        exact_group_count = (
            base[base["pattern_text"].isin(repeated_texts)]
            .groupby("commenter_hash")["pattern_text"]
            .nunique()
            .to_dict()
        )
    else:
        base["pattern_text"] = base["comment_clean"].astype(str).str.split().str.join(" ")

    # 2) Cross-video repetition signal.
    cross_file = analysis_dir / "coordination_cross_video_repetition.csv"
    cross = pd.read_csv(cross_file, dtype=str).fillna("") if cross_file.exists() else pd.DataFrame()
    cross_texts = set(cross["pattern_text"]) if "pattern_text" in cross.columns else set()

    cross_signal = (
        base[base["pattern_text"].isin(cross_texts)]
        .groupby("commenter_hash")["pattern_text"]
        .nunique()
        .to_dict()
    )

    # 3) Temporal signal: same normalized text in the same 10-minute bucket
    # with multiple commenters.
    base["time_bucket"] = base["published_at"].dt.floor("10min")
    temporal_group = (
        base.groupby(["time_bucket", "pattern_text"])
        .agg(
            comment_count=("comment_id", "size"),
            distinct_commenters=("commenter_channel_id", "nunique"),
        )
        .reset_index()
    )

    temporal_group = temporal_group[
        (temporal_group["comment_count"] >= 2)
        & (temporal_group["distinct_commenters"] >= 2)
    ]

    temporal_keys = set(
        zip(
            temporal_group["time_bucket"].astype(str),
            temporal_group["pattern_text"],
        )
    )

    temporal_counts = {}
    for row in base.itertuples(index=False):
        key = (str(row.time_bucket), row.pattern_text)
        if key in temporal_keys:
            temporal_counts[row.commenter_hash] = (
                temporal_counts.get(row.commenter_hash, 0) + 1
            )

    # 4) Text-similarity signal from level-1 pairs.
    similar_file = analysis_dir / "coordination_similar_text_pairs.csv"
    similar = (
        pd.read_csv(similar_file, dtype=str).fillna("")
        if similar_file.exists()
        else pd.DataFrame()
    )

    similarity_counts = {}
    similarity_partner_counts = {}

    if not similar.empty:
        high_pairs = similar.copy()

        if "similarity" in high_pairs.columns:
            high_pairs["similarity"] = pd.to_numeric(
                high_pairs["similarity"], errors="coerce"
            )

        # Count unique comment IDs and unique paired IDs per commenter.
        comment_to_commenters = {}
        for row in base[["comment_id", "commenter_hash"]].itertuples(index=False):
            comment_to_commenters[str(row.comment_id)] = row.commenter_hash

        for row in high_pairs.itertuples(index=False):
            a = str(getattr(row, "comment_id_a", ""))
            b = str(getattr(row, "comment_id_b", ""))

            ha = comment_to_commenters.get(a)
            hb = comment_to_commenters.get(b)

            if ha and hb and ha != hb:
                similarity_counts[ha] = similarity_counts.get(ha, 0) + 1
                similarity_counts[hb] = similarity_counts.get(hb, 0) + 1

                similarity_partner_counts.setdefault(ha, set()).add(hb)
                similarity_partner_counts.setdefault(hb, set()).add(ha)

    # 5) Basic activity context. These are descriptive, not suspicious by
    # themselves.
    activity = (
        base.groupby("commenter_hash")
        .agg(
            comment_count=("comment_id", "size"),
            unique_videos=("video_id", "nunique"),
            unique_sentiments=("sentiment", "nunique"),
            first_comment_at=("published_at", "min"),
            last_comment_at=("published_at", "max"),
        )
        .reset_index()
    )

    activity["exact_repetition_groups"] = (
        activity["commenter_hash"].map(exact_group_count).fillna(0).astype(int)
    )
    activity["cross_video_repetition_groups"] = (
        activity["commenter_hash"].map(cross_signal).fillna(0).astype(int)
    )
    activity["similar_text_pairs"] = (
        activity["commenter_hash"].map(similarity_counts).fillna(0).astype(int)
    )
    activity["similar_text_partner_count"] = (
        activity["commenter_hash"]
        .map(lambda x: len(similarity_partner_counts.get(x, set())))
        .astype(int)
    )
    activity["temporal_pattern_comments"] = (
        activity["commenter_hash"].map(temporal_counts).fillna(0).astype(int)
    )

    # A descriptive multi-signal count. This is deliberately NOT a buzzer
    # label or risk score. It only counts how many different pattern types
    # are present for a commenter in this dataset.
    activity["pattern_types_observed"] = (
        (activity["exact_repetition_groups"] > 0).astype(int)
        + (activity["cross_video_repetition_groups"] > 0).astype(int)
        + (activity["similar_text_pairs"] > 0).astype(int)
        + (activity["temporal_pattern_comments"] > 0).astype(int)
    )

    activity["multiple_pattern_types_observed"] = (
        activity["pattern_types_observed"] >= 2
    )

    return activity.sort_values(
        ["pattern_types_observed", "similar_text_pairs", "comment_count"],
        ascending=[False, False, False],
    )


def build_pattern_review_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["pattern_text"] = (
        work["comment_clean"].astype(str).str.split().str.join(" ")
    )
    work = work[work["pattern_text"].ne("")].copy()

    groups = (
        work.groupby("pattern_text")
        .agg(
            comment_count=("comment_id", "size"),
            distinct_commenters=("commenter_channel_id", "nunique"),
            distinct_videos=("video_id", "nunique"),
            first_published_at=("published_at", "min"),
            last_published_at=("published_at", "max"),
            sentiment_values=("sentiment", lambda x: "|".join(sorted(set(x)))),
        )
        .reset_index()
    )

    groups["repeated_text"] = groups["comment_count"] >= 2
    groups["cross_commenter"] = groups["distinct_commenters"] >= 2
    groups["cross_video"] = groups["distinct_videos"] >= 2

    groups["duration_hours"] = (
        (
            pd.to_datetime(groups["last_published_at"], utc=True)
            - pd.to_datetime(groups["first_published_at"], utc=True)
        ).dt.total_seconds()
        / 3600
    ).round(2)

    groups["pattern_cluster_candidate"] = (
        groups["repeated_text"]
        & groups["cross_commenter"]
        & (groups["cross_video"] | (groups["duration_hours"] <= 1))
    )

    return groups[
        [
            "pattern_text",
            "comment_count",
            "distinct_commenters",
            "distinct_videos",
            "first_published_at",
            "last_published_at",
            "duration_hours",
            "sentiment_values",
            "repeated_text",
            "cross_commenter",
            "cross_video",
            "pattern_cluster_candidate",
        ]
    ].sort_values(
        [
            "pattern_cluster_candidate",
            "distinct_commenters",
            "distinct_videos",
            "comment_count",
        ],
        ascending=[False, False, False, False],
    )


def build_summary(
    commenter_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
) -> dict:
    return {
        "unique_commenters": int(len(commenter_df)),
        "commenters_with_multiple_pattern_types": int(
            commenter_df["multiple_pattern_types_observed"].sum()
        ),
        "commenters_with_exact_repetition": int(
            (commenter_df["exact_repetition_groups"] > 0).sum()
        ),
        "commenters_with_cross_video_repetition": int(
            (commenter_df["cross_video_repetition_groups"] > 0).sum()
        ),
        "commenters_in_similar_text_pairs": int(
            (commenter_df["similar_text_pairs"] > 0).sum()
        ),
        "commenters_in_temporal_patterns": int(
            (commenter_df["temporal_pattern_comments"] > 0).sum()
        ),
        "pattern_clusters_for_review": int(
            pattern_df["pattern_cluster_candidate"].sum()
        ),
        "interpretation_boundary": (
            "These are descriptive pattern indicators for review. "
            "They do not establish that any commenter is a buzzer, bot, "
            "or coordinated actor, and they are not a supervised classifier."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate coordination-like indicators without assigning buzzer labels."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Coordination-ready CSV.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Directory containing level-1 coordination outputs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    commenter_df = prepare_indicators(df, analysis_dir)
    pattern_df = build_pattern_review_table(df)

    commenter_file = analysis_dir / "commenter_pattern_indicators_anonymized.csv"
    pattern_file = analysis_dir / "coordination_pattern_review_candidates.csv"
    summary_file = analysis_dir / "coordination_screening_summary.json"

    # Do not export commenter names or raw channel IDs in the screening table.
    commenter_df.to_csv(
        commenter_file,
        index=False,
        encoding="utf-8-sig",
    )

    pattern_df.to_csv(
        pattern_file,
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary(commenter_df, pattern_df)
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("COORDINATION SCREENING - MULTI-SIGNAL AGGREGATION")
    print("=" * 80)
    print(f"Unique commenters                     : {len(commenter_df):,}")
    print(
        "Commenters with >=2 pattern types    : "
        f"{int(commenter_df['multiple_pattern_types_observed'].sum()):,}"
    )
    print(
        "Pattern clusters for review           : "
        f"{int(pattern_df['pattern_cluster_candidate'].sum()):,}"
    )
    print(f"Indicator table                       : {commenter_file}")
    print(f"Pattern review table                  : {pattern_file}")
    print(f"Summary                                : {summary_file}")

    print("\nInterpretation:")
    print(
        "Use these outputs to prioritize descriptive review of repeated or "
        "similar commenting patterns. Do not convert the flags into a "
        "definitive buzzer/bot label."
    )


if __name__ == "__main__":
    main()
