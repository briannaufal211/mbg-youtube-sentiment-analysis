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

# Keep stage-2 review aligned with the same minimum text-quality rule used
# by Scripts/07_coordination_pattern_analysis.py. This prevents short/common
# strings and emoji-only comments from becoming review clusters.
MIN_TEXT_CHARS = 20
MIN_TEXT_WORDS = 4

# Similarity is a candidate-generation signal in stage 1. For stage 2,
# only stronger near-duplicate pairs are used as a review signal.
STRONG_SIMILARITY_THRESHOLD = 0.92


def hash_id(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def make_pattern_key(text: str) -> str:
    return " ".join(str(text).split())


def is_eligible_text(text: str) -> bool:
    clean = make_pattern_key(text)
    return (
        len(clean) >= MIN_TEXT_CHARS
        and len(clean.split()) >= MIN_TEXT_WORDS
    )


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

    df["pattern_text"] = df["comment_clean"].map(make_pattern_key)
    df["eligible_for_pattern_analysis"] = df["pattern_text"].map(
        is_eligible_text
    )

    return df


def _ensure_set_map():
    return {}


def prepare_indicators(
    df: pd.DataFrame,
    analysis_dir: Path,
) -> pd.DataFrame:
    base = df.copy()
    base["commenter_hash"] = base["commenter_channel_id"].map(hash_id)

    # Signal-family participation is tracked by comment ID rather than only
    # by group counts. This avoids treating one comment as "multi-signal"
    # merely because the same text was picked up by several overlapping
    # detection rules.
    repetition_comments = {}
    cross_video_comments = {}
    temporal_comments = {}
    similarity_comments = {}

    # 1) Exact repeated-text signal.
    eligible = base[base["eligible_for_pattern_analysis"]].copy()

    repeated = (
        eligible.groupby("pattern_text", dropna=False)
        .agg(
            repeated_comment_count=("comment_id", "size"),
            distinct_commenters=("commenter_channel_id", "nunique"),
            distinct_videos=("video_id", "nunique"),
        )
        .reset_index()
    )

    repeated = repeated[
        (repeated["repeated_comment_count"] >= 2)
        & (repeated["distinct_commenters"] >= 2)
    ]

    repeated_texts = set(repeated["pattern_text"])

    for row in eligible[eligible["pattern_text"].isin(repeated_texts)].itertuples(
        index=False
    ):
        repetition_comments.setdefault(row.commenter_hash, set()).add(
            str(row.comment_id)
        )

    # 2) Cross-video exact repetition. This is retained as a separate
    # descriptive count, but it is part of the same "repetition" signal family
    # for the stricter multi-signal decision below.
    cross_file = analysis_dir / "coordination_cross_video_repetition.csv"
    cross = (
        pd.read_csv(cross_file, dtype=str).fillna("")
        if cross_file.exists()
        else pd.DataFrame()
    )
    cross_texts = (
        set(cross["pattern_text"])
        if "pattern_text" in cross.columns
        else set()
    )

    cross_rows = eligible[eligible["pattern_text"].isin(cross_texts)]
    for row in cross_rows.itertuples(index=False):
        cross_video_comments.setdefault(row.commenter_hash, set()).add(
            str(row.comment_id)
        )

    # 3) Temporal signal: same eligible text used by multiple commenters in
    # the same 10-minute bucket.
    base["time_bucket"] = base["published_at"].dt.floor("10min")
    eligible = base[base["eligible_for_pattern_analysis"]].copy()

    temporal_group = (
        eligible.groupby(["time_bucket", "pattern_text"])
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

    for row in eligible.itertuples(index=False):
        key = (str(row.time_bucket), row.pattern_text)
        if key in temporal_keys:
            temporal_comments.setdefault(row.commenter_hash, set()).add(
                str(row.comment_id)
            )

    # 4) Strong near-duplicate text signal.
    similar_file = analysis_dir / "coordination_similar_text_pairs.csv"
    similar = (
        pd.read_csv(similar_file, dtype=str).fillna("")
        if similar_file.exists()
        else pd.DataFrame()
    )

    similarity_pair_counts = {}
    similarity_partner_counts = {}

    if not similar.empty:
        if "similarity" in similar.columns:
            similar["similarity"] = pd.to_numeric(
                similar["similarity"], errors="coerce"
            )
        else:
            similar["similarity"] = float("nan")

        comment_lookup = (
            base[["comment_id", "commenter_hash", "video_id"]]
            .astype(str)
            .set_index("comment_id")
            .to_dict("index")
        )

        strong_pairs = similar[
            similar["similarity"].ge(STRONG_SIMILARITY_THRESHOLD)
        ].copy()

        for row in strong_pairs.itertuples(index=False):
            a = str(getattr(row, "comment_id_a", ""))
            b = str(getattr(row, "comment_id_b", ""))

            info_a = comment_lookup.get(a)
            info_b = comment_lookup.get(b)

            if not info_a or not info_b:
                continue

            # Only compare different commenters. For stage 2, cross-video
            # similarity is preferred because same-video similarity can often
            # arise from normal discussion or replies to the same topic.
            if info_a["commenter_hash"] == info_b["commenter_hash"]:
                continue

            if info_a["video_id"] == info_b["video_id"]:
                continue

            ha = info_a["commenter_hash"]
            hb = info_b["commenter_hash"]

            similarity_comments.setdefault(ha, set()).add(a)
            similarity_comments.setdefault(hb, set()).add(b)

            similarity_pair_counts[ha] = similarity_pair_counts.get(ha, 0) + 1
            similarity_pair_counts[hb] = similarity_pair_counts.get(hb, 0) + 1

            similarity_partner_counts.setdefault(ha, set()).add(hb)
            similarity_partner_counts.setdefault(hb, set()).add(ha)

    # 5) Descriptive activity context. These are not suspicious by themselves.
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

    def count_for(mapping, key):
        return len(mapping.get(key, set()))

    def union_for(key):
        return (
            repetition_comments.get(key, set())
            | temporal_comments.get(key, set())
            | similarity_comments.get(key, set())
        )

    activity["eligible_pattern_comments"] = (
        base.groupby("commenter_hash")["eligible_for_pattern_analysis"]
        .sum()
        .astype(int)
        .reindex(activity["commenter_hash"])
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    activity["exact_repetition_comments"] = activity["commenter_hash"].map(
        lambda x: count_for(repetition_comments, x)
    )
    activity["cross_video_repetition_comments"] = activity["commenter_hash"].map(
        lambda x: count_for(cross_video_comments, x)
    )
    activity["temporal_pattern_comments"] = activity["commenter_hash"].map(
        lambda x: count_for(temporal_comments, x)
    )
    activity["strong_similarity_comments"] = activity["commenter_hash"].map(
        lambda x: count_for(similarity_comments, x)
    )

    # Preserve pair/partner counts as descriptive context.
    activity["strong_similarity_pairs"] = (
        activity["commenter_hash"]
        .map(similarity_pair_counts)
        .fillna(0)
        .astype(int)
    )
    activity["strong_similarity_partner_count"] = (
        activity["commenter_hash"]
        .map(lambda x: len(similarity_partner_counts.get(x, set())))
        .astype(int)
    )

    # Exact repetition and cross-video repetition overlap, so they are treated
    # as one signal family for the stricter review candidate logic.
    activity["repetition_signal_observed"] = (
        activity["exact_repetition_comments"] > 0
    )
    activity["temporal_signal_observed"] = (
        activity["temporal_pattern_comments"] > 0
    )
    activity["similarity_signal_observed"] = (
        activity["strong_similarity_comments"] > 0
    )

    activity["signal_families_observed"] = (
        activity["repetition_signal_observed"].astype(int)
        + activity["temporal_signal_observed"].astype(int)
        + activity["similarity_signal_observed"].astype(int)
    )

    activity["pattern_comment_count"] = activity["commenter_hash"].map(
        lambda x: len(union_for(x))
    )

    # A stricter, portfolio-safe review candidate requires:
    # 1) at least two comments from the same commenter,
    # 2) at least two comments participating in detected patterns, and
    # 3) at least two independent signal families.
    #
    # This is still a descriptive screening rule, NOT a buzzer/bot label.
    activity["multi_signal_review_candidate"] = (
        (activity["comment_count"] >= 2)
        & (activity["pattern_comment_count"] >= 2)
        & (activity["signal_families_observed"] >= 2)
    )

    # Backward-compatible descriptive field. It should NOT be used as the
    # final screening count because overlapping rules can fire on one comment.
    activity["pattern_types_observed_legacy"] = (
        (activity["exact_repetition_comments"] > 0).astype(int)
        + (activity["cross_video_repetition_comments"] > 0).astype(int)
        + (activity["strong_similarity_comments"] > 0).astype(int)
        + (activity["temporal_pattern_comments"] > 0).astype(int)
    )

    activity["multiple_pattern_types_observed_legacy"] = (
        activity["pattern_types_observed_legacy"] >= 2
    )

    return activity.sort_values(
        [
            "multi_signal_review_candidate",
            "signal_families_observed",
            "pattern_comment_count",
            "comment_count",
            "strong_similarity_pairs",
        ],
        ascending=[False, False, False, False, False],
    )


def build_pattern_review_table(df: pd.DataFrame) -> pd.DataFrame:
    # IMPORTANT: use the same eligibility filter as stage 1 so short/common
    # strings and emoji-only comments never become review clusters.
    work = df[
        df["eligible_for_pattern_analysis"]
        & df["pattern_text"].ne("")
    ].copy()

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

    # A final review cluster must have:
    # - meaningful text length,
    # - repeated text across commenters, and
    # - either cross-video reuse OR very tight temporal reuse (<= 1 hour).
    # This sharply reduces noise from generic single-word comments.
    groups["pattern_cluster_candidate"] = (
        groups["repeated_text"]
        & groups["cross_commenter"]
        & (
            groups["cross_video"]
            | (groups["duration_hours"] <= 1)
        )
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
        "commenters_with_legacy_multiple_pattern_types": int(
            commenter_df["multiple_pattern_types_observed_legacy"].sum()
        ),
        "commenters_with_two_or_more_signal_families": int(
            (commenter_df["signal_families_observed"] >= 2).sum()
        ),
        "strict_multi_signal_review_candidates": int(
            commenter_df["multi_signal_review_candidate"].sum()
        ),
        "commenters_with_exact_repetition": int(
            (commenter_df["exact_repetition_comments"] > 0).sum()
        ),
        "commenters_with_cross_video_repetition": int(
            (commenter_df["cross_video_repetition_comments"] > 0).sum()
        ),
        "commenters_in_strong_cross_video_similarity": int(
            (commenter_df["strong_similarity_comments"] > 0).sum()
        ),
        "commenters_in_temporal_patterns": int(
            (commenter_df["temporal_pattern_comments"] > 0).sum()
        ),
        "pattern_clusters_for_review": int(
            pattern_df["pattern_cluster_candidate"].sum()
        ),
        "minimum_text_chars": MIN_TEXT_CHARS,
        "minimum_text_words": MIN_TEXT_WORDS,
        "strong_similarity_threshold": STRONG_SIMILARITY_THRESHOLD,
        "interpretation_boundary": (
            "These are descriptive pattern indicators for review. "
            "They do not establish that any commenter is a buzzer, bot, "
            "or coordinated actor, and they are not a supervised classifier."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate stricter coordination-like indicators without "
            "assigning buzzer labels."
        )
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
    parser.add_argument(
        "--strong-similarity-threshold",
        type=float,
        default=STRONG_SIMILARITY_THRESHOLD,
        help="Minimum cosine similarity for stage-2 near-duplicate review.",
    )
    args = parser.parse_args()

    if not 0 < args.strong_similarity_threshold <= 1:
        raise ValueError(
            "--strong-similarity-threshold harus di antara 0 dan 1."
        )

    global STRONG_SIMILARITY_THRESHOLD
    STRONG_SIMILARITY_THRESHOLD = args.strong_similarity_threshold

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
    print("COORDINATION SCREENING - STRICT MULTI-SIGNAL REVIEW")
    print("=" * 80)
    print(f"Unique commenters                     : {len(commenter_df):,}")
    print(
        "Legacy overlapping multi-pattern rows : "
        f"{int(commenter_df['multiple_pattern_types_observed_legacy'].sum()):,}"
    )
    print(
        "Commenters with >=2 signal families   : "
        f"{int((commenter_df['signal_families_observed'] >= 2).sum()):,}"
    )
    print(
        "Strict review candidates              : "
        f"{int(commenter_df['multi_signal_review_candidate'].sum()):,}"
    )
    print(
        "Pattern clusters for review           : "
        f"{int(pattern_df['pattern_cluster_candidate'].sum()):,}"
    )
    print(
        f"Strong similarity threshold           : "
        f"{args.strong_similarity_threshold:.2f}"
    )
    print(f"Indicator table                       : {commenter_file}")
    print(f"Pattern review table                  : {pattern_file}")
    print(f"Summary                                : {summary_file}")

    print("\nInterpretation:")
    print(
        "The strict candidate set is only a descriptive review queue. "
        "It is not a buzzer, bot, or coordination label."
    )


if __name__ == "__main__":
    main()
