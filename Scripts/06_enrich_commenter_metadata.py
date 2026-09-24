from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "mbg_comments_labeled.csv"
CACHE_FILE = DATA_DIR / "commenter_metadata_cache.csv"
OUTPUT_FILE = DATA_DIR / "mbg_comments_coordination_ready.csv"

API_URL = "https://www.googleapis.com/youtube/v3/comments"
DEFAULT_BATCH_SIZE = 25
DEFAULT_TIMEOUT = 30
DEFAULT_SLEEP_SECONDS = 0.05


def load_api_key() -> str:
    """Load YOUTUBE_API_KEY from environment or project .env without requiring python-dotenv."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if api_key:
        return api_key.strip()

    env_candidates = [
        PROJECT_ROOT / ".env",
        Path(__file__).resolve().parent / ".env",
    ]

    for env_file in env_candidates:
        if not env_file.exists():
            continue

        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key == "YOUTUBE_API_KEY" and value:
                return value

    raise RuntimeError(
        "YOUTUBE_API_KEY tidak ditemukan. "
        "Pastikan API key tersedia sebagai environment variable "
        "atau di file .env pada project root."
    )


def chunked(values: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def request_comment_metadata(comment_ids: List[str], api_key: str, timeout: int) -> Dict[str, dict]:
    """Fetch commenter metadata for a batch of comment IDs."""
    params = {
        "part": "id,snippet",
        "id": ",".join(comment_ids),
        "key": api_key,
        "textFormat": "plainText",
    }

    url = f"{API_URL}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": "MBG-YouTube-Coordination-Analysis/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"YouTube API HTTP {exc.code}: {body[:600]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Gagal menghubungi YouTube API: {exc}") from exc

    result: Dict[str, dict] = {}

    for item in payload.get("items", []):
        snippet = item.get("snippet", {}) or {}
        author_channel = snippet.get("authorChannelId") or {}

        result[item.get("id", "")] = {
            "commenter_channel_id": author_channel.get("value", ""),
            "commenter_name": snippet.get("authorDisplayName", ""),
            "parent_comment_id": snippet.get("parentId", ""),
        }

    return result


def fetch_with_fallback(
    comment_ids: List[str],
    api_key: str,
    timeout: int,
) -> tuple[Dict[str, dict], List[str]]:
    """
    Fetch a batch. If the batch fails (for example because one ID is
    unavailable), retry each ID individually so one inaccessible comment
    does not block the remaining IDs.
    """
    try:
        return request_comment_metadata(comment_ids, api_key, timeout), []
    except RuntimeError as batch_error:
        print(
            f"Batch {len(comment_ids)} gagal; fallback ke individual request. "
            f"Alasan: {batch_error}"
        )

        found: Dict[str, dict] = {}
        failed: List[str] = []

        for comment_id in comment_ids:
            try:
                found.update(
                    request_comment_metadata([comment_id], api_key, timeout)
                )
            except RuntimeError as exc:
                print(f"  Gagal mengambil {comment_id}: {exc}")
                failed.append(comment_id)

        return found, failed


def load_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame(
            columns=[
                "comment_id",
                "commenter_channel_id",
                "commenter_name",
                "parent_comment_id",
                "metadata_status",
            ]
        )

    cache = pd.read_csv(CACHE_FILE, dtype=str).fillna("")

    required = {
        "comment_id",
        "commenter_channel_id",
        "commenter_name",
        "parent_comment_id",
        "metadata_status",
    }

    missing = required - set(cache.columns)
    if missing:
        raise ValueError(
            f"Cache metadata tidak lengkap. Kolom yang hilang: {sorted(missing)}"
        )

    return cache


def save_cache(cache: pd.DataFrame) -> None:
    cache = cache.sort_values("comment_id").drop_duplicates(
        "comment_id", keep="last"
    )
    cache.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich MBG YouTube comments with commenter metadata."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Jumlah comment yang diproses untuk smoke test. Default: semua.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Jumlah comment ID per request. Default: 25.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout per request dalam detik.",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input tidak ditemukan: {INPUT_FILE}"
        )

    if not 1 <= args.batch_size <= 50:
        raise ValueError("--batch-size harus berada di antara 1 dan 50.")

    api_key = load_api_key()

    df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

    required_columns = {
        "comment_id",
        "video_id",
        "comment",
        "published_at",
        "sentiment",
    }
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di labeled dataset: {sorted(missing)}"
        )

    if df["comment_id"].duplicated().any():
        raise ValueError("comment_id pada dataset labeled tidak unik.")

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit harus > 0.")
        work_df = df.head(args.limit).copy()
    else:
        work_df = df.copy()

    comment_ids = work_df["comment_id"].astype(str).tolist()

    cache = load_cache()
    cache_lookup = {
        str(row["comment_id"]): row.to_dict()
        for _, row in cache.iterrows()
    }

    missing_ids = [
        comment_id
        for comment_id in comment_ids
        if comment_id not in cache_lookup
    ]

    print("=" * 80)
    print("YOUTUBE COMMENTER METADATA ENRICHMENT")
    print("=" * 80)
    print(f"Input rows             : {len(work_df):,}")
    print(f"Cached metadata        : {len(cache_lookup):,}")
    print(f"IDs needing API fetch  : {len(missing_ids):,}")
    print(f"Batch size             : {args.batch_size}")

    fetched: Dict[str, dict] = {}
    failed_ids: List[str] = []

    for batch_no, batch in enumerate(
        chunked(missing_ids, args.batch_size), start=1
    ):
        print(
            f"Request {batch_no}: "
            f"{len(batch)} comment IDs"
        )

        batch_found, batch_failed = fetch_with_fallback(
            batch,
            api_key,
            args.timeout,
        )

        for comment_id, metadata in batch_found.items():
            status = (
                "ok"
                if metadata.get("commenter_channel_id")
                or metadata.get("commenter_name")
                or metadata.get("parent_comment_id")
                else "metadata_unavailable"
            )

            cache_lookup[comment_id] = {
                "comment_id": comment_id,
                **metadata,
                "metadata_status": status,
            }

        failed_ids.extend(batch_failed)

        if DEFAULT_SLEEP_SECONDS:
            time.sleep(DEFAULT_SLEEP_SECONDS)

        # Persist after every batch so an interrupted run can resume.
        cache_df = pd.DataFrame(
            list(cache_lookup.values()),
            columns=[
                "comment_id",
                "commenter_channel_id",
                "commenter_name",
                "parent_comment_id",
                "metadata_status",
            ],
        )
        save_cache(cache_df)

    cache_df = pd.DataFrame(
        [cache_lookup.get(comment_id, {
            "comment_id": comment_id,
            "commenter_channel_id": "",
            "commenter_name": "",
            "parent_comment_id": "",
            "metadata_status": "api_failed",
        }) for comment_id in comment_ids]
    )

    enriched = work_df.merge(
        cache_df,
        on="comment_id",
        how="left",
        validate="one_to_one",
    )

    # Add enrichment columns after the existing sentiment fields.
    existing_columns = [c for c in df.columns if c in enriched.columns]
    enrichment_columns = [
        "commenter_channel_id",
        "commenter_name",
        "parent_comment_id",
        "metadata_status",
    ]

    ordered_columns = existing_columns + [
        c for c in enrichment_columns
        if c in enriched.columns and c not in existing_columns
    ]

    enriched = enriched[ordered_columns]

    if args.limit is not None:
        # Smoke tests do not overwrite the full portfolio output.
        smoke_output = DATA_DIR / "mbg_comments_coordination_smoke_test.csv"
        enriched.to_csv(smoke_output, index=False, encoding="utf-8-sig")
        output_path = smoke_output
    else:
        enriched.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        output_path = OUTPUT_FILE

    channel_found = enriched["commenter_channel_id"].astype(str).str.strip().ne("").sum()
    name_found = enriched["commenter_name"].astype(str).str.strip().ne("").sum()
    reply_count = enriched["parent_comment_id"].astype(str).str.strip().ne("").sum()
    failed_count = enriched["metadata_status"].eq("api_failed").sum()

    print("\n" + "=" * 80)
    print("ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"Rows processed                 : {len(enriched):,}")
    print(f"Commenter channel IDs found   : {channel_found:,}")
    print(f"Commenter names found         : {name_found:,}")
    print(f"Replies detected              : {reply_count:,}")
    print(f"API failed / unavailable      : {failed_count:,}")
    print(f"Cache file                    : {CACHE_FILE}")
    print(f"Output file                   : {output_path}")

    if failed_ids:
        failed_path = DATA_DIR / "commenter_metadata_failed_ids.txt"
        failed_path.write_text(
            "\n".join(failed_ids) + "\n",
            encoding="utf-8",
        )
        print(f"Failed IDs log               : {failed_path}")


if __name__ == "__main__":
    main()
