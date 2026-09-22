import os
import re
import html
import unicodedata
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "mbg_comments.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLEAN_FILE = os.path.join(OUTPUT_DIR, "mbg_comments_clean.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "mbg_quality_report.csv")


def normalize_text(text: str) -> str:
    """Clean text while preserving sentiment-bearing words."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities: &amp; -> &, etc.
    text = html.unescape(text)

    # Convert line breaks/tabs to spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)

    # Remove HTML tags such as <br>, <p>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Mentions are not useful as lexical features
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag words, remove only the # symbol
    text = re.sub(r"#(\w+)", r"\1", text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase for TF-IDF / classical ML
    text = text.lower()

    # Normalize excessive repeated characters:
    # "baguuuuusss" -> "baguus"
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_comment(text: str) -> bool:
    """Reject only empty / meaningless rows. Do not remove short sentiment words."""
    if not text:
        return False

    # At least one alphanumeric character
    return bool(re.search(r"[a-zA-Z0-9]", text))


def suspected_spam(text: str) -> bool:
    """
    Flag suspicious rows for review; does NOT automatically delete them.
    """
    if not text:
        return True

    # Very high symbol ratio
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True

    alnum = sum(c.isalnum() for c in chars)
    symbol_ratio = 1 - (alnum / len(chars))

    # Repeated same token many times
    words = text.split()
    if len(words) >= 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return True

    return symbol_ratio > 0.85


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}\n"
            "Letakkan mbg_comments.csv di folder yang sama dengan script."
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("DATA QUALITY CHECK + CLEANING")
    print("=" * 80)

    print(f"Input rows      : {len(df):,}")
    print(f"Input columns   : {len(df.columns)}")
    print(f"Columns         : {list(df.columns)}")

    if "comment" not in df.columns:
        raise ValueError("Kolom 'comment' tidak ditemukan.")

    # -----------------------------
    # QUALITY REPORT
    # -----------------------------
    report_rows = []

    report_rows.append({
        "metric": "row_count",
        "value": len(df)
    })

    report_rows.append({
        "metric": "column_count",
        "value": len(df.columns)
    })

    report_rows.append({
        "metric": "missing_comment",
        "value": int(df["comment"].isna().sum())
    })

    report_rows.append({
        "metric": "empty_comment",
        "value": int(
            df["comment"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
    })

    if "comment_id" in df.columns:
        report_rows.append({
            "metric": "duplicate_comment_id",
            "value": int(df["comment_id"].duplicated().sum())
        })
    else:
        report_rows.append({
            "metric": "comment_id_column_present",
            "value": 0
        })

    report_rows.append({
        "metric": "duplicate_exact_comment_text",
        "value": int(df["comment"].duplicated().sum())
    })

    report_rows.append({
        "metric": "unique_comments",
        "value": int(df["comment"].nunique(dropna=True))
    })

    # Missing values for all columns
    for col in df.columns:
        report_rows.append({
            "metric": f"missing_{col}",
            "value": int(df[col].isna().sum())
        })

    quality_report = pd.DataFrame(report_rows)
    quality_report.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------
    # CLEANING
    # -----------------------------
    df["comment_original"] = df["comment"]

    df["comment_clean"] = (
        df["comment"]
        .apply(normalize_text)
    )

    df["is_valid_comment"] = (
        df["comment_clean"]
        .apply(is_valid_comment)
    )

    df["suspected_spam"] = (
        df["comment_clean"]
        .apply(suspected_spam)
    )

    # -----------------------------
    # DEDUPLICATION
    # -----------------------------
    # Best practice:
    # - If comment_id exists, deduplicate by comment_id.
    # - Do NOT deduplicate purely by text because two different
    #   users may post exactly the same words.
    before = len(df)

    if "comment_id" in df.columns:
        df = df.drop_duplicates(
            subset=["comment_id"],
            keep="first"
        )

    after_id_dedup = len(df)

    # Remove only invalid/empty comments.
    df = df[df["is_valid_comment"]].copy()

    after_valid_filter = len(df)

    df = df.reset_index(drop=True)
    df.insert(0, "row_id", df.index + 1)

    # Save
    df.to_csv(
        CLEAN_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("CLEANING RESULT")
    print("=" * 80)
    print(f"Rows before cleaning          : {before:,}")
    print(f"After comment_id dedup        : {after_id_dedup:,}")
    print(f"After invalid/empty removal   : {after_valid_filter:,}")
    print(f"Suspected spam (flag only)    : {int(df['suspected_spam'].sum()):,}")
    print(f"\nClean file                    : {CLEAN_FILE}")
    print(f"Quality report                : {REPORT_FILE}")

    print("\nPreview:")
    preview_cols = [
        c for c in [
            "row_id",
            "comment_original",
            "comment_clean",
            "suspected_spam"
        ] if c in df.columns
    ]
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
