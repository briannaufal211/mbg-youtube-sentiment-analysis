from pathlib import Path
import re
import html
import unicodedata
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    DATA_DIR / "mbg_comments_raw.csv",
    PROJECT_ROOT / "mbg_comments.csv",
    Path(__file__).resolve().parent / "mbg_comments.csv",
]
INPUT_FILE = next((p for p in INPUT_CANDIDATES if p.exists()), None)

CLEAN_FILE = DATA_DIR / "mbg_comments_clean.csv"
REPORT_FILE = DATA_DIR / "mbg_quality_report.csv"


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def is_valid_comment(text: str) -> bool:
    return bool(text) and bool(re.search(r"[a-zA-Z0-9]", text))


def suspected_spam(text: str) -> bool:
    if not text:
        return True
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    alnum = sum(c.isalnum() for c in chars)
    symbol_ratio = 1 - (alnum / len(chars))
    words = text.split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.25:
        return True
    return symbol_ratio > 0.85


def main():
    if INPUT_FILE is None:
        raise FileNotFoundError(
            "Raw comment file tidak ditemukan. Letakkan mbg_comments_raw.csv di data/ "
            "atau mbg_comments.csv di project root."
        )

    df = pd.read_csv(INPUT_FILE)

    if "comment" not in df.columns:
        raise ValueError("Kolom 'comment' tidak ditemukan.")

    report_rows = [
        {"metric": "row_count", "value": len(df)},
        {"metric": "column_count", "value": len(df.columns)},
        {"metric": "missing_comment", "value": int(df["comment"].isna().sum())},
        {
            "metric": "empty_comment",
            "value": int(df["comment"].fillna("").astype(str).str.strip().eq("").sum()),
        },
        {
            "metric": "duplicate_exact_comment_text",
            "value": int(df["comment"].duplicated().sum()),
        },
        {
            "metric": "unique_comments",
            "value": int(df["comment"].nunique(dropna=True)),
        },
    ]

    for col in df.columns:
        report_rows.append(
            {"metric": f"missing_{col}", "value": int(df[col].isna().sum())}
        )

    pd.DataFrame(report_rows).to_csv(
        REPORT_FILE, index=False, encoding="utf-8-sig"
    )

    df["comment_original"] = df["comment"]
    df["comment_clean"] = df["comment"].apply(normalize_text)
    df["is_valid_comment"] = df["comment_clean"].apply(is_valid_comment)
    df["suspected_spam"] = df["comment_clean"].apply(suspected_spam)

    before = len(df)

    if "comment_id" in df.columns:
        df = df.drop_duplicates(subset=["comment_id"], keep="first")

    df = df[df["is_valid_comment"]].copy().reset_index(drop=True)
    df.insert(0, "row_id", df.index + 1)

    df.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("DATA QUALITY CHECK + CLEANING")
    print("=" * 80)
    print(f"Input rows                : {before:,}")
    print(f"Output clean rows         : {len(df):,}")
    print(f"Exact duplicate texts     : {int(pd.read_csv(INPUT_FILE)['comment'].duplicated().sum()):,}")
    print(f"Suspected spam (flag only): {int(df['suspected_spam'].sum()):,}")
    print(f"Clean file                : {CLEAN_FILE}")
    print(f"Quality report            : {REPORT_FILE}")


if __name__ == "__main__":
    main()
