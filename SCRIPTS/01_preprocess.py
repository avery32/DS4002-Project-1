#!/usr/bin/env python3
"""
01_preprocess.py — Clean CrowdFlower hate/offensive speech dataset (MI3)

Default behavior:
- Filters to rows with exactly 3 annotators AND unanimous agreement.
- Builds label (0=hate, 1=offensive, 2=neither).
- Normalizes text: lowercase, strip leading "RT", replace URLs <url>, mentions <user>,
  normalize emphatic punctuation (!!! -> <exclaim>, ??? -> <question>, ?! -> <mixedpunct>), collapse whitespace.
- Optional: drop exact duplicates.
- Adds label_confidence = max(votes)/count (==1.0 for unanimous).
- Writes CSV with: text,label,label_confidence,hate_speech,offensive_language,neither.

Flexibility:
- --min-confidence X  → keep rows with max(votes)/count >= X (e.g., 0.67 for 2/3); overrides default unanimous-only.
- --keep-all          → keep all rows with count==3 (no agreement filter).

Usage:
python SCRIPTS/01_preprocess.py --input DATA/labeled_data.csv --drop-duplicates --min-confidence 0.67
"""
from __future__ import annotations
import argparse, re, sys, html
from pathlib import Path
import pandas as pd

# Patterns
URL_RE       = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE   = re.compile(r"(?<!\w)@[\w_]+")
RT_PREFIX_RE = re.compile(r"^\s*rt\s*[:\s]+", re.IGNORECASE)
WS_RE        = re.compile(r"\s+")
MIXED_RE     = re.compile(r"(?:(?:\?+!+)|(?:!+\?+))")
EXC_RE       = re.compile(r"!{2,}")
Q_RE         = re.compile(r"\?{2,}")

def normalize_punct(s: str) -> str:
    s = MIXED_RE.sub(" <mixedpunct> ", s)
    s = EXC_RE.sub(" <exclaim> ", s)
    s = Q_RE.sub(" <question> ", s)
    return s

def preprocess_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s).strip()
    s = RT_PREFIX_RE.sub("", s)           # remove leading RT:
    s = URL_RE.sub("<url>", s)            # replace URLs
    s = MENTION_RE.sub("<user>", s)       # replace mentions
    s = normalize_punct(s)                # normalize !!!, ???, ?!
    s = s.lower()
    s = WS_RE.sub(" ", s).strip()
    return s

def derive_class_from_unanimous(row) -> int:
    if row["hate_speech"] == 3:          # unanimous hate
        return 0
    if row["offensive_language"] == 3:   # unanimous offensive
        return 1
    if row["neither"] == 3:              # unanimous neither
        return 2
    return -1  # shouldn't happen if unanimous filter applied

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Preprocess hate/offensive speech dataset.")
    p.add_argument("--input", required=True, help="Raw CSV path (e.g., DATA/labeled_data.csv)")
    p.add_argument("--output", type=str, default="DATA/processed/labeled_data_clean.csv", help="Output CSV path (default: DATA/processed/labeled_data_clean.csv)")
    p.add_argument("--drop-duplicates", action="store_true", help="Drop exact duplicate tweets after cleaning.")
    p.add_argument("--min-confidence", dest="min_confidence", type=float, default=None,
                   help="Keep rows with max(votes)/count >= this threshold (e.g., 0.67 for 2/3). Overrides default unanimous-only.")
    p.add_argument("--keep-all", action="store_true",
                   help="Keep all rows with count==3 (no agreement filter).")
    args = p.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    try:
        df = pd.read_csv(in_path, encoding="utf-8", low_memory=False)
    except Exception as e:
        print(f"[ERROR] reading {in_path}: {e}", file=sys.stderr)
        return 2

    # Normalize common column names (case-insensitive mapping)
    name_map = {c.lower(): c for c in df.columns}
    # Ensure we have a text column
    if "tweet" in name_map:
        df = df.rename(columns={name_map["tweet"]: "text"})
    elif "text" in name_map:
        df = df.rename(columns={name_map["text"]: "text"})
    else:
        print("[ERROR] Missing 'tweet' or 'text' column.", file=sys.stderr)
        return 3

    # Optional unnamed index column
    if "unnamed: 0" in name_map:
        df = df.drop(columns=[name_map["unnamed: 0"]])

    # Required annotation columns
    required_lower = ["count", "hate_speech", "offensive_language", "neither"]
    missing = [col for col in required_lower if col not in {c.lower() for c in df.columns}]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}", file=sys.stderr)
        return 3

    # Harmonize exact column names
    for col in required_lower:
        if col not in df.columns:
            df = df.rename(columns={name_map[col]: col})

    # Compute label_confidence for filtering/reporting
    df["label_confidence"] = df[["hate_speech","offensive_language","neither"]].max(axis=1) / df["count"]

    # Base filter: exactly 3 annotators
    df = df[df["count"] == 3].copy()

    # Decide filter mode
    if args.keep_all:
        print("[INFO] keep-all: retaining all rows with count==3 (no agreement filter).")
        # no additional filtering
    elif args.min_confidence is not None:
        thr = float(args.min_confidence)
        print(f"[INFO] min-confidence: retaining rows with label_confidence >= {thr}")
        df = df[df["label_confidence"] >= thr].copy()
    else:
        print("[INFO] unanimous-only (default): retaining rows with any vote == 3")
        unanimous_mask = (
            (df["hate_speech"] == 3) |
            (df["offensive_language"] == 3) |
            (df["neither"] == 3)
        )
        df = df[unanimous_mask].copy()

    # Build label
    # If any unanimous rows exist: derive their labels; for others (in min-confidence/keep-all), use majority argmax
    if (df[["hate_speech","offensive_language","neither"]] == 3).any(axis=None):
        df["label"] = df.apply(derive_class_from_unanimous, axis=1)
        needs_argmax = df["label"] == -1
        if needs_argmax.any():
            df.loc[needs_argmax, "label"] = df.loc[needs_argmax,
                                                   ["hate_speech","offensive_language","neither"]].idxmax(axis=1).map(
                {"hate_speech":0,"offensive_language":1,"neither":2}
            )
    else:
        df["label"] = df[["hate_speech","offensive_language","neither"]].idxmax(axis=1).map(
            {"hate_speech":0,"offensive_language":1,"neither":2}
        )
    df["label"] = df["label"].astype(int)

    # Clean text
    df["text"] = df["text"].astype(str).map(preprocess_text)
    df = df[df["text"].str.len() > 0].copy()

    # Deduplicate if requested
    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["text"]).copy()
        print(f"[INFO] Dropped {before - len(df)} duplicate rows.")

    # Select/save columns (keep votes for auditability)
    keep = ["text", "label", "label_confidence", "hate_speech", "offensive_language", "neither"]
    df[keep].to_csv(out_path, index=False)

    # Summary
    print(f"[OK] Wrote {len(df):,} rows to {out_path}")
    print("[INFO] Label counts:", df["label"].value_counts().sort_index().to_dict())
    print("[INFO] Mean confidence by label:",
          df.groupby("label")["label_confidence"].mean().round(3).to_dict())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
