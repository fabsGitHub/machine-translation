"""
Builds a genuine DE -> SV evaluation set for the pivot task (Task 5, +15%).

The project has no direct DE-SV parallel corpus - only DE-EN and EN-SV. But
Europarl is multi-parallel: the same proceedings were transcribed once and
translated into every language, so the English side of the DE-EN corpus and
the English side of the EN-SV corpus overlap heavily (>1.5M exact-matching
lines out of ~1.9M, verified). This script aligns the two corpora by exact
match on their shared English sentence, giving real (not synthetic) DE->SV
reference pairs: for each English sentence that appears in both corpora, the
German translation from one side and the Swedish translation from the other
side are a genuine translation of the same original sentence.

This has to run against the RAW corpora, before preprocess.py's independent
10% sampling of each pair destroys the overlap (two independent random 10%
samples of ~1.9M lines each share almost nothing).

Run from the repo root: .venv/bin/python src/build_pivot_eval_set.py
Output: data/processed/pivot_de_en_sv_eval.csv (columns: de, en, sv)
"""
import os
import sys
import random
import codecs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from preprocess import preprocess_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

EVAL_SET_SIZE = 3000
SEED = 42


def read_lines(path):
    with codecs.open(path, "r", encoding="utf-8", errors="replace") as f:
        return [l.strip() for l in f]


def main():
    print("=" * 75)
    print("Building genuine DE -> SV pivot evaluation set (via shared English side)")
    print("=" * 75)

    de_path = os.path.join(RAW_DIR, "europarl-v7.de-en.de")
    en1_path = os.path.join(RAW_DIR, "europarl-v7.de-en.en")
    en2_path = os.path.join(RAW_DIR, "europarl-v7.sv-en.en")
    sv_path = os.path.join(RAW_DIR, "europarl-v7.sv-en.sv")

    for p in (de_path, en1_path, en2_path, sv_path):
        if not os.path.exists(p):
            print(f"ERROR: required raw file missing: {p}")
            sys.exit(1)

    print("Reading raw corpora...")
    de_lines = read_lines(de_path)
    en1_lines = read_lines(en1_path)
    en2_lines = read_lines(en2_path)
    sv_lines = read_lines(sv_path)
    print(f"  DE-EN: {len(de_lines):,} lines | EN-SV: {len(en2_lines):,} lines")

    # Clean each pair with the exact same rules used for training data
    # (lowercase, strip empty, remove XML-tag lines), so the alignment key
    # (the English text) is cleaned consistently on both sides.
    de_en_df = pd.DataFrame({"de": de_lines, "en": en1_lines})
    de_en_clean = preprocess_data(de_en_df, src_col="de", trg_col="en", token_type="word")

    en_sv_df = pd.DataFrame({"en": en2_lines, "sv": sv_lines})
    en_sv_clean = preprocess_data(en_sv_df, src_col="en", trg_col="sv", token_type="word")

    print(f"  After cleaning: DE-EN {len(de_en_clean):,} rows | EN-SV {len(en_sv_clean):,} rows")

    # Deduplicate by English text (first occurrence wins) to build the join key.
    de_en_map = dict(zip(de_en_clean["en"], de_en_clean["de"]))
    en_sv_map = dict(zip(en_sv_clean["en"], en_sv_clean["sv"]))

    shared_en = list(set(de_en_map.keys()) & set(en_sv_map.keys()))
    print(f"  Shared English sentences (alignment anchors): {len(shared_en):,}")

    random.seed(SEED)
    random.shuffle(shared_en)
    selected_en = shared_en[:EVAL_SET_SIZE]

    rows = [{"de": de_en_map[en], "en": en, "sv": en_sv_map[en]} for en in selected_en]
    out_df = pd.DataFrame(rows)

    out_path = os.path.join(PROCESSED_DIR, "pivot_de_en_sv_eval.csv")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved {len(out_df):,} genuine DE->SV pivot evaluation triples -> {out_path}")
    print("\nSample rows:")
    print(out_df.head(3).to_string())
    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()