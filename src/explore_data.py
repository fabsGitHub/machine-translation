"""
Task 1: Data Exploration.

Extracts corpus statistics and graphs from the raw Europarl DE-EN and EN-SV
parallel corpora, per the assignment's Task 1 requirements: sentence-length
differences between languages, sentence counts, and "more insights" beyond
those two examples. Also compares the full corpus against the actual 10%
training sample (data/processed/train_de_en.csv) to confirm sampling didn't
introduce distributional bias - not required by the assignment, but a
standard, cheap sanity check worth including.

Run from the repo root: .venv/bin/python src/explore_data.py
Outputs land in data/results/eda/ (stats.json, summary.md, and PNG figures).
"""
import os
import json
import codecs

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUT_DIR = os.path.join(ROOT_DIR, "data", "results", "eda")
os.makedirs(OUT_DIR, exist_ok=True)


def read_lines(path, max_lines=None):
    """Reads a raw Europarl text file. Uses codecs per the assignment's own
    hint for handling codec errors in these files."""
    lines = []
    with codecs.open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            lines.append(line.strip())
    return lines


def word_len(s):
    return len(s.split()) if s else 0


def char_len(s):
    return len(s) if s else 0


def is_xml_tag_line(s):
    return s.startswith("<")


def corpus_stats(name, src_lines, trg_lines, src_lang, trg_lang):
    n_total = len(src_lines)
    empty_src = sum(1 for s in src_lines if not s.strip())
    empty_trg = sum(1 for s in trg_lines if not s.strip())
    xml_src = sum(1 for s in src_lines if is_xml_tag_line(s))
    xml_trg = sum(1 for s in trg_lines if is_xml_tag_line(s))

    # Usable = both sides non-empty and neither is an XML tag line (mirrors
    # preprocess.py's actual filtering logic, so these stats explain *why*
    # that filtering matters).
    usable_mask = [
        bool(s.strip()) and bool(t.strip()) and not is_xml_tag_line(s) and not is_xml_tag_line(t)
        for s, t in zip(src_lines, trg_lines)
    ]
    n_usable = sum(usable_mask)

    src_wlen = np.array([word_len(s) for s, keep in zip(src_lines, usable_mask) if keep])
    trg_wlen = np.array([word_len(t) for t, keep in zip(trg_lines, usable_mask) if keep])
    src_clen = np.array([char_len(s) for s, keep in zip(src_lines, usable_mask) if keep])
    trg_clen = np.array([char_len(t) for t, keep in zip(trg_lines, usable_mask) if keep])

    src_vocab = set()
    trg_vocab = set()
    for s, t, keep in zip(src_lines, trg_lines, usable_mask):
        if keep:
            src_vocab.update(s.lower().split())
            trg_vocab.update(t.lower().split())

    length_diff = src_wlen - trg_wlen

    stats = {
        "corpus": name,
        "src_lang": src_lang,
        "trg_lang": trg_lang,
        "n_total_lines": n_total,
        "n_empty_src": empty_src,
        "n_empty_trg": empty_trg,
        "n_xml_tag_lines_src": xml_src,
        "n_xml_tag_lines_trg": xml_trg,
        "n_usable_after_filtering": n_usable,
        "pct_usable": round(100 * n_usable / n_total, 2) if n_total else 0.0,
        f"{src_lang}_vocab_size": len(src_vocab),
        f"{trg_lang}_vocab_size": len(trg_vocab),
        f"{src_lang}_word_len_mean": round(float(src_wlen.mean()), 2),
        f"{src_lang}_word_len_median": float(np.median(src_wlen)),
        f"{src_lang}_word_len_p95": float(np.percentile(src_wlen, 95)),
        f"{src_lang}_word_len_max": int(src_wlen.max()),
        f"{trg_lang}_word_len_mean": round(float(trg_wlen.mean()), 2),
        f"{trg_lang}_word_len_median": float(np.median(trg_wlen)),
        f"{trg_lang}_word_len_p95": float(np.percentile(trg_wlen, 95)),
        f"{trg_lang}_word_len_max": int(trg_wlen.max()),
        "length_diff_mean": round(float(length_diff.mean()), 3),
        "length_diff_std": round(float(length_diff.std()), 3),
        "pearson_corr_src_trg_len": round(float(np.corrcoef(src_wlen, trg_wlen)[0, 1]), 3),
    }
    return stats, src_wlen, trg_wlen, length_diff, src_clen, trg_clen


def plot_length_distributions(src_wlen, trg_wlen, src_lang, trg_lang, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, 101, 2)
    ax.hist(np.clip(src_wlen, 0, 100), bins=bins, alpha=0.6, label=f"{src_lang.upper()}", color="#2E4C8A")
    ax.hist(np.clip(trg_wlen, 0, 100), bins=bins, alpha=0.6, label=f"{trg_lang.upper()}", color="#B8792B")
    ax.set_xlabel("Sentence length (words, clipped at 100)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_length_scatter(src_wlen, trg_wlen, src_lang, trg_lang, out_path, title, sample_n=20000):
    if len(src_wlen) > sample_n:
        idx = np.random.RandomState(42).choice(len(src_wlen), sample_n, replace=False)
        x, y = src_wlen[idx], trg_wlen[idx]
    else:
        x, y = src_wlen, trg_wlen
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(np.clip(x, 0, 100), np.clip(y, 0, 100), gridsize=40, cmap="viridis", mincnt=1)
    ax.plot([0, 100], [0, 100], color="white", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(f"{src_lang.upper()} length (words)")
    ax.set_ylabel(f"{trg_lang.upper()} length (words)")
    ax.set_title(title)
    fig.colorbar(hb, ax=ax, label="sentence pairs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_length_diff(length_diff, src_lang, trg_lang, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.clip(length_diff, -50, 50), bins=np.arange(-50, 51, 2), color="#3A8451", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"Length difference ({src_lang.upper()} words - {trg_lang.upper()} words)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vocab_sizes(vocab_stats, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    langs = list(vocab_stats.keys())
    sizes = list(vocab_stats.values())
    bars = ax.bar(langs, sizes, color=["#2E4C8A", "#B8792B", "#3A8451"][: len(langs)])
    ax.set_ylabel("Vocabulary size (unique lowercased word forms)")
    ax.set_title("Vocabulary size per language")
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{size:,}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_vs_population(pop_wlen, sample_wlen, lang, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, 101, 2)
    ax.hist(np.clip(pop_wlen, 0, 100), bins=bins, density=True, alpha=0.5,
            label="Full corpus", color="#6b675e")
    ax.hist(np.clip(sample_wlen, 0, 100), bins=bins, density=True, alpha=0.5,
            label="10% training sample", color="#2E4C8A")
    ax.set_xlabel(f"{lang.upper()} sentence length (words, clipped at 100)")
    ax.set_ylabel("Density")
    ax.set_title(f"Sample representativeness check ({lang.upper()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 75)
    print("TASK 1: DATA EXPLORATION")
    print("=" * 75)

    all_stats = {}

    # ---- DE-EN corpus ----
    de_path = os.path.join(RAW_DIR, "europarl-v7.de-en.de")
    en_path = os.path.join(RAW_DIR, "europarl-v7.de-en.en")
    if os.path.exists(de_path) and os.path.exists(en_path):
        print(f"\nReading {de_path} / {en_path} ...")
        de_lines = read_lines(de_path)
        en_lines = read_lines(en_path)
        print(f"  {len(de_lines):,} lines each side")

        stats, de_wlen, en_wlen, diff, de_clen, en_clen = corpus_stats(
            "de-en", de_lines, en_lines, "de", "en"
        )
        all_stats["de_en"] = stats

        plot_length_distributions(de_wlen, en_wlen, "de", "en",
                                   os.path.join(OUT_DIR, "de_en_length_hist.png"),
                                   "DE-EN: sentence length distribution")
        plot_length_scatter(de_wlen, en_wlen, "de", "en",
                             os.path.join(OUT_DIR, "de_en_length_scatter.png"),
                             "DE-EN: source vs. target sentence length")
        plot_length_diff(diff, "de", "en",
                          os.path.join(OUT_DIR, "de_en_length_diff.png"),
                          "DE-EN: sentence length difference (DE tends to be longer)")

        print(f"  DE mean length: {stats['de_word_len_mean']} words | EN mean length: {stats['en_word_len_mean']} words")
        print(f"  Mean length difference (DE-EN): {stats['length_diff_mean']} words")
        print(f"  Usable after filtering (non-empty, non-XML both sides): {stats['pct_usable']}%")
    else:
        print(f"\n[skip] DE-EN raw files not found at {de_path}")
        de_lines = en_lines = []

    # ---- EN-SV corpus ----
    sv_en_path = os.path.join(RAW_DIR, "europarl-v7.sv-en.en")
    sv_path = os.path.join(RAW_DIR, "europarl-v7.sv-en.sv")
    if os.path.exists(sv_en_path) and os.path.exists(sv_path):
        print(f"\nReading {sv_en_path} / {sv_path} ...")
        en2_lines = read_lines(sv_en_path)
        sv_lines = read_lines(sv_path)
        print(f"  {len(en2_lines):,} lines each side")

        stats_sv, en2_wlen, sv_wlen, diff_sv, en2_clen, sv_clen = corpus_stats(
            "en-sv", en2_lines, sv_lines, "en", "sv"
        )
        all_stats["en_sv"] = stats_sv

        plot_length_distributions(en2_wlen, sv_wlen, "en", "sv",
                                   os.path.join(OUT_DIR, "en_sv_length_hist.png"),
                                   "EN-SV: sentence length distribution")
        plot_length_scatter(en2_wlen, sv_wlen, "en", "sv",
                             os.path.join(OUT_DIR, "en_sv_length_scatter.png"),
                             "EN-SV: source vs. target sentence length")
        plot_length_diff(diff_sv, "en", "sv",
                          os.path.join(OUT_DIR, "en_sv_length_diff.png"),
                          "EN-SV: sentence length difference")

        print(f"  EN mean length: {stats_sv['en_word_len_mean']} words | SV mean length: {stats_sv['sv_word_len_mean']} words")
    else:
        print(f"\n[skip] EN-SV raw files not found at {sv_en_path}")
        en2_lines = sv_lines = []

    # ---- Vocabulary size comparison across all three languages ----
    if de_lines and en2_lines:
        de_vocab = set(w.lower() for line in de_lines for w in line.split() if not is_xml_tag_line(line))
        en_vocab = set(w.lower() for line in en_lines for w in line.split() if not is_xml_tag_line(line))
        sv_vocab = set(w.lower() for line in sv_lines for w in line.split() if not is_xml_tag_line(line)) if sv_lines else set()
        vocab_sizes = {"DE": len(de_vocab), "EN": len(en_vocab)}
        if sv_vocab:
            vocab_sizes["SV"] = len(sv_vocab)
        plot_vocab_sizes(vocab_sizes, os.path.join(OUT_DIR, "vocab_sizes.png"))
        all_stats["vocab_sizes"] = vocab_sizes
        print(f"\nVocabulary sizes: {vocab_sizes}")

    # ---- Sample representativeness: full corpus vs. actual 10% training sample ----
    train_csv = os.path.join(PROCESSED_DIR, "train_de_en.csv")
    if os.path.exists(train_csv) and de_lines:
        print(f"\nComparing full DE-EN corpus against the actual training sample ({train_csv}) ...")
        sample_df = pd.read_csv(train_csv)
        sample_de_wlen = sample_df["de"].astype(str).apply(word_len).to_numpy()
        sample_en_wlen = sample_df["en"].astype(str).apply(word_len).to_numpy()

        plot_sample_vs_population(de_wlen, sample_de_wlen, "de", os.path.join(OUT_DIR, "sample_check_de.png"))
        plot_sample_vs_population(en_wlen, sample_en_wlen, "en", os.path.join(OUT_DIR, "sample_check_en.png"))

        all_stats["sample_representativeness"] = {
            "train_sample_size": len(sample_df),
            "population_de_mean_len": round(float(de_wlen.mean()), 2),
            "sample_de_mean_len": round(float(sample_de_wlen.mean()), 2),
            "population_en_mean_len": round(float(en_wlen.mean()), 2),
            "sample_en_mean_len": round(float(sample_en_wlen.mean()), 2),
        }
        print(f"  Population DE mean len: {de_wlen.mean():.2f} | Sample DE mean len: {sample_de_wlen.mean():.2f}")
        print(f"  Population EN mean len: {en_wlen.mean():.2f} | Sample EN mean len: {sample_en_wlen.mean():.2f}")
    else:
        print(f"\n[skip] No processed training sample found at {train_csv} yet - run preprocess.py first for this comparison.")

    # ---- Save stats + a markdown summary for direct inclusion in the report ----
    stats_path = os.path.join(OUT_DIR, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    md_path = os.path.join(OUT_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Task 1: Data Exploration Summary\n\n")
        for corpus_key in ("de_en", "en_sv"):
            if corpus_key not in all_stats:
                continue
            s = all_stats[corpus_key]
            f.write(f"## {s['corpus'].upper()} corpus\n\n")
            f.write(f"- Total lines: {s['n_total_lines']:,}\n")
            f.write(f"- Usable after filtering (non-empty, non-XML-tag both sides): "
                    f"{s['n_usable_after_filtering']:,} ({s['pct_usable']}%)\n")
            f.write(f"- {s['src_lang'].upper()} vocabulary size: {s[s['src_lang']+'_vocab_size']:,}\n")
            f.write(f"- {s['trg_lang'].upper()} vocabulary size: {s[s['trg_lang']+'_vocab_size']:,}\n")
            f.write(f"- {s['src_lang'].upper()} mean/median/p95 length (words): "
                    f"{s[s['src_lang']+'_word_len_mean']} / {s[s['src_lang']+'_word_len_median']} / "
                    f"{s[s['src_lang']+'_word_len_p95']}\n")
            f.write(f"- {s['trg_lang'].upper()} mean/median/p95 length (words): "
                    f"{s[s['trg_lang']+'_word_len_mean']} / {s[s['trg_lang']+'_word_len_median']} / "
                    f"{s[s['trg_lang']+'_word_len_p95']}\n")
            f.write(f"- Mean length difference ({s['src_lang'].upper()}-{s['trg_lang'].upper()}): "
                    f"{s['length_diff_mean']} words (std {s['length_diff_std']})\n")
            f.write(f"- Pearson correlation between source/target length: {s['pearson_corr_src_trg_len']}\n\n")
        if "sample_representativeness" in all_stats:
            sr = all_stats["sample_representativeness"]
            f.write("## Sample representativeness (10% training sample vs. full corpus)\n\n")
            f.write(f"- Training sample size: {sr['train_sample_size']:,} sentence pairs\n")
            f.write(f"- DE mean length: population {sr['population_de_mean_len']} vs. sample {sr['sample_de_mean_len']}\n")
            f.write(f"- EN mean length: population {sr['population_en_mean_len']} vs. sample {sr['sample_en_mean_len']}\n")

    print(f"\nSaved stats -> {stats_path}")
    print(f"Saved markdown summary -> {md_path}")
    print(f"Saved figures -> {OUT_DIR}/*.png")
    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()