import argparse
import os
import shutil
import tarfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

from config import load_config
from dataset import PretokenizedNMTDataset
from embeddings import download_and_extract_glove, precompute_word2vec_embeddings

MOCK_DATA_DE_EN = {
    "de": [
        "hallo welt",
        "wie geht es dir heute",
        "maschinelles lernen macht unglaublichen spass",
        "der schnelle braune fuchs springt ueber den faulen hund",
        "tiefe neuronale netze erfordern strukturelle optimierung",
        "hallo welt und maschinelles lernen",
        "wie geht es den tiefen neuronalen netzen",
        "der schnelle fuchs macht unglaublichen spass",
    ],
    "en": [
        "hello world",
        "how are you today",
        "machine learning is incredibly fun",
        "the quick brown fox jumps over the lazy dog",
        "deep neural networks require structural optimization",
        "hello world and machine learning",
        "how are deep neural networks doing today",
        "the quick fox is incredibly fun",
    ],
}

MOCK_DATA_SV_EN = {
    "en": [
        "hello world",
        "how are you today",
        "machine learning is incredibly fun",
        "the quick brown fox jumps over the lazy dog",
        "deep neural networks require structural optimization",
        "hello world and machine learning",
        "how are deep neural networks doing today",
        "the quick fox is incredibly fun",
    ],
    "sv": [
        "hej världen",
        "hur mår du idag",
        "maskininlärning är otroligt roligt",
        "den snabba bruna räven hoppar över den lata hunden",
        "djupa neurala nätverk kräver strukturell optimering",
        "hej världen och maskininlärning",
        "hur mår de djupa neurala nätverken idag",
        "den snabba räven är otroligt rolig",
    ],
}


def get_split_path(processed_dir, split_type, src_lang, trg_lang):
    """Generates standardized mutual filename: <split>_<src>_<trg>.csv"""
    return os.path.join(processed_dir, f"{split_type}_{src_lang}_{trg_lang}.csv")


def locate_raw_files(raw_dir, lang_pair="de-en"):
    """Robust multi-tier file locator across data/raw, Kaggle inputs, and custom structures."""
    l1, l2 = lang_pair.split("-")

    # 1. Direct standard expected paths in data/raw/
    f1_std = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l1}")
    f2_std = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l2}")
    if os.path.exists(f1_std) and os.path.exists(f2_std):
        return f1_std, f2_std

    # 2. Search locally inside data/raw/
    if os.path.exists(raw_dir):
        f1_cand, f2_cand = None, None
        for f in os.listdir(raw_dir):
            f_lower = f.lower()
            if lang_pair in f_lower or "europarl" in f_lower:
                if f_lower.endswith(f".{l1}"):
                    f1_cand = os.path.join(raw_dir, f)
                elif f_lower.endswith(f".{l2}"):
                    f2_cand = os.path.join(raw_dir, f)
        if f1_cand and f2_cand:
            return f1_cand, f2_cand

    # 3. Search /kaggle/input directory tree
    if os.path.exists("/kaggle/input"):
        f1_cand, f2_cand = None, None
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                f_lower = f.lower()
                if lang_pair in f_lower or "europarl" in f_lower:
                    if f_lower.endswith(f".{l1}") and (
                        f1_cand is None or lang_pair in f_lower
                    ):
                        f1_cand = os.path.join(root, f)
                    elif f_lower.endswith(f".{l2}") and (
                        f2_cand is None or lang_pair in f_lower
                    ):
                        f2_cand = os.path.join(root, f)
        if f1_cand and f2_cand:
            return f1_cand, f2_cand

    return None, None


def download_and_extract_europarl(raw_dir, lang_pair="de-en"):
    """Generic handler for downloading and extracting Europarl datasets."""
    f1, f2 = locate_raw_files(raw_dir, lang_pair)
    if f1 and f2 and os.path.exists(f1) and os.path.exists(f2):
        print(f"✓ Europarl {lang_pair.upper()} text files already present locally.")
        return f1, f2

    url = f"https://www.statmt.org/europarl/v7/{lang_pair}.tgz"
    tar_path = os.path.join(raw_dir, f"{lang_pair}.tgz")

    if not os.path.exists(tar_path):
        print(f"Downloading Europarl {lang_pair.upper()} dataset...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")

    print(f"Extracting {lang_pair} tarball...")
    with tarfile.open(tar_path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=raw_dir, filter="data")
        else:
            tar.extractall(path=raw_dir)
    print("✓ Extraction complete.")

    l1, l2 = lang_pair.split("-")
    f1 = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l1}")
    f2 = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l2}")
    return f1, f2


def preprocess_data(
    df,
    src_col="de",
    trg_col="en",
    token_type="word",
    max_word_len=64,
    max_char_len=256,
):
    df = df.copy()

    # Use PyArrow string backend when available without dropping back to object dtype
    try:
        df[src_col] = df[src_col].astype("string[pyarrow]")
        df[trg_col] = df[trg_col].astype("string[pyarrow]")
    except Exception:
        df[src_col] = df[src_col].astype(str)
        df[trg_col] = df[trg_col].astype(str)

    df[src_col] = df[src_col].str.strip()
    df[trg_col] = df[trg_col].str.strip()
    
    # Filter empty rows and XML tag markers
    df = df[(df[src_col] != "") & (df[trg_col] != "")]
    df = df[~df[src_col].str.startswith("<") & ~df[trg_col].str.startswith("<")]
    
    df[src_col] = df[src_col].str.lower()
    df[trg_col] = df[trg_col].str.lower()

    # Punctuation isolation regex
    punct_regex = r"([.,!?\"':;)(])"
    df[src_col] = (
        df[src_col]
        .str.replace(punct_regex, r" \1 ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df[trg_col] = (
        df[trg_col]
        .str.replace(punct_regex, r" \1 ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    
    df = df.drop_duplicates()

    # Optimized vectorized word count
    def get_word_len(series):
        return series.str.count(" ") + 1

    if token_type == "char":
        df["src_len"] = df[src_col].str.len()
        df["trg_len"] = df[trg_col].str.len()
        df = df[(df["src_len"] <= max_char_len) & (df["trg_len"] <= max_char_len)]
        df = df.drop(columns=["src_len", "trg_len"])
    elif token_type == "both":
        src_w_len = get_word_len(df[src_col])
        trg_w_len = get_word_len(df[trg_col])
        src_c_len = df[src_col].str.len()
        trg_c_len = df[trg_col].str.len()
        df = df[
            (src_w_len <= max_word_len)
            & (trg_w_len <= max_word_len)
            & (src_c_len <= max_char_len)
            & (trg_c_len <= max_char_len)
        ]
    else:  # "word"
        src_w_len = get_word_len(df[src_col])
        trg_w_len = get_word_len(df[trg_col])
        df = df[(src_w_len <= max_word_len) & (trg_w_len <= max_word_len)]

    return df.reset_index(drop=True)


def process_and_save_pair(
    df, src_lang, trg_lang, processed_dir, test_split, seed, mock=False
):
    """Splits data and saves to standard mutual paths: <split>_<src>_<trg>.csv once."""
    if mock:
        train_df = df
        val_df = df.iloc[3:4]
        test_df = df.iloc[4:]
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=test_split, random_state=seed
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.10, random_state=seed
        )
        print(
            f"📦 [{src_lang.upper()}-{trg_lang.upper()} SPLITS] Train:"
            f" {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
        )

    train_path = get_split_path(processed_dir, "train", src_lang, trg_lang)
    val_path = get_split_path(processed_dir, "val", src_lang, trg_lang)
    test_path = get_split_path(processed_dir, "test", src_lang, trg_lang)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


def execute_offline_caching(processed_dir, token_type="word"):
    """
    Pre-tokenizes CSV splits and caches binary tensors (.pt matrix files)
    and pre-computed Word2Vec embedding weights locally.
    """
    print("\n" + "─" * 75)
    print("⚡ [OFFLINE BINARY CACHING & TENSOR PRE-SERIALIZATION]")
    print("─" * 75)

    pairs = [("de", "en"), ("en", "de"), ("en", "sv")]
    for src, trg in pairs:
        pair_tag = f"{src}_{trg}"
        train_csv = get_split_path(processed_dir, "train", src, trg)
        val_csv = get_split_path(processed_dir, "val", src, trg)
        test_csv = get_split_path(processed_dir, "test", src, trg)

        if os.path.exists(train_csv):
            print(
                "\n📦 Pre-tokenizing & caching binary tensors for pair:"
                f" {src.upper()} -> {trg.upper()}"
            )
            train_ds = PretokenizedNMTDataset(
                train_csv, src_lang=src, trg_lang=trg, token_type=token_type
            )
            if os.path.exists(val_csv):
                PretokenizedNMTDataset(
                    val_csv,
                    src_lang=src,
                    trg_lang=trg,
                    token_type=token_type,
                    src_vocab=train_ds.src_vocab,
                    trg_vocab=train_ds.trg_vocab,
                )
            if os.path.exists(test_csv):
                PretokenizedNMTDataset(
                    test_csv,
                    src_lang=src,
                    trg_lang=trg,
                    token_type=token_type,
                    src_vocab=train_ds.src_vocab,
                    trg_vocab=train_ds.trg_vocab,
                )

            if token_type in ["word", "both"]:
                for dim in [128, 256]:
                    precompute_word2vec_embeddings(
                        train_csv, train_ds.src_vocab, src, emb_dim=dim, pair_prefix=pair_tag
                    )
                    precompute_word2vec_embeddings(
                        train_csv, train_ds.trg_vocab, trg, emb_dim=dim, pair_prefix=pair_tag
                    )


def main():
    parser = argparse.ArgumentParser(
        description="NMT Pipeline Preprocessing Stage"
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--token_type",
        type=str,
        default="word",
        choices=["word", "char", "both"],
    )
    args = parser.parse_args()

    config = load_config()
    sample_rate = config.get("data", {}).get("sample_rate", 1.0)
    test_split = config.get("data", {}).get("test_split", 0.1)
    seed = config.get("data", {}).get("seed", 42)
    max_word_len = config.get("data", {}).get("max_word_len", 64)
    max_char_len = config.get("data", {}).get("max_char_len", 256)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    raw_dir = os.path.join(REPO_ROOT, "data", "raw")
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if "glove" in f.lower() and "300d" in f.lower() and f.endswith(".txt"):
                    glove_dest = os.path.join(REPO_ROOT, "data", "glove.6B.300d.txt")
                    if not os.path.exists(glove_dest):
                        try:
                            os.symlink(os.path.join(root, f), glove_dest)
                        except Exception:
                            shutil.copy(os.path.join(root, f), glove_dest)
                    break

    # 1. GERMAN - ENGLISH PATHWAY
    if args.mock:
        cleaned_de_df = preprocess_data(
            pd.DataFrame(MOCK_DATA_DE_EN),
            src_col="de",
            trg_col="en",
            token_type=args.token_type,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )
    else:
        de_file, en_file = locate_raw_files(raw_dir, "de-en")
        if not de_file or not en_file:
            de_file, en_file = download_and_extract_europarl(raw_dir, "de-en")

        download_and_extract_glove(os.path.dirname(raw_dir))

        with open(de_file, "r", encoding="utf-8") as f:
            de_sentences = f.read().splitlines()
        with open(en_file, "r", encoding="utf-8") as f:
            en_sentences = f.read().splitlines()
        raw_df = pd.DataFrame({"de": de_sentences, "en": en_sentences})

        sampled_df = raw_df.sample(
            frac=min(1.0, sample_rate), random_state=seed
        ).reset_index(drop=True)
        cleaned_de_df = preprocess_data(
            sampled_df,
            src_col="de",
            trg_col="en",
            token_type=args.token_type,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )

    process_and_save_pair(
        cleaned_de_df,
        src_lang="de",
        trg_lang="en",
        processed_dir=processed_dir,
        test_split=test_split,
        seed=seed,
        mock=args.mock,
    )
    process_and_save_pair(
        cleaned_de_df,
        src_lang="en",
        trg_lang="de",
        processed_dir=processed_dir,
        test_split=test_split,
        seed=seed,
        mock=args.mock,
    )

    # 2. ENGLISH - SWEDISH PATHWAY
    if args.mock:
        cleaned_sv_df = preprocess_data(
            pd.DataFrame(MOCK_DATA_SV_EN),
            src_col="en",
            trg_col="sv",
            token_type=args.token_type,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )
    else:
        sv_file, en_sv_file = locate_raw_files(raw_dir, "sv-en")
        if not sv_file or not en_sv_file:
            sv_file, en_sv_file = download_and_extract_europarl(raw_dir, "sv-en")

        with open(sv_file, "r", encoding="utf-8") as f:
            sv_sentences = f.read().splitlines()
        with open(en_sv_file, "r", encoding="utf-8") as f:
            en_sv_sentences = f.read().splitlines()
        raw_sv_df = pd.DataFrame({"en": en_sv_sentences, "sv": sv_sentences})

        sampled_sv_df = raw_sv_df.sample(
            frac=min(1.0, sample_rate), random_state=seed
        ).reset_index(drop=True)
        cleaned_sv_df = preprocess_data(
            sampled_sv_df,
            src_col="en",
            trg_col="sv",
            token_type=args.token_type,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )

    process_and_save_pair(
        cleaned_sv_df,
        src_lang="en",
        trg_lang="sv",
        processed_dir=processed_dir,
        test_split=test_split,
        seed=seed,
        mock=args.mock,
    )

    # 3. OFFLINE BINARY CACHING
    execute_offline_caching(processed_dir, token_type=args.token_type)

    print("\n✓ Dataset preprocessing and binary caching completed successfully.")


if __name__ == "__main__":
    main()