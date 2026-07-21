import os
import tarfile
import urllib.request
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from dataset import PretokenizedNMTDataset
from config import load_config
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
        "der schnelle fuchs macht unglaublichen spass"
    ],
    "en": [
        "hello world", 
        "how are you today", 
        "machine learning is incredibly fun", 
        "the quick brown fox jumps over the lazy dog", 
        "deep neural networks require structural optimization",
        "hello world and machine learning",
        "how are deep neural networks doing today",
        "the quick fox is incredibly fun"
    ]
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
        "the quick fox is incredibly fun"
    ],
    "sv": [
        "hej världen", 
        "hur mår du idag", 
        "maskininlärning är otroligt roligt", 
        "den snabba bruna räven hoppar över den lata hunden", 
        "djupa neurala nätverk kräver strukturell optimering",
        "hej världen och maskininlärning",
        "hur mår de djupa neurala nätverken idag",
        "den snabba räven är otroligt rolig"
    ]
}


def get_split_path(processed_dir, split_type, src_lang, trg_lang):
    """Generates standardized mutual filename: <split>_<src>_<trg>.csv"""
    return os.path.join(processed_dir, f"{split_type}_{src_lang}_{trg_lang}.csv")


def download_and_extract_europarl(raw_dir, lang_pair="de-en"):
    """Generic handler for downloading and extracting Europarl datasets."""
    l1, l2 = lang_pair.split("-")
    f1 = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l1}")
    f2 = os.path.join(raw_dir, f"europarl-v7.{lang_pair}.{l2}")
    
    if os.path.exists(f1) and os.path.exists(f2):
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
        if hasattr(tarfile, 'data_filter'):
            tar.extractall(path=raw_dir, filter='data')
        else:
            tar.extractall(path=raw_dir)
    print("✓ Extraction complete.")
    
    return f1, f2


def preprocess_data(df, src_col="de", trg_col="en", token_type="word", max_word_len=64, max_char_len=256):
    df = df.copy()
    df[src_col] = df[src_col].astype(str).str.strip()
    df[trg_col] = df[trg_col].astype(str).str.strip()
    df = df[(df[src_col] != "") & (df[trg_col] != "")]
    df = df[~df[src_col].str.match(r"^\s*<") & ~df[trg_col].str.match(r"^\s*<")]
    df[src_col] = df[src_col].str.lower()
    df[trg_col] = df[trg_col].str.lower()

    punct_regex = r"([.,!?\"':;)(])"
    df[src_col] = df[src_col].str.replace(punct_regex, r" \1 ", regex=True)
    df[trg_col] = df[trg_col].str.replace(punct_regex, r" \1 ", regex=True)
    df[src_col] = df[src_col].str.replace(r"\s+", " ", regex=True).str.strip()
    df[trg_col] = df[trg_col].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates()

    if token_type == "char":
        df["src_len"] = df[src_col].str.len()
        df["trg_len"] = df[trg_col].str.len()
        df = df[(df["src_len"] <= max_char_len) & (df["trg_len"] <= max_char_len)]
        df = df.drop(columns=["src_len", "trg_len"])
    elif token_type == "both":
        src_w_len = df[src_col].str.split().str.len()
        trg_w_len = df[trg_col].str.split().str.len()
        src_c_len = df[src_col].str.len()
        trg_c_len = df[trg_col].str.len()
        df = df[
            (src_w_len <= max_word_len) & (trg_w_len <= max_word_len) &
            (src_c_len <= max_char_len) & (trg_c_len <= max_char_len)
        ]
    else:  # "word"
        df["src_len"] = df[src_col].str.split().str.len()
        df["trg_len"] = df[trg_col].str.split().str.len()
        df = df[(df["src_len"] <= max_word_len) & (df["trg_len"] <= max_word_len)]
        df = df.drop(columns=["src_len", "trg_len"])

    return df.reset_index(drop=True)


def process_and_save_pair(df, src_lang, trg_lang, processed_dir, test_split, seed, mock=False):
    """Splits data and saves to standard mutual paths: <split>_<src>_<trg>.csv once."""
    if mock:
        train_df = df
        val_df = df.iloc[3:4]
        test_df = df.iloc[4:]
    else:
        train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=seed)
        train_df, val_df = train_test_split(train_val_df, test_size=0.10, random_state=seed)
        print(f"📦 [{src_lang.upper()}-{trg_lang.upper()} SPLITS] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_path = get_split_path(processed_dir, "train", src_lang, trg_lang)
    val_path = get_split_path(processed_dir, "val", src_lang, trg_lang)
    test_path = get_split_path(processed_dir, "test", src_lang, trg_lang)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


def execute_offline_caching(processed_dir, token_type="word"):
    """Pre-tokenizes splits into binary tensor files and computes offline Word2Vec matrices."""
    print("\n" + "─"*75)
    print("⚡ [OFFLINE BINARY CACHING & TENSOR PRE-SERIALIZATION]")
    print("─"*75)

    pairs = [("de", "en"), ("en", "sv")]
    for src, trg in pairs:
        train_csv = get_split_path(processed_dir, "train", src, trg)
        val_csv = get_split_path(processed_dir, "val", src, trg)
        test_csv = get_split_path(processed_dir, "test", src, trg)

        if os.path.exists(train_csv):
            print(f"\n📦 Pre-tokenizing & caching binary tensors for pair: {src.upper()} -> {trg.upper()}")
            train_ds = PretokenizedNMTDataset(train_csv, src_lang=src, trg_lang=trg, token_type=token_type)
            if os.path.exists(val_csv):
                PretokenizedNMTDataset(val_csv, src_lang=src, trg_lang=trg, token_type=token_type,
                                       src_vocab=train_ds.src_vocab, trg_vocab=train_ds.trg_vocab)
            if os.path.exists(test_csv):
                PretokenizedNMTDataset(test_csv, src_lang=src, trg_lang=trg, token_type=token_type,
                                       src_vocab=train_ds.src_vocab, trg_vocab=train_ds.trg_vocab)

            if token_type in ["word", "both"]:
                for dim in [128, 256]:
                    precompute_word2vec_embeddings(train_csv, train_ds.src_vocab, src, emb_dim=dim)
                    precompute_word2vec_embeddings(train_csv, train_ds.trg_vocab, trg, emb_dim=dim)


def main():
    parser = argparse.ArgumentParser(description="NMT Pipeline Preprocessing Stage")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char", "both"])
    args = parser.parse_args()

    config = load_config()
    sample_rate = config.get("data", {}).get("sample_rate", 1.0)
    test_split = config.get("data", {}).get("test_split", 0.1)
    seed = config.get("data", {}).get("seed", 42)
    max_word_len = config.get("data", {}).get("max_word_len", 64)
    max_char_len = config.get("data", {}).get("max_char_len", 256)

    IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/input")
    de_file, en_file, sv_file, en_sv_file = None, None, None, None

    if IS_KAGGLE:
        print("🤖 Kaggle environment detected!")
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        REPO_ROOT = os.path.dirname(SCRIPT_DIR)
        processed_dir = os.path.join(REPO_ROOT, "data", "processed")
        found_files = {}
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                path = os.path.join(root, f)
                f_lower = f.lower()
                if "europarl" in f_lower and "de-en" in f_lower and f_lower.endswith(".de"):
                    found_files["de"] = path
                elif "europarl" in f_lower and "de-en" in f_lower and f_lower.endswith(".en"):
                    found_files["de_en"] = path
                elif "europarl" in f_lower and "sv-en" in f_lower and f_lower.endswith(".sv"):
                    found_files["sv"] = path
                elif "europarl" in f_lower and "sv-en" in f_lower and (f_lower.endswith(".en") or f_lower.endswith(".cn")):
                    found_files["sv_en"] = path
                elif "glove" in f_lower and "300d" in f_lower and f_lower.endswith(".txt"):
                    found_files["glove"] = path

        de_file = found_files.get("de")
        en_file = found_files.get("de_en")
        sv_file = found_files.get("sv")
        en_sv_file = found_files.get("sv_en")
        glove_source = found_files.get("glove")

        if glove_source:
            glove_dest_dir = os.path.dirname(processed_dir)
            os.makedirs(glove_dest_dir, exist_ok=True)
            glove_dest = os.path.join(glove_dest_dir, "glove.6B.300d.txt")
            if not os.path.exists(glove_dest):
                os.symlink(glove_source, glove_dest)
        raw_dir = os.path.dirname(de_file) if de_file else "/kaggle/input/europarl"
    else:
        raw_dir = config["data"]["raw_dir"]
        processed_dir = config["data"]["processed_dir"]
        os.makedirs(raw_dir, exist_ok=True)

    os.makedirs(processed_dir, exist_ok=True)

    # 1. GERMAN - ENGLISH PATHWAY
    if args.mock:
        cleaned_de_df = preprocess_data(
            pd.DataFrame(MOCK_DATA_DE_EN), src_col="de", trg_col="en",
            token_type=args.token_type, max_word_len=max_word_len, max_char_len=max_char_len
        )
    else:
        if not IS_KAGGLE:
            de_file, en_file = download_and_extract_europarl(raw_dir, "de-en")
            download_and_extract_glove(os.path.dirname(raw_dir))

        with open(de_file, "r", encoding="utf-8") as f: de_sentences = f.read().splitlines()
        with open(en_file, "r", encoding="utf-8") as f: en_sentences = f.read().splitlines()
        raw_df = pd.DataFrame({"de": de_sentences, "en": en_sentences})
        
        sampled_df = raw_df.sample(frac=min(1.0, sample_rate), random_state=seed).reset_index(drop=True)
        cleaned_de_df = preprocess_data(
            sampled_df, src_col="de", trg_col="en",
            token_type=args.token_type, max_word_len=max_word_len, max_char_len=max_char_len
        )

    process_and_save_pair(
        cleaned_de_df, src_lang="de", trg_lang="en",
        processed_dir=processed_dir, test_split=test_split, seed=seed,
        mock=args.mock
    )

    # 2. ENGLISH - SWEDISH PATHWAY
    if args.mock:
        cleaned_sv_df = preprocess_data(
            pd.DataFrame(MOCK_DATA_SV_EN), src_col="en", trg_col="sv",
            token_type=args.token_type, max_word_len=max_word_len, max_char_len=max_char_len
        )
    else:
        if not IS_KAGGLE:
            sv_file, en_sv_file = download_and_extract_europarl(raw_dir, "sv-en")

        with open(sv_file, "r", encoding="utf-8") as f: sv_sentences = f.read().splitlines()
        with open(en_sv_file, "r", encoding="utf-8") as f: en_sv_sentences = f.read().splitlines()
        raw_sv_df = pd.DataFrame({"en": en_sv_sentences, "sv": sv_sentences})
        
        sampled_sv_df = raw_sv_df.sample(frac=min(1.0, sample_rate), random_state=seed).reset_index(drop=True)
        cleaned_sv_df = preprocess_data(
            sampled_sv_df, src_col="en", trg_col="sv",
            token_type=args.token_type, max_word_len=max_word_len, max_char_len=max_char_len
        )

    process_and_save_pair(
        cleaned_sv_df, src_lang="en", trg_lang="sv",
        processed_dir=processed_dir, test_split=test_split, seed=seed,
        mock=args.mock
    )

    # 3. OFFLINE BINARY CACHING
    execute_offline_caching(processed_dir, token_type=args.token_type)

    print("\n✓ Dataset preprocessing and binary caching completed successfully.")

if __name__ == "__main__":
    main()