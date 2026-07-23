import os
import numpy as np
import torch


def load_pretrained_word_vectors(
    vocab, lang="en", emb_dim=300, glove_dir="data", silent=False
):
    """Loads pre-trained Word2Vec/FastText vectors for EN, DE, SV.

    Checks local volume directory first before attempting remote API downloads.
    """
    local_file_candidates = {
        "en": [
            "GoogleNews-vectors-negative300.bin.gz",
            "GoogleNews-vectors-negative300.bin",
            "cc.en.300.vec",
            "wiki.en.vec",
        ],
        "de": [
            "wiki.de.vec",
            "cc.de.300.vec",
            "fasttext-wiki-news-subwords-300.vec",
            "glove.de.300d.txt",
        ],
        "sv": [
            "wiki.sv.vec",
            "cc.sv.300.vec",
            "fasttext-wiki-news-subwords-300.vec",
            "glove.sv.300d.txt",
        ],
    }

    candidates = local_file_candidates.get(lang, [])
    filepath = None

    # Check for local file existence
    for cand in candidates:
        full_p = os.path.join(glove_dir, cand)
        if os.path.exists(full_p):
            filepath = full_p
            break
        elif os.path.exists(cand):
            filepath = cand
            break

    wv = None

    # Option A: Load from local volume file
    if filepath:
        if not silent:
            print(
                f"📦 [Word2Vec/FastText] Loading local file for '{lang}' from: {filepath}"
            )
        try:
            from gensim.models import KeyedVectors

            is_binary = filepath.endswith(".bin") or filepath.endswith(".bin.gz")
            wv = KeyedVectors.load_word2vec_format(filepath, binary=is_binary)
        except Exception as e:
            if not silent:
                print(
                    f"⚠️ Failed to parse local file with Gensim ({e}). Falling back..."
                )

    # Option B: Download via Gensim API if local file doesn't exist
    if wv is None:
        try:
            import gensim.downloader as api

            model_name_map = {
                "en": "word2vec-google-news-300",
                "de": "fasttext-wiki-news-subwords-300",
                "sv": "fasttext-wiki-news-subwords-300",
            }
            model_name = model_name_map.get(lang, "word2vec-google-news-300")
            if not silent:
                print(
                    f"📦 [Word2Vec] Local file not found. Attempting download via Gensim API ({model_name})..."
                )
            wv = api.load(model_name)
        except Exception as e:
            if not silent:
                print(
                    f"⚠️ Could not download Word2Vec model via Gensim API ({e}). Falling back to randomized initialization."
                )
            return torch.randn(len(vocab), emb_dim) * 0.01

    # Populate embedding matrix
    actual_dim = wv.vector_size
    weights = torch.randn(len(vocab), actual_dim) * 0.01

    found = 0
    stoi_dict = (
        vocab.stoi if hasattr(vocab, "stoi") else getattr(vocab, "word2idx", {})
    )

    for token, idx in stoi_dict.items():
        if token in wv:
            weights[idx] = torch.from_numpy(wv[token].copy())
            found += 1
        elif token.lower() in wv:
            weights[idx] = torch.from_numpy(wv[token.lower()].copy())
            found += 1

    if not silent:
        coverage = (found / max(1, len(vocab))) * 100
        print(
            f"✅ Loaded {found}/{len(vocab)} tokens ({coverage:.1f}%) from pre-trained vectors ({lang})."
        )

    return weights


def load_language_matched_glove(
    vocab, lang="en", emb_dim=300, glove_dir="data", silent=False
):
    """Loads language-matched GloVe / FastText text vector files (.txt / .vec) from disk."""
    lang_file_map = {
        "en": [
            f"glove.6B.{emb_dim}d.txt",
            "glove.6B.300d.txt",
            "glove.en.300d.txt",
            "cc.en.300.vec",
        ],
        "de": [
            "glove.de.300d.txt",
            "wiki.de.vec",
            "cc.de.300.vec",
            "german_glove.txt",
            f"glove.de.{emb_dim}d.txt",
        ],
        "sv": [
            "glove.sv.300d.txt",
            "wiki.sv.vec",
            "cc.sv.300.vec",
            f"glove.sv.{emb_dim}d.txt",
        ],
    }

    candidates = lang_file_map.get(
        lang, [f"glove.{lang}.300d.txt", "glove.6B.300d.txt"]
    )
    filepath = None

    for cand in candidates:
        full_p = os.path.join(glove_dir, cand)
        if os.path.exists(full_p):
            filepath = full_p
            break
        elif os.path.exists(cand):
            filepath = cand
            break

    embeddings_dict = {}
    detected_dim = emb_dim

    if filepath and os.path.exists(filepath):
        if not silent:
            print(
                f"📖 [GloVe/Text] Loading language-matched vectors for '{lang}' from: {filepath}"
            )
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if (
                    len(parts) < 10
                ):  # Skip header lines (e.g. FastText "2000000 300")
                    continue
                word = parts[0]
                vector = np.asarray(parts[1:], dtype="float32")
                embeddings_dict[word] = vector
                detected_dim = len(vector)
    else:
        if not silent:
            print(
                f"⚠️ [GloVe/Text] Language vector file for '{lang}' not found in {candidates}. Falling back to random initialization."
            )

    weights = torch.randn(len(vocab), detected_dim) * 0.01
    found = 0
    stoi_dict = (
        vocab.stoi if hasattr(vocab, "stoi") else getattr(vocab, "word2idx", {})
    )

    for token, idx in stoi_dict.items():
        if token in embeddings_dict:
            weights[idx] = torch.from_numpy(embeddings_dict[token])
            found += 1
        elif token.lower() in embeddings_dict:
            weights[idx] = torch.from_numpy(embeddings_dict[token.lower()])
            found += 1

    if not silent:
        coverage = (found / max(1, len(vocab))) * 100
        print(
            f"✅ [GloVe Language Alignment] Matched {found}/{len(vocab)} tokens ({coverage:.1f}%) for language '{lang}'."
        )

    return weights

import zipfile
import urllib.request

def download_and_extract_glove(glove_dir="data"):
    """Downloads and extracts Stanford GloVe embeddings if not present locally."""
    os.makedirs(glove_dir, exist_ok=True)
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(glove_dir, "glove.6B.zip")

    # Check if extracted files already exist
    existing_files = os.listdir(glove_dir) if os.path.exists(glove_dir) else []
    if not any(f.startswith("glove") and f.endswith(".txt") for f in existing_files):
        if not os.path.exists(zip_path):
            print(f"📥 Downloading GloVe embeddings from {glove_url}...")
            urllib.request.urlretrieve(glove_url, zip_path)
        print("📦 Extracting GloVe embeddings...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(glove_dir)
        print("✅ GloVe extraction complete.")


def generate_word2vec_embeddings(
    vocab, train_csv=None, lang="en", emb_dim=300, silent=False, pair_prefix=None
):
    """Wrapper mapping generate_word2vec_embeddings to load_pretrained_word_vectors."""
    return load_pretrained_word_vectors(
        vocab=vocab, lang=lang, emb_dim=emb_dim, silent=silent
    )


def precompute_word2vec_embeddings(
    vocab=None, train_csv=None, lang="en", emb_dim=300, silent=False, pair_prefix=None, **kwargs
):
    """Pre-computes or caches Word2Vec embeddings for offline processing."""
    if vocab is not None:
        return load_pretrained_word_vectors(
            vocab=vocab, lang=lang, emb_dim=emb_dim, silent=silent
        )
    return None


def load_glove_embeddings_pair(
    src_vocab,
    trg_vocab,
    src_lang="de",
    trg_lang="en",
    emb_dim=300,
    glove_dir="data",
    silent=False,
):
    if not silent:
        print(
            f"🌐 [Embedding Pipeline] Building Language-Matched Matrices: Src={src_lang.upper()} | Trg={trg_lang.upper()}"
        )

    src_weights = load_language_matched_glove(
        src_vocab,
        lang=src_lang,
        emb_dim=emb_dim,
        glove_dir=glove_dir,
        silent=silent,
    )
    trg_weights = load_language_matched_glove(
        trg_vocab,
        lang=trg_lang,
        emb_dim=emb_dim,
        glove_dir=glove_dir,
        silent=silent,
    )

    return src_weights, trg_weights


def generate_word2vec_embeddings(
    vocab,
    train_csv=None,
    lang="en",
    emb_dim=300,
    glove_dir="data",
    silent=False,
    pair_prefix="de_en",
):
    return load_pretrained_word_vectors(
        vocab, lang=lang, emb_dim=emb_dim, glove_dir=glove_dir, silent=silent
    )