import os
import urllib.request
import zipfile
import numpy as np
import torch


def _get_cache_dir():
    cache_dir = os.path.join("data", ".embeddings_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_and_extract_glove(glove_dir="data", emb_dim=300):
    """Downloads and extracts GloVe vectors if they do not already exist."""
    os.makedirs(glove_dir, exist_ok=True)
    txt_path = os.path.join(glove_dir, f"glove.6B.{emb_dim}d.txt")

    if os.path.exists(txt_path):
        return txt_path

    zip_path = os.path.join(glove_dir, "glove.6B.zip")
    url = "https://nlp.stanford.edu/data/glove.6B.zip"

    if not os.path.exists(zip_path):
        print(f"📥 Downloading GloVe embeddings from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"⚠️ Failed to download GloVe: {e}")
            return None

    print(f"📦 Extracting {zip_path} to {glove_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(glove_dir)
        print("✅ GloVe embeddings extracted successfully.")
    except Exception as e:
        print(f"⚠️ Failed to extract GloVe: {e}")
        return None

    return txt_path


def load_word2vec_keyed_vectors(filepath, binary=False):
    """Loads KeyedVectors using fast binary PyTorch disk caching to eliminate parse overhead."""
    cache_dir = _get_cache_dir()
    base_name = os.path.basename(filepath).replace(".", "_")
    pt_cache_path = os.path.join(cache_dir, f"cache_{base_name}.pt")

    if os.path.exists(pt_cache_path):
        try:
            return torch.load(pt_cache_path, weights_only=False)
        except Exception:
            pass

    # Rank 0 handles initial parsing if distributed
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        torch.distributed.barrier()
        if os.path.exists(pt_cache_path):
            return torch.load(pt_cache_path, weights_only=False)

    from gensim.models import KeyedVectors

    print(
        f"📦 Loading pre-trained vectors from {filepath} (Building fast binary cache)..."
    )
    wv = KeyedVectors.load_word2vec_format(filepath, binary=binary)

    vector_dict = {word: wv[word] for word in wv.key_to_index}
    torch.save(vector_dict, pt_cache_path)
    print(f"⚡ Saved fast binary embedding cache -> {pt_cache_path}")

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        torch.distributed.barrier()

    return vector_dict


def populate_embedding_matrix(vocab, vector_dict, emb_dim=300, token_type="word"):
    """Maps pre-trained vector dictionary to a vocabulary tensor matrix."""
    vocab_size = len(vocab)
    weights = torch.randn(vocab_size, emb_dim) * 0.01

    if token_type == "char":
        print("⚠️ [Word2Vec/GloVe] Token level is 'char'. Pre-trained word vectors are word-level. Using standard initialized embeddings.")
        return weights

    stoi = vocab.stoi if hasattr(vocab, "stoi") else getattr(vocab, "word2idx", {})
    found = 0
    special_tokens = {"<PAD>", "<UNK>", "<SOS>", "<EOS>"}

    for token, idx in stoi.items():
        if token in special_tokens:
            if token == "<PAD>":
                weights[idx] = torch.zeros(emb_dim)
            continue

        clean_token = token.strip(".,!?\"'()[]{}")
        candidates = [
            token,
            clean_token,
            token.lower(),
            clean_token.lower(),
            token.capitalize(),
            clean_token.capitalize(),
        ]

        matched_vec = None
        for cand in candidates:
            if cand in vector_dict:
                matched_vec = vector_dict[cand]
                break

        if matched_vec is not None:
            # Adjust vector length to match requested emb_dim (truncation/padding)
            vec_len = len(matched_vec)
            if vec_len > emb_dim:
                matched_vec = matched_vec[:emb_dim]
            elif vec_len < emb_dim:
                matched_vec = np.pad(matched_vec, (0, emb_dim - vec_len), mode="constant")

            weights[idx] = torch.from_numpy(matched_vec.copy())
            found += 1

    total_eval = max(1, len(stoi) - len(special_tokens))
    coverage = (found / total_eval) * 100.0
    print(f"✅ Loaded {found}/{total_eval} tokens ({coverage:.1f}%) from pre-trained vectors.")
    return weights


def generate_word2vec_embeddings(
    vocab,
    train_csv=None,
    lang="en",
    emb_dim=300,
    silent=False,
    pair_prefix=None,
    token_type="word",
):
    """Generates Word2Vec embeddings for a given language vocabulary."""
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping Word2Vec loading.")
        return None

    if lang == "de":
        vec_file = os.path.join("data", "wiki.de.vec")
        binary = False
    else:
        vec_file = os.path.join("data", "GoogleNews-vectors-negative300.bin")
        binary = True

    if not os.path.exists(vec_file):
        if not silent:
            print(f"⚠️ Vector file {vec_file} not found. Skipping.")
        return None

    try:
        vector_dict = load_word2vec_keyed_vectors(vec_file, binary=binary)
        return populate_embedding_matrix(
            vocab, vector_dict, emb_dim=emb_dim, token_type=token_type
        )
    except Exception as e:
        if not silent:
            print(f"⚠️ Failed to load Word2Vec for {lang}: {e}")
        return None


# Alias expected by preprocess.py
def precompute_word2vec_embeddings(
    vocab,
    train_csv=None,
    lang="en",
    emb_dim=300,
    silent=False,
    pair_prefix=None,
    token_type="word",
):
    return generate_word2vec_embeddings(
        vocab=vocab,
        train_csv=train_csv,
        lang=lang,
        emb_dim=emb_dim,
        silent=silent,
        pair_prefix=pair_prefix,
        token_type=token_type,
    )


def load_glove_embeddings(
    vocab,
    glove_file_path=None,
    emb_dim=300,
    silent=False,
    token_type="word",
    glove_dir="data",
):
    """Loads GloVe embeddings for a single vocabulary."""
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping GloVe loading.")
        return None

    if glove_file_path and os.path.exists(glove_file_path):
        glove_path = glove_file_path
    else:
        glove_path = download_and_extract_glove(glove_dir=glove_dir, emb_dim=emb_dim)

    if not glove_path or not os.path.exists(glove_path):
        if not silent:
            print("⚠️ GloVe embeddings file unavailable.")
        return None

    try:
        vector_dict = load_word2vec_keyed_vectors(glove_path, binary=False)
        return populate_embedding_matrix(
            vocab, vector_dict, emb_dim=emb_dim, token_type=token_type
        )
    except Exception as e:
        if not silent:
            print(f"⚠️ Failed to load GloVe embeddings: {e}")
        return None


def load_glove_embeddings_pair(
    src_vocab,
    trg_vocab,
    src_lang="de",
    trg_lang="en",
    emb_dim=300,
    glove_dir="data",
    silent=False,
    token_type="word",
):
    """Loads GloVe embeddings for source and target vocabulary pair."""
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping GloVe loading.")
        return None, None

    glove_path = download_and_extract_glove(glove_dir=glove_dir, emb_dim=emb_dim)
    if not glove_path or not os.path.exists(glove_path):
        if not silent:
            print("⚠️ GloVe embeddings file unavailable.")
        return None, None

    try:
        vector_dict = load_word2vec_keyed_vectors(glove_path, binary=False)
        src_emb = populate_embedding_matrix(
            src_vocab, vector_dict, emb_dim=emb_dim, token_type=token_type
        )
        trg_emb = populate_embedding_matrix(
            trg_vocab, vector_dict, emb_dim=emb_dim, token_type=token_type
        )
        return src_emb, trg_emb
    except Exception as e:
        if not silent:
            print(f"⚠️ Failed to load GloVe embeddings: {e}")
        return None, None