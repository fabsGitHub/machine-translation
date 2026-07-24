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


def _load_headerless_vector_dict(filepath, emb_dim=300):
    """Parses classic GloVe .txt format (no leading '<vocab> <dim>' header line,
    unlike word2vec/fastText .vec files), with the same .pt disk cache as
    load_word2vec_keyed_vectors so re-runs don't re-parse the multi-GB file."""
    cache_dir = _get_cache_dir()
    base_name = os.path.basename(filepath).replace(".", "_")
    pt_cache_path = os.path.join(cache_dir, f"cache_{base_name}.pt")

    if os.path.exists(pt_cache_path):
        try:
            return torch.load(pt_cache_path, weights_only=False)
        except Exception:
            pass

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        torch.distributed.barrier()
        if os.path.exists(pt_cache_path):
            return torch.load(pt_cache_path, weights_only=False)

    print(f"📦 Loading pre-trained vectors from {filepath} (Building fast binary cache)...")
    vector_dict = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != emb_dim + 1:
                continue
            vector_dict[parts[0]] = np.array(parts[1:], dtype=np.float32)

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


# Per-language pretrained vector files. English has genuine Word2Vec (GoogleNews,
# 300d) and GloVe (Stanford glove.6B, 300d) releases. German/Swedish have no
# public GloVe release, and no public release of the ORIGINAL GoogleNews-style
# Word2Vec either - so both embedding_source options fall back to the best real
# Word2Vec models available for those languages instead: German from devmount's
# GermanWordEmbeddings (gensim word2vec .bin, 300d, German Wikipedia + news
# corpus, MIT license - https://devmount.github.io/GermanWordEmbeddings/) and
# Swedish from the NLPL Word Vectors Repository (word2vec Continuous Skipgram,
# 100d, Swedish CoNLL17 corpus - http://vectors.nlpl.eu/repository/, model id
# 69). Note the Swedish model is only 100d vs 300d for English/German -
# populate_embedding_matrix() zero-pads it out to whatever emb_dim is requested,
# which is a minor real limitation worth naming in the report, not a bug.
# This means "word2vec" and "glove" are IDENTICAL on the German/Swedish side
# (no separate real GloVe exists there) and only differ on the English side of
# a pair; report this explicitly rather than presenting it as a full ablation on
# the non-English side.
_WORD2VEC_FILES = {
    "en": ("GoogleNews-vectors-negative300.bin", True),
    "de": ("german.word2vec.bin", True),
    "sv": ("swedish.word2vec.bin", True),
}
_GLOVE_FILES = {
    "en": ("glove.6B.300d.txt", "glove_txt"),
    "de": ("german.word2vec.bin", True),
    "sv": ("swedish.word2vec.bin", True),
}


def generate_word2vec_embeddings(
    vocab,
    train_csv=None,
    lang="en",
    emb_dim=300,
    silent=False,
    pair_prefix=None,
    token_type="word",
    data_dir="data",
):
    """Loads pretrained Word2Vec-family embeddings for a given language vocabulary.

    Language-correct: English uses real GoogleNews Word2Vec vectors; German/Swedish
    use fastText Wikipedia vectors (no public German/Swedish Word2Vec release exists
    here) instead of silently reusing the English file.
    """
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping Word2Vec loading.")
        return None

    filename, binary = _WORD2VEC_FILES.get(lang, _WORD2VEC_FILES["en"])
    vec_file = os.path.join(data_dir, filename)

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


def _load_pretrained_vector_dict(lang, source, emb_dim, data_dir, silent):
    """Resolves and loads the language-correct pretrained vector dict for
    embedding_source in {'glove', 'word2vec'}. English uses the real GloVe/
    Word2Vec release; German uses devmount's German Word2Vec, Swedish uses the
    NLPL Swedish Word2Vec (see _WORD2VEC_FILES/_GLOVE_FILES above) since no
    public German/Swedish GloVe release exists - using the English file for
    those languages (the original behavior) would silently score near-zero
    real coverage."""
    files = _GLOVE_FILES if source == "glove" else _WORD2VEC_FILES
    filename, mode = files.get(lang, files["en"])
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        if not silent:
            print(f"⚠️ Pretrained vector file {filepath} unavailable for lang={lang}.")
        return None

    if mode == "glove_txt":
        return _load_headerless_vector_dict(filepath, emb_dim=emb_dim)
    return load_word2vec_keyed_vectors(filepath, binary=bool(mode))


def load_glove_embeddings(
    vocab,
    glove_file_path=None,
    emb_dim=300,
    silent=False,
    token_type="word",
    glove_dir="data",
    lang="en",
):
    """Loads pretrained embeddings for a single vocabulary under the 'glove'
    embedding_source condition, using the language-correct file (see
    _load_pretrained_vector_dict)."""
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping GloVe loading.")
        return None

    if glove_file_path and os.path.exists(glove_file_path) and lang == "en":
        vector_dict = _load_headerless_vector_dict(glove_file_path, emb_dim=emb_dim)
    else:
        vector_dict = _load_pretrained_vector_dict(lang, "glove", emb_dim, glove_dir, silent)

    if vector_dict is None:
        if not silent:
            print("⚠️ GloVe embeddings file unavailable.")
        return None

    try:
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
    """Loads pretrained 'glove'-condition embeddings for a source/target vocab
    pair, resolving each side to its own language-correct file (see
    _load_pretrained_vector_dict) instead of applying one language's vectors
    to both vocabularies."""
    if token_type == "char":
        if not silent:
            print("⚠️ Token level is 'char'. Skipping GloVe loading.")
        return None, None

    try:
        src_dict = _load_pretrained_vector_dict(src_lang, "glove", emb_dim, glove_dir, silent)
        trg_dict = (
            src_dict
            if trg_lang == src_lang
            else _load_pretrained_vector_dict(trg_lang, "glove", emb_dim, glove_dir, silent)
        )

        src_emb = (
            populate_embedding_matrix(src_vocab, src_dict, emb_dim=emb_dim, token_type=token_type)
            if src_dict is not None
            else None
        )
        trg_emb = (
            populate_embedding_matrix(trg_vocab, trg_dict, emb_dim=emb_dim, token_type=token_type)
            if trg_dict is not None
            else None
        )
        return src_emb, trg_emb
    except Exception as e:
        if not silent:
            print(f"⚠️ Failed to load GloVe embeddings: {e}")
        return None, None