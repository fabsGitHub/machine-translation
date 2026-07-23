import os
import zipfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch


def _tokenize_sentence_chunk(chunk, token_type):
    """Worker process function to tokenize sentence chunks in parallel."""
    if token_type == "char":
        return [list(str(text).strip()) for text in chunk]
    return [str(text).strip().split() for text in chunk]


def precompute_word2vec_embeddings(csv_path, vocab, lang_col, emb_dim=256, cache_dir=None, silent=False, pair_prefix=""):
    """
    Offline/Online pre-computation of Gensim Word2Vec weight matrices.
    Utilizes parallelized host worker threads and disambiguates cache files using `pair_prefix`.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(csv_path), ".matrix_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Disambiguate cache file by dataset file/pair prefix
    prefix_str = f"{pair_prefix}_" if pair_prefix else ""
    cache_path = os.path.join(cache_dir, f"w2v_{prefix_str}{lang_col}_{emb_dim}.pt")

    if os.path.exists(cache_path):
        if not silent:
            print(f"✓ Word2Vec matrix cache present at: {cache_path}")
        return cache_path

    if not silent:
        print(f"⌛ Training Gensim Word2Vec for '{lang_col}' (Emb={emb_dim})...")
    try:
        from gensim.models import Word2Vec
        df = pd.read_csv(csv_path)
        texts = df[lang_col].astype(str).tolist()

        num_texts = len(texts)
        num_workers = min(32, os.cpu_count() or 16)

        # Parallelize CPU tokenization across host cores for large datasets
        if num_texts >= 5000:
            chunk_size = (num_texts + num_workers - 1) // num_workers
            chunks = [texts[i:i + chunk_size] for i in range(0, num_texts, chunk_size)]
            token_type = getattr(vocab, "token_type", "word")

            sentences = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_tokenize_sentence_chunk, chunk, token_type) for chunk in chunks]
                for future in futures:
                    sentences.extend(future.result())
        else:
            sentences = [vocab.tokenize(text) for text in texts]

        w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=emb_dim,
            window=5,
            min_count=1,
            workers=num_workers
        )

        weight_matrix = np.random.normal(scale=0.6, size=(len(vocab), emb_dim)).astype(np.float32)
        for word, idx in vocab.stoi.items():
            if word in w2v_model.wv:
                weight_matrix[idx] = w2v_model.wv[word]

        weight_tensor = torch.from_numpy(weight_matrix).float()
        torch.save(weight_tensor, cache_path)
        if not silent:
            print(f"⚡ Saved Word2Vec binary matrix -> {cache_path}")
        return cache_path
    except Exception as e:
        if not silent:
            print(f"⚠️ Word2Vec generation skipped: {e}")
        return None


def generate_word2vec_embeddings(vocab, csv_path, lang_col, emb_dim, silent=False, pair_prefix=""):
    """Loads pre-computed Word2Vec embeddings matrix or triggers pre-computation with pair prefix disambiguation."""
    cache_dir = os.path.join(os.path.dirname(csv_path), ".matrix_cache")
    prefix_str = f"{pair_prefix}_" if pair_prefix else ""
    cache_path = os.path.join(cache_dir, f"w2v_{prefix_str}{lang_col}_{emb_dim}.pt")

    if not os.path.exists(cache_path):
        cache_path = precompute_word2vec_embeddings(
            csv_path, vocab, lang_col, emb_dim=emb_dim, cache_dir=cache_dir, silent=silent, pair_prefix=pair_prefix
        )

    if cache_path and os.path.exists(cache_path):
        try:
            if not silent:
                print(f"⚡ Loading pre-computed Word2Vec matrix cache: {cache_path}")
            tensor = torch.load(cache_path, weights_only=False)
            if tensor.shape[0] == len(vocab) and tensor.shape[1] == emb_dim:
                # Pin memory for fast DMA transfers to GPU instead of using share_memory_()
                return tensor.pin_memory()
        except Exception as e:
            if not silent:
                print(f"⚠️ Cache read failed ({e}). Defaulting to standard distributions.")

    return None


def load_glove_embeddings_pair(src_vocab, trg_vocab, glove_file_path, emb_dim=300, silent=False):
    """
    Single-pass C-accelerated parser extracting vectors for both source and target vocabularies.
    Reads GloVe vectors using fast Pandas C-engine routines and vectorized NumPy boolean masking.
    """
    if not silent:
        print(f"⌛ Single-pass GloVe extraction for SRC & TRG vocabularies from {glove_file_path}...")

    src_matrix = np.random.normal(scale=0.6, size=(len(src_vocab), emb_dim)).astype(np.float32)
    trg_matrix = np.random.normal(scale=0.6, size=(len(trg_vocab), emb_dim)).astype(np.float32)

    if not os.path.exists(glove_file_path):
        if not silent:
            print(f"⚠️ GloVe file missing at {glove_file_path}. Initializing randomly.")
        return (torch.from_numpy(src_matrix).pin_memory(),
                torch.from_numpy(trg_matrix).pin_memory())

    src_stoi = src_vocab.stoi
    trg_stoi = trg_vocab.stoi

    # C-engine accelerated CSV parser replacing Python line-by-line loops
    df_glove = pd.read_csv(
        glove_file_path, sep=" ", quoting=3, header=None, engine="c", dtype={0: str}
    )
    words = df_glove[0].values
    vectors = df_glove.iloc[:, 1:].values.astype(np.float32)

    # Vectorized NumPy boolean masking replacing sequential for loops
    src_mask = np.array([w in src_stoi for w in words])
    if np.any(src_mask):
        src_words = words[src_mask]
        src_indices = np.array([src_stoi[w] for w in src_words], dtype=np.int64)
        src_matrix[src_indices] = vectors[src_mask]

    trg_mask = np.array([w in trg_stoi for w in words])
    if np.any(trg_mask):
        trg_words = words[trg_mask]
        trg_indices = np.array([trg_stoi[w] for w in trg_words], dtype=np.int64)
        trg_matrix[trg_indices] = vectors[trg_mask]

    return (torch.from_numpy(src_matrix).pin_memory(),
            torch.from_numpy(trg_matrix).pin_memory())


def load_glove_embeddings(vocab, glove_file_path, emb_dim=300, silent=False):
    """Single vocabulary C-accelerated GloVe loader using Pandas multi-threaded routines and vectorized NumPy boolean masking."""
    if not silent:
        print(f"⌛ Multi-threaded GloVe loading from {glove_file_path}...")
    weight_matrix = np.random.normal(scale=0.6, size=(len(vocab), emb_dim)).astype(np.float32)

    if not os.path.exists(glove_file_path):
        if not silent:
            print(f"⚠️ GloVe file missing at {glove_file_path}. Initializing randomly.")
        return torch.from_numpy(weight_matrix).pin_memory()

    vocab_stoi = vocab.stoi

    # C-engine accelerated CSV parser
    df_glove = pd.read_csv(
        glove_file_path, sep=" ", quoting=3, header=None, engine="c", dtype={0: str}
    )
    words = df_glove[0].values
    vectors = df_glove.iloc[:, 1:].values.astype(np.float32)

    # Vectorized NumPy boolean masking replacing sequential for loops
    mask = np.array([w in vocab_stoi for w in words])
    if np.any(mask):
        matching_words = words[mask]
        vocab_indices = np.array([vocab_stoi[w] for w in matching_words], dtype=np.int64)
        weight_matrix[vocab_indices] = vectors[mask]

    return torch.from_numpy(weight_matrix).pin_memory()


def download_and_extract_glove(data_dir):
    glove_target_file = os.path.join(data_dir, "glove.6B.300d.txt")
    glove_alt_file = os.path.join(data_dir, "raw", "glove.6B.300d.txt")

    if os.path.exists(glove_target_file):
        print("✓ Pre-trained GloVe 300d vectors already present locally in data/.")
        return

    if os.path.exists(glove_alt_file):
        print("✓ Found GloVe 300d vectors in 'raw/' directory. Creating link/copy in data/...")
        try:
            os.symlink(os.path.abspath(glove_alt_file), glove_target_file)
        except Exception:
            import shutil
            shutil.copy(glove_alt_file, glove_target_file)
        return

    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(data_dir, "glove.6B.zip")

    print("\n🌐 GloVe embeddings missing...")
    if not os.path.exists(zip_path):
        print("Downloading GloVe 6B word vectors (approx. 822MB)...")
        urllib.request.urlretrieve(glove_url, zip_path)

    print("⚡ Extracting 'glove.6B.300d.txt' from zip archive...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract("glove.6B.300d.txt", data_dir)

    if os.path.exists(zip_path):
        os.remove(zip_path)