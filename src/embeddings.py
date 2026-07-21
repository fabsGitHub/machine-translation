import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch

def precompute_word2vec_embeddings(csv_path, vocab, lang_col, emb_dim=256, cache_dir=None, silent=False):
    """
    Offline/Online pre-computation of Gensim Word2Vec weight matrices.
    Utilizes parallelized host worker threads.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(csv_path), ".matrix_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"w2v_{lang_col}_{emb_dim}.pt")
    if os.path.exists(cache_path):
        if not silent:
            print(f"✓ Word2Vec matrix cache present at: {cache_path}")
        return cache_path

    if not silent:
        print(f"⌛ Training Gensim Word2Vec for '{lang_col}' (Emb={emb_dim})...")
    try:
        from gensim.models import Word2Vec
        df = pd.read_csv(csv_path)
        sentences = [vocab.tokenize(str(text)) for text in df[lang_col].tolist()]
        
        w2v_model = Word2Vec(sentences=sentences, vector_size=emb_dim, window=5, min_count=1, workers=16)

        weight_matrix = np.random.normal(scale=0.6, size=(len(vocab), emb_dim))
        for word, idx in vocab.stoi.items():
            if word in w2v_model.wv:
                weight_matrix[idx] = w2v_model.wv[word]

        weight_tensor = torch.tensor(weight_matrix, dtype=torch.float32)
        torch.save(weight_tensor, cache_path)
        if not silent:
            print(f"⚡ Saved Word2Vec binary matrix -> {cache_path}")
        return cache_path
    except Exception as e:
        if not silent:
            print(f"⚠️ Word2Vec generation skipped: {e}")
        return None

def generate_word2vec_embeddings(vocab, csv_path, lang_col, emb_dim, silent=False):
    cache_dir = os.path.join(os.path.dirname(csv_path), ".matrix_cache")
    cache_path = os.path.join(cache_dir, f"w2v_{lang_col}_{emb_dim}.pt")

    if not os.path.exists(cache_path):
        cache_path = precompute_word2vec_embeddings(csv_path, vocab, lang_col, emb_dim, cache_dir, silent=silent)

    if cache_path and os.path.exists(cache_path):
        try:
            if not silent:
                print(f"⚡ Loading pre-computed Word2Vec matrix cache: {cache_path}")
            tensor = torch.load(cache_path, weights_only=False)
            if tensor.shape[0] == len(vocab) and tensor.shape[1] == emb_dim:
                return tensor.share_memory_()
        except Exception as e:
            if not silent:
                print(f"⚠️ Cache read failed ({e}). Defaulting to standard distributions.")

    return None

def load_glove_embeddings_pair(src_vocab, trg_vocab, glove_file_path, emb_dim=300, silent=False):
    """
    Single-pass parser extracting vectors for both source and target vocabularies.
    Prevents reading/parsing the GloVe text file twice and eliminates full dictionary RAM/disk caching.
    """
    if not silent:
        print(f"⌛ Single-pass GloVe extraction for SRC & TRG vocabularies from {glove_file_path}...")
        
    src_matrix = np.random.normal(scale=0.6, size=(len(src_vocab), emb_dim))
    trg_matrix = np.random.normal(scale=0.6, size=(len(trg_vocab), emb_dim))

    if not os.path.exists(glove_file_path):
        if not silent:
            print(f"⚠️ GloVe file missing at {glove_file_path}. Initializing randomly.")
        return (torch.tensor(src_matrix, dtype=torch.float32).share_memory_(),
                torch.tensor(trg_matrix, dtype=torch.float32).share_memory_())

    combined_words = set(src_vocab.stoi.keys()).union(set(trg_vocab.stoi.keys()))
    
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == emb_dim + 1:
                word = parts[0]
                if word in combined_words:
                    vec = np.array(parts[1:], dtype=np.float32)
                    if word in src_vocab.stoi:
                        src_matrix[src_vocab.stoi[word]] = vec
                    if word in trg_vocab.stoi:
                        trg_matrix[trg_vocab.stoi[word]] = vec

    return (torch.tensor(src_matrix, dtype=torch.float32).share_memory_(),
            torch.tensor(trg_matrix, dtype=torch.float32).share_memory_())

def load_glove_embeddings(vocab, glove_file_path, emb_dim=300, silent=False):
    """Single vocabulary GloVe loader without full-dictionary disk caching."""
    if not silent:
        print(f"⌛ Mapping GloVe embeddings from {glove_file_path} to vocabulary...")
    weight_matrix = np.random.normal(scale=0.6, size=(len(vocab), emb_dim))

    if not os.path.exists(glove_file_path):
        if not silent:
            print(f"⚠️ GloVe file missing at {glove_file_path}. Initializing randomly.")
        return torch.tensor(weight_matrix, dtype=torch.float32).share_memory_()

    vocab_words = set(vocab.stoi.keys())
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == emb_dim + 1:
                word = parts[0]
                if word in vocab_words:
                    weight_matrix[vocab.stoi[word]] = np.array(parts[1:], dtype=np.float32)

    return torch.tensor(weight_matrix, dtype=torch.float32).share_memory_()

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