import os
import random
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from utils import pad_vocab_size

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def _build_vocab_worker(chunk, token_type):
    """Worker process function to compute unique token sets from text chunks."""
    unique_tokens = set()
    if token_type == "char":
        for sentence in chunk:
            unique_tokens.update(str(sentence).strip())
    else:
        for sentence in chunk:
            unique_tokens.update(str(sentence).strip().split())
    return unique_tokens


def _numericalize_chunk_worker(chunk_src, chunk_trg, token_type, src_stoi, trg_stoi, src_max_idx, trg_max_idx):
    """Worker process function to numericalize source/target text pairs into NumPy arrays."""
    results = []
    src_get = src_stoi.get
    trg_get = trg_stoi.get
    is_char = (token_type == "char")

    for src_text, trg_text in zip(chunk_src, chunk_trg):
        s_str = str(src_text).strip()
        t_str = str(trg_text).strip()

        src_tokens = list(s_str) if is_char else s_str.split()
        trg_tokens = list(t_str) if is_char else t_str.split()

        # Source sequence processing
        src_idx = [SOS_IDX]
        for t in src_tokens:
            idx = src_get(t, UNK_IDX)
            if not (0 <= idx <= src_max_idx):
                idx = UNK_IDX
            src_idx.append(idx)
        src_idx.append(EOS_IDX)

        # Target sequence processing
        trg_idx = [SOS_IDX]
        for t in trg_tokens:
            idx = trg_get(t, UNK_IDX)
            if not (0 <= idx <= trg_max_idx):
                idx = UNK_IDX
            trg_idx.append(idx)
        trg_idx.append(EOS_IDX)

        results.append((
            np.array(src_idx, dtype=np.int64),
            np.array(trg_idx, dtype=np.int64)
        ))
    return results


class Vocabulary:
    def __init__(self, token_type="word", pad_multiple=16):
        self.token_type = token_type
        self.pad_multiple = pad_multiple
        self.itos = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN, SOS_IDX: SOS_TOKEN, EOS_TOKEN: EOS_TOKEN}
        self.stoi = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX, SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX}

    def __len__(self):
        """Returns the actual unpadded vocabulary size."""
        return len(self.itos)

    @property
    def padded_size(self):
        """Returns vocabulary size padded to nearest multiple for Tensor Core optimization."""
        return pad_vocab_size(len(self.itos), multiple=self.pad_multiple)

    def get_padded_size(self, multiple=16):
        """Legacy helper for backward compatibility."""
        return pad_vocab_size(len(self.itos), multiple=multiple)

    def tokenize(self, text):
        text = str(text).strip()
        if self.token_type == "char":
            return list(text)
        return text.split()

    def build_vocab(self, sentence_list):
        num_sentences = len(sentence_list)
        # Multiprocess token extraction across CPU cores for large datasets
        if num_sentences >= 10000:
            num_workers = min(32, os.cpu_count() or 16)
            chunk_size = (num_sentences + num_workers - 1) // num_workers
            chunks = [sentence_list[i:i + chunk_size] for i in range(0, num_sentences, chunk_size)]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_build_vocab_worker, chunk, self.token_type) for chunk in chunks]
                all_unique = set()
                for future in futures:
                    all_unique.update(future.result())

            for token in all_unique:
                if token not in self.stoi:
                    idx = len(self.itos)
                    self.stoi[token] = idx
                    self.itos[idx] = token
        else:
            for sentence in sentence_list:
                for token in self.tokenize(sentence):
                    if token not in self.stoi:
                        idx = len(self.itos)
                        self.stoi[token] = idx
                        self.itos[idx] = token

    def numericalize(self, text):
        tokenized = self.tokenize(text)
        max_valid_idx = len(self.itos) - 1
        indices = []

        for token in tokenized:
            idx = self.stoi.get(token, UNK_IDX)
            # Guard against any index equal to or exceeding vocab boundary
            if not (0 <= idx <= max_valid_idx):
                idx = UNK_IDX
            indices.append(idx)

        return indices


class PretokenizedNMTDataset(Dataset):
    def __init__(self, csv_path, src_lang="de", trg_lang="en", token_type="word", src_vocab=None, trg_vocab=None):
        df = pd.read_csv(csv_path)
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.token_type = token_type

        src_texts = df[src_lang].astype(str).tolist()
        trg_texts = df[trg_lang].astype(str).tolist()
        del df  # Free raw dataframe memory to reduce IPC overhead in worker processes

        if src_vocab is None:
            self.src_vocab = Vocabulary(token_type)
            self.src_vocab.build_vocab(src_texts)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None:
            self.trg_vocab = Vocabulary(token_type)
            self.trg_vocab.build_vocab(trg_texts)
        else:
            self.trg_vocab = trg_vocab

        num_samples = len(src_texts)
        raw_data = []

        # Vectorize numericalization across host CPU cores via ProcessPoolExecutor
        if num_samples >= 5000:
            num_workers = min(32, os.cpu_count() or 16)
            chunk_size = (num_samples + num_workers - 1) // num_workers

            src_chunks = [src_texts[i:i + chunk_size] for i in range(0, num_samples, chunk_size)]
            trg_chunks = [trg_texts[i:i + chunk_size] for i in range(0, num_samples, chunk_size)]

            src_max_idx = len(self.src_vocab.itos) - 1
            trg_max_idx = len(self.trg_vocab.itos) - 1

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _numericalize_chunk_worker,
                        s_chunk,
                        t_chunk,
                        token_type,
                        self.src_vocab.stoi,
                        self.trg_vocab.stoi,
                        src_max_idx,
                        trg_max_idx,
                    )
                    for s_chunk, t_chunk in zip(src_chunks, trg_chunks)
                ]
                for future in futures:
                    raw_data.extend(future.result())
        else:
            for src, trg in zip(src_texts, trg_texts):
                src_num = np.array([SOS_IDX] + self.src_vocab.numericalize(src) + [EOS_IDX], dtype=np.int64)
                trg_num = np.array([SOS_IDX] + self.trg_vocab.numericalize(trg) + [EOS_IDX], dtype=np.int64)
                raw_data.append((src_num, trg_num))

        # Flatten storage into contiguous 1D NumPy arrays to eliminate IPC serialization bloat
        src_arrays = [pair[0] for pair in raw_data]
        trg_arrays = [pair[1] for pair in raw_data]

        self.src_lengths = np.array([len(s) for s in src_arrays], dtype=np.int32)
        self.trg_lengths = np.array([len(t) for t in trg_arrays], dtype=np.int32)

        self.src_offsets = np.zeros(num_samples + 1, dtype=np.int64)
        self.trg_offsets = np.zeros(num_samples + 1, dtype=np.int64)

        np.cumsum(self.src_lengths, out=self.src_offsets[1:])
        np.cumsum(self.trg_lengths, out=self.trg_offsets[1:])

        self.src_data = np.concatenate(src_arrays) if src_arrays else np.array([], dtype=np.int64)
        self.trg_data = np.concatenate(trg_arrays) if trg_arrays else np.array([], dtype=np.int64)

        del raw_data, src_arrays, trg_arrays

    def __len__(self):
        return len(self.src_offsets) - 1

    def __getitem__(self, idx):
        s_start, s_end = self.src_offsets[idx], self.src_offsets[idx + 1]
        t_start, t_end = self.trg_offsets[idx], self.trg_offsets[idx + 1]

        src_arr = self.src_data[s_start:s_end]
        trg_arr = self.trg_data[t_start:t_end]

        return torch.from_numpy(src_arr), torch.from_numpy(trg_arr)


class BucketBatchSampler(Sampler):
    """
    Groups sequences of similar lengths together to minimize PAD computation
    using megabatch bucketing to preserve batch variance and optimize sorting.
    """
    def __init__(self, dataset, batch_size, shuffle=True, mega_batch_mult=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mega_batch_size = batch_size * mega_batch_mult

        # Precompute length cache as contiguous 1D NumPy array for fast megabatch sorting
        if hasattr(self.dataset, 'src_lengths'):
            self.lengths = self.dataset.src_lengths
        elif hasattr(self.dataset, 'data'):
            self.lengths = np.array([len(item[0]) for item in self.dataset.data], dtype=np.int32)
        else:
            self.lengths = np.array([len(self.dataset[i][0]) for i in range(len(self.dataset))], dtype=np.int32)

    def _get_src_len(self, idx):
        return int(self.lengths[idx])

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        batches = []
        lengths = self.lengths
        for i in range(0, len(indices), self.mega_batch_size):
            mega_batch = indices[i:i + self.mega_batch_size]

            # Vectorized argsort replacing Python lambda sort
            sorted_order = np.argsort(lengths[mega_batch])
            sorted_mega = mega_batch[sorted_order]

            for j in range(0, len(sorted_mega), self.batch_size):
                # Retain NumPy array slicing directly without Python list conversion overhead
                batches.append(sorted_mega[j:j + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    src_list, trg_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_list, batch_first=True, padding_value=PAD_IDX)
    
    # Explicitly pin memory in the collate function for asynchronous CUDA transfers
    # if torch.cuda.is_available():
    #     src_padded = src_padded.pin_memory()
    #     trg_padded = trg_padded.pin_memory()
        
    return src_padded, trg_padded


def get_dataloader(csv_path, batch_size=512, shuffle=True, src_vocab=None, trg_vocab=None,
                   src_lang="de", trg_lang="en", token_type="word", num_workers=16):
    dataset = PretokenizedNMTDataset(
        csv_path, src_lang=src_lang, trg_lang=trg_lang, token_type=token_type,
        src_vocab=src_vocab, trg_vocab=trg_vocab
    )

    sampler = BucketBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    use_workers = num_workers > 0

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_workers,
        prefetch_factor=4 if use_workers else None
    )
    return loader, dataset.src_vocab, dataset.trg_vocab