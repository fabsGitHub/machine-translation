from concurrent.futures import ProcessPoolExecutor
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def pad_vocab_size(size, multiple=16):
    return ((size + multiple - 1) // multiple) * multiple


def _build_vocab_worker(chunk, token_type):
    unique_tokens = set()
    if token_type == "char":
        for sentence in chunk:
            unique_tokens.update(str(sentence).strip())
    else:
        for sentence in chunk:
            unique_tokens.update(str(sentence).strip().split())
    return unique_tokens


def _numericalize_chunk_worker(
    chunk_src,
    chunk_trg,
    token_type,
    src_stoi,
    trg_stoi,
    src_max_idx,
    trg_max_idx,
):
    results = []
    src_get = src_stoi.get
    trg_get = trg_stoi.get
    is_char = token_type == "char"

    for src_text, trg_text in zip(chunk_src, chunk_trg):
        s_str = str(src_text).strip()
        t_str = str(trg_text).strip()

        src_tokens = list(s_str) if is_char else s_str.split()
        trg_tokens = list(t_str) if is_char else t_str.split()

        src_idx = [SOS_IDX]
        for t in src_tokens:
            idx = src_get(t, UNK_IDX)
            if not (0 <= idx <= src_max_idx):
                idx = UNK_IDX
            src_idx.append(idx)
        src_idx.append(EOS_IDX)

        trg_idx = [SOS_IDX]
        for t in trg_tokens:
            idx = trg_get(t, UNK_IDX)
            if not (0 <= idx <= trg_max_idx):
                idx = UNK_IDX
            trg_idx.append(idx)
        trg_idx.append(EOS_IDX)

        results.append((
            np.array(src_idx, dtype=np.int64),
            np.array(trg_idx, dtype=np.int64),
        ))
    return results


class Vocabulary:

    def __init__(self, token_type="word", pad_multiple=16):
        self.token_type = token_type
        self.pad_multiple = pad_multiple
        self.itos = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
            SOS_IDX: SOS_TOKEN,
            EOS_IDX: EOS_TOKEN,
        }
        self.stoi = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
        }

    def __len__(self):
        return len(self.itos)

    @property
    def padded_size(self):
        return pad_vocab_size(len(self.itos), multiple=self.pad_multiple)

    def tokenize(self, text):
        text = str(text).strip()
        if self.token_type == "char":
            return list(text)
        return text.split()

    def build_vocab(self, sentence_list):
        num_sentences = len(sentence_list)
        if num_sentences >= 10000:
            num_workers = min(32, os.cpu_count() or 16)
            chunk_size = (num_sentences + num_workers - 1) // num_workers
            chunks = [
                sentence_list[i : i + chunk_size]
                for i in range(0, num_sentences, chunk_size)
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _build_vocab_worker, chunk, self.token_type
                    )
                    for chunk in chunks
                ]
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
            if not (0 <= idx <= max_valid_idx):
                idx = UNK_IDX
            indices.append(idx)

        return indices


class PretokenizedNMTDataset(Dataset):

    def __init__(
        self,
        csv_path,
        src_lang="de",
        trg_lang="en",
        token_type="word",
        src_vocab=None,
        trg_vocab=None,
        mock_mode=False,
    ):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.token_type = token_type

        cache_dir = os.path.join(os.path.dirname(csv_path), ".matrix_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_name = os.path.basename(csv_path).replace(".csv", "")
        cache_path = os.path.join(
            cache_dir, f"matrix_{base_name}_{token_type}.pt"
        )

        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=False)
            self.src_data = cached["src_data"]
            self.trg_data = cached["trg_data"]
            self.src_offsets = cached["src_offsets"]
            self.trg_offsets = cached["trg_offsets"]
            self.src_lengths = cached["src_lengths"]
            self.trg_lengths = cached["trg_lengths"]
            self.src_vocab = (
                src_vocab if src_vocab is not None else cached["src_vocab"]
            )
            self.trg_vocab = (
                trg_vocab if trg_vocab is not None else cached["trg_vocab"]
            )
            return

        df = pd.read_csv(csv_path)
        src_texts = df[src_lang].astype(str).tolist()
        trg_texts = df[trg_lang].astype(str).tolist()
        del df

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

        if num_samples >= 5000:
            num_workers = min(32, os.cpu_count() or 16)
            chunk_size = (num_samples + num_workers - 1) // num_workers

            src_chunks = [
                src_texts[i : i + chunk_size]
                for i in range(0, num_samples, chunk_size)
            ]
            trg_chunks = [
                trg_texts[i : i + chunk_size]
                for i in range(0, num_samples, chunk_size)
            ]

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
                src_num = np.array(
                    [SOS_IDX] + self.src_vocab.numericalize(src) + [EOS_IDX],
                    dtype=np.int64,
                )
                trg_num = np.array(
                    [SOS_IDX] + self.trg_vocab.numericalize(trg) + [EOS_IDX],
                    dtype=np.int64,
                )
                raw_data.append((src_num, trg_num))

        src_arrays = [pair[0] for pair in raw_data]
        trg_arrays = [pair[1] for pair in raw_data]

        self.src_lengths = np.array(
            [len(s) for s in src_arrays], dtype=np.int32
        )
        self.trg_lengths = np.array(
            [len(t) for t in trg_arrays], dtype=np.int32
        )

        self.src_offsets = np.zeros(num_samples + 1, dtype=np.int64)
        self.trg_offsets = np.zeros(num_samples + 1, dtype=np.int64)

        np.cumsum(self.src_lengths, out=self.src_offsets[1:])
        np.cumsum(self.trg_lengths, out=self.trg_offsets[1:])

        self.src_data = (
            np.concatenate(src_arrays)
            if src_arrays
            else np.array([], dtype=np.int64)
        )
        self.trg_data = (
            np.concatenate(trg_arrays)
            if trg_arrays
            else np.array([], dtype=np.int64)
        )

        del raw_data, src_arrays, trg_arrays

        torch.save({
            "src_data": self.src_data,
            "trg_data": self.trg_data,
            "src_offsets": self.src_offsets,
            "trg_offsets": self.trg_offsets,
            "src_lengths": self.src_lengths,
            "trg_lengths": self.trg_lengths,
            "src_vocab": self.src_vocab,
            "trg_vocab": self.trg_vocab,
        }, cache_path)
        print(f"⚡ Binary matrix cache saved -> {cache_path}")

    def __len__(self):
        return len(self.src_offsets) - 1

    def __getitem__(self, idx):
        s_start, s_end = self.src_offsets[idx], self.src_offsets[idx + 1]
        t_start, t_end = self.trg_offsets[idx], self.trg_offsets[idx + 1]

        src_arr = self.src_data[s_start:s_end]
        trg_arr = self.trg_data[t_start:t_end]

        return torch.from_numpy(src_arr), torch.from_numpy(trg_arr)


class BucketBatchSampler(Sampler):

    def __init__(self, dataset, batch_size, shuffle=True, mega_batch_mult=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mega_batch_size = batch_size * mega_batch_mult

        if hasattr(self.dataset, "src_lengths"):
            self.lengths = self.dataset.src_lengths
        elif hasattr(self.dataset, "data"):
            self.lengths = np.array(
                [len(item[0]) for item in self.dataset.data], dtype=np.int32
            )
        else:
            self.lengths = np.array(
                [len(self.dataset[i][0]) for i in range(len(self.dataset))],
                dtype=np.int32,
            )

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        batches = []
        lengths = self.lengths
        for i in range(0, len(indices), self.mega_batch_size):
            mega_batch = indices[i : i + self.mega_batch_size]
            sorted_order = mega_batch[np.argsort(lengths[mega_batch])]
            for j in range(0, len(sorted_order), self.batch_size):
                batch = sorted_order[j : j + self.batch_size]
                if len(batch) == self.batch_size or not self.shuffle:
                    batches.append(batch)

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
    return src_padded, trg_padded


def get_dataloader(
    csv_path,
    batch_size=128,
    shuffle=True,
    src_vocab=None,
    trg_vocab=None,
    src_lang="de",
    trg_lang="en",
    token_type="word",
):
    dataset = PretokenizedNMTDataset(
        csv_path=csv_path,
        src_lang=src_lang,
        trg_lang=trg_lang,
        token_type=token_type,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
    )

    if shuffle:
        sampler = BucketBatchSampler(
            dataset, batch_size=batch_size, shuffle=True
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    return loader, dataset.src_vocab, dataset.trg_vocab