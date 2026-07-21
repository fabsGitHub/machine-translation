import os
import random
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

class Vocabulary:
    def __init__(self, token_type="word"):
        self.token_type = token_type
        self.itos = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN, SOS_IDX: SOS_TOKEN, EOS_IDX: EOS_TOKEN}
        self.stoi = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX, SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX}
        
    def __len__(self):
        return len(self.itos)

    def get_padded_size(self, multiple=16):
        """Returns vocabulary size padded to nearest multiple for Tensor Core optimization."""
        return pad_vocab_size(len(self.itos), multiple=multiple)

    def tokenize(self, text):
        text = str(text).strip()
        if self.token_type == "char":
            return list(text)
        return text.split()

    def build_vocab(self, sentence_list):
        for sentence in sentence_list:
            for token in self.tokenize(sentence):
                if token not in self.stoi:
                    idx = len(self.itos)
                    self.stoi[token] = idx
                    self.itos[idx] = token

    def numericalize(self, text):
        tokenized = self.tokenize(text)
        return [self.stoi.get(token, UNK_IDX) for token in tokenized]

class PretokenizedNMTDataset(Dataset):
    def __init__(self, csv_path, src_lang="de", trg_lang="en", token_type="word", src_vocab=None, trg_vocab=None):
        self.df = pd.read_csv(csv_path)
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.token_type = token_type
        
        src_texts = self.df[src_lang].astype(str).tolist()
        trg_texts = self.df[trg_lang].astype(str).tolist()

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

        self.data = []
        for src, trg in zip(src_texts, trg_texts):
            src_num = [SOS_IDX] + self.src_vocab.numericalize(src) + [EOS_IDX]
            trg_num = [SOS_IDX] + self.trg_vocab.numericalize(trg) + [EOS_IDX]
            self.data.append((torch.tensor(src_num, dtype=torch.long), torch.tensor(trg_num, dtype=torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BucketBatchSampler(Sampler):
    """
    Groups sequences of similar lengths together to minimize PAD computation.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Sort indices by source sequence length
        self.indices = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))

    def __iter__(self):
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
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

def get_dataloader(csv_path, batch_size=512, shuffle=True, src_vocab=None, trg_vocab=None,
                   src_lang="de", trg_lang="en", token_type="word"):
    dataset = PretokenizedNMTDataset(
        csv_path, src_lang=src_lang, trg_lang=trg_lang, token_type=token_type,
        src_vocab=src_vocab, trg_vocab=trg_vocab
    )
    
    sampler = BucketBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Highly optimized DataLoader options for AMD EPYC host
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    return loader, dataset.src_vocab, dataset.trg_vocab