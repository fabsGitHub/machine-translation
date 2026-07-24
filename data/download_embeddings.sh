#!/bin/bash
set -x
cd "$(dirname "$0")"

echo "[START] $(date)"

curl -L -C - -o GoogleNews-vectors-negative300.bin "https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/resolve/main/GoogleNews-vectors-negative300.bin"
echo "[DONE GoogleNews (en, 300d)] $(date)"

# German Word2Vec - devmount's GermanWordEmbeddings, gensim word2vec binary,
# 300d, German Wikipedia + news corpus, MIT license.
# https://devmount.github.io/GermanWordEmbeddings/
curl -L -C - -o german.word2vec.bin "https://cloud.devmount.de/d2bc5672c523b086/german.model"
echo "[DONE German Word2Vec (de, 300d)] $(date)"

# Swedish Word2Vec - NLPL Word Vectors Repository, Continuous Skipgram, 100d,
# Swedish CoNLL17 corpus. Ships as a zip with both model.bin and model.txt -
# only model.bin (binary, faster to load) is kept.
# http://vectors.nlpl.eu/repository/ (model id 69)
if [ ! -f swedish.word2vec.bin ]; then
    curl -L -C - -o swedish_nlpl_69.zip "http://vectors.nlpl.eu/repository/20/69.zip"
    python3 -c "
import zipfile
with zipfile.ZipFile('swedish_nlpl_69.zip') as z:
    z.extract('model.bin', path='.')
import os
os.replace('model.bin', 'swedish.word2vec.bin')
"
    rm -f swedish_nlpl_69.zip
fi
echo "[DONE Swedish Word2Vec (sv, 100d)] $(date)"

echo "[ALL COMPLETE] $(date)"
