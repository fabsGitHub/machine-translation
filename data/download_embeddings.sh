#!/bin/bash
set -x
cd /home/fabian/machine-translation/data

echo "[START] $(date)"

curl -L -C - -o GoogleNews-vectors-negative300.bin "https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/resolve/main/GoogleNews-vectors-negative300.bin"
echo "[DONE GoogleNews] $(date)"

curl -L -C - -o wiki.de.vec "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec"
echo "[DONE wiki.de] $(date)"

curl -L -C - -o wiki.en.vec "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec"
echo "[DONE wiki.en] $(date)"

curl -L -C - -o wiki.sv.vec "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.sv.vec"
echo "[DONE wiki.sv] $(date)"

echo "[ALL COMPLETE] $(date)"
