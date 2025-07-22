#!/bin/bash -e

cd $(dirname $0)

if which python3 >/dev/null 2>&1; then
	PYTHON=python3
else
	PYTHON=python
fi

[ -e data/fineweb10B/fineweb_val_000000.bin ] || $PYTHON data/cached_fineweb10B.py 8

torchrun --standalone --nproc_per_node=8 ./train_gpt_v0.py
