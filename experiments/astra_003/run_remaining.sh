#!/bin/sh
set -eu
export POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=2
venv/bin/python experiments/astra_003/features.py dev > experiments/astra_003/features_dev.log 2>&1
venv/bin/python experiments/astra_003/train.py > experiments/astra_003/train.log 2>&1
