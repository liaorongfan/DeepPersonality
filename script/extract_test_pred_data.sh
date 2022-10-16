#!/usr/bin/env bash

CONFIG=$1
echo "TEST: $CONFIG"
WEIGHT=$2
echo "LOAD: $WEIGHT"
OUTPUT=$3
echo "SAVE: pred data to $OUTPUT"

python script/run_exp.py -c "$CONFIG" --test_only --set TEST.WEIGHT "$WEIGHT"  TEST.SAVE_DATASET_OUTPUT "$OUTPUT"
