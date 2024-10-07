#!/usr/bin/env bash

CONFIG=$1
echo "TEST: $CONFIG"
WEIGHT=$2
echo "LOAD: $WEIGHT"

if [ -z "$3" ]
then
    SAVE_OUTPUT=""
else
    SAVE_OUTPUT=$3
fi

python ./tools/run_exp.py -c "$CONFIG" --test_only --set TEST.WEIGHT "$WEIGHT" TEST.SAVE_DATASET_OUTPUT "$SAVE_OUTPUT"
