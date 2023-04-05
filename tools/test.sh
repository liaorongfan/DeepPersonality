#!/usr/bin/env bash

CONFIG=$1
echo "TEST: $CONFIG"
WEIGHT=$2
echo "LOAD: $WEIGHT"

python ./tools/run_exp.py -c "$CONFIG" --test_only --set TEST.WEIGHT "$WEIGHT"