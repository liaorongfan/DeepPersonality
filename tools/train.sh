#!/usr/bin/env bash

CONFIG=$1
echo "TRAIN FROM CONFIG: $CONFIG"
python ./tools/run_exp.py -c "$CONFIG"