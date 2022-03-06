#!/usr/bin/env bash

python run_exp.py -c config/second_stage/statistic_method/01_deep_bimodal_regression_statistic.yaml
python run_exp.py -c config/second_stage/statistic_method/03_bimodal_resnet18_statistic.yaml
python run_exp.py -c config/second_stage/statistic_method/04_crnet_statistic.yaml
python run_exp.py -c config/second_stage/statistic_method/05_persemon_statistic.yaml
python run_exp.py -c config/second_stage/statistic_method/06_interpret_cnn_statistic.yaml
python run_exp.py -c config/second_stage/statistic_method/08_senet_statistic.yaml
python run_exp.py -c config/second_stage/statistic_method/09_hrnet_statistic.yaml
python run_exp.py -c config/second_stage/statistic_method/10_swin_transformer_statistic.yaml
