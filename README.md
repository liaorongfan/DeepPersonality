# Benchmarking Deep Learning Models for Audio-visual Automatic Self-reported and Apparent Personality Recognition



## Requirements and Dependecies



## Usage for reproducing reproted experiments





## Usage for developing new personality computing models




## Command line interface
* train:
```shell
python run_exp.py \
--cfg_file config/unified_frame_images/03_bimodal_resnet18.yaml 

```

* resume
```shell
python run_exp.py \
-c config/unified_frame_images/03_bimodal_resnet18.yaml \
--resume results/unified_frame_images/03_bimodal_resnet/12-19_18-15/checkpoint_199.pkl \
--max_epoch 210 \
--lr 0.001
```
* test
```shell
python run_exp.py \
-c config/unified_frame_images/09_hrnet.yaml \
--test_only \
--set TEST.WEIGHT results/unified_frame_images/09_hrnet/12-20_22-12/checkpoint_186.pkl

```

