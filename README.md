# DeepPersonality
Personality prediction with neural networks

## Command line interface
* train:
```shell
python run_exp.py \
--cfg_file config/standard/03_bimodal_resnet18.yaml 

```

* resume
```shell
python run_exp.py \
-c config/standard/03_bimodal_resnet18.yaml \
--resume results/standard/03_bimodal_resnet/12-19_18-15/checkpoint_199.pkl \
--max_epoch 210 \
--lr 0.001
```
* test
```shell
python run_exp.py \
-c config/standard/09_hrnet.yaml \
--test_only \
--set TEST.WEIGHT results/standard/09_hrnet/12-20_22-12/checkpoint_186.pkl

```

