
## Command line interface
    usage: run_exp.py [-h] [-c CFG_FILE] [--weight WEIGHT] [--test_only] [--lr LR] [--bs BS] [--resume RESUME]
                  [-m MAX_EPOCH] [--set ...]

    deep learning on personality

    optional arguments:
        -h, --help            show this help message and exit
        -c , --cfg_file       experiment config file
        --resume              saved model path to last training epoch
        --test_only           only test model on specified weights
        --weight              initialize with pretrained model weights
        --lr                  learning rate
        --bs                  training batch size
        -m, --max_epoch       set max training epochs
        --set ...             set config keys

### Training sample
If we want to start an experiment, training can be triggered by corresponding config file
```shell
# <DeepPersonality as the top dir>
script/run_exp.py \
--cfg_file config/unified_frame_images/03_bimodal_resnet18.yaml 

```

### Resume sample
If we want to resume training from a certain training checkpoint(saved model weights), parameter `resume` can be specified 
along with the saved weights. And before re-training, the training epochs and learning rate can be reset again if needed.
```shell
# <DeepPersonality as the top dir>
script/run_exp.py \
-c config/unified_frame_images/03_bimodal_resnet18.yaml \
--resume results/unified_frame_images/03_bimodal_resnet/12-19_18-15/checkpoint_199.pkl \
--max_epoch 210 \
--lr 0.001
```
### Test sample
If we only want to test a trained model, parameter `test_only` can be used, and along with `set` parameters to specify the model 
weights used, shown as below:
```shell
script/run_exp.py \
-c config/unified_frame_images/09_hrnet.yaml \
--test_only \
--set TEST.WEIGHT results/unified_frame_images/09_hrnet/12-20_22-12/checkpoint_186.pkl
```
