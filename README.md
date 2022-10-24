# An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition
## Introduction
This is the code repo of our paper *An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition*.
we propose the audio-visual personality computing benchmark for both self-reported and apparent personality
recognition tasks, which are evaluated on two widely-used and publicly available personality computing datasets: 
the [ChaLearn First Impression dataset](https://chalearnlap.cvc.uab.cat/dataset/24/description/#) and
the [ChaLearn UDIVA self-reported personality dataset](https://chalearnlap.cvc.uab.es/dataset/41/description/#)

`Seven visual models`, `six audio models` and `five audio-visual models` have been reproduced and evaluated on 
the personality datasets mentioned above. Besides, `seven widely-used visual deep learning models`, which have not been
applied to video-based personality computing before, have also been employed for benchmark. Detailed description can be
found in our paper.

<center>
<img src="docs/figures/pipeline.png" />
</center>

This code base is designed to develop new personality computing models and to conduct quick evaluation of
new models on both self-reported and apparent personality recognition tasks. Future work will focus on 
updating the code base by adding newly published personality computing models, evaluating the benchmarked models on new
personality datasets as well as adding more modalities (e.g.,language). **We will also continuously improve the code framework
and try to make it easy-to-use, hoping this work would save as a useful tool-kit to all interested researchers.**


## Requirements and Dependencies
 **Setup project**: you can use both Conda and Virtualenv to create a virtual environment to run this program.
```shell
# clone current repo
git clone DeepPersonality
cd DeepPersonality

# create and activate a virtual environment
virtualenv -p python38 venv
source venv/bin/activate

# install required packages and dependencies
pip install -r requirements.txt
```
## Datasets 
The datasets we used for benchmark are [Chalearn First Impression](https://chalearnlap.cvc.uab.cat/dataset/24/description/#) 
and [UDIVA](https://chalearnlap.cvc.uab.es/dataset/41/description/#). 

The former contains  10, 000 video clips that come from 2, 764 YouTube users for apparent personality recognition(impression), 
where each video lasts for about 15 seconds with 30 fps. 

The latter, UDIVA for self-reported personality, is released in 2021 and records 188 dyadic interaction video clips between 147 
voluntary participants, with total 90.5h of recordings. Every clip contains two audiovisual files, where each records a 
single participant’s behaviours. 

Each video in both datasets is labelled with the Big-Five personality traits.

To meet various requirements from different models or experiments, we extract raw audio file and all frames from a video
and then extract all face images from each full frame, termed as face frames.

**Please find the process methods in [dataset preparation](datasets/README.md).**
## Usage for reproducing reproted experiments
We employ a build-from-config manner to conduct an experiment. After setting up the environments and preparing the data needed,
we can have a quick start by the following command line:
```shell
# cd DeepPersonality # top directory 
script/run_exp.py --config path/to/config.yaml 
```
For example:
```shell
# cd DeepPersonality # top directory
script/run_exp.py --config config/unified_frame_images/01_deep_bimodal_regression.yaml
```


### Command line interface
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

#### Training sample
If we want to start an experiment, training can be triggered by corresponding config file
```shell
# <DeepPersonality as the top dir>
script/run_exp.py \
--cfg_file config/unified_frame_images/03_bimodal_resnet18.yaml 

```

#### Resume sample
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
#### Test sample
If we only want to test a trained model, parameter `test_only` can be used, and along with `set` parameters to specify the model 
weights used, shown as below:
```shell
script/run_exp.py \
-c config/unified_frame_images/09_hrnet.yaml \
--test_only \
--set TEST.WEIGHT results/unified_frame_images/09_hrnet/12-20_22-12/checkpoint_186.pkl
```



## Usage for developing new personality computing models
We use config-pipe line files and registration mechanism to organize our experiments. If user want to add their own 
models or algorithms into this program please reference the config files in it.




## Papers from which the models are reproduced

- Deep bimodal regression of apparent personality traits from short video sequences
- Bi-modal first impressions recognition using temporally ordered deep audio and stochastic visual features
- Deep impression: Audiovisual deep residual networks for multimodal apparent personality trait recognition
- Cr-net: A deep classification-regression network for multimodal apparent personality analysis
- Interpreting cnn models for apparent personality trait regression
- On the use of interpretable cnn for personality trait recognition from audio
- Persemon: a deep network for joint analysis of apparent personality, emotion and their relationship
- A multi-modal personality prediction system
- Squeeze-and-excitation networks
- Deep high-resolution representation learning for visual recognition
- Swin transformer: Hierarchical vision transformer using shifted windows
- Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet
- Slowfast networks for video recognition
- Temporal pyramid network for action recognition
- Video action transformer network



## To Be Updated
- [ ] Detailed Data prepare pipeline description
- [ ] Pip install
- [ ] Description of adding new models
