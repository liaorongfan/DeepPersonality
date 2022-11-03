# An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition
## Introduction
This is the official code repo of *An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition* (https://arxiv.org/abs/2210.09138).

In this project, **seven visual models**, **six audio models** and **five audio-visual models** have been reproduced 
and evaluated. Besides, **seven widely-used visual deep learning models**, which have not been applied to video-based 
personality computing before, have also been employed for benchmark. Detailed description can be found in our paper.

All benchmarked models are evaluated on: 
the [ChaLearn First Impression dataset](https://chalearnlap.cvc.uab.cat/dataset/24/description/#) and
the [ChaLearn UDIVA self-reported personality dataset](https://chalearnlap.cvc.uab.es/dataset/41/description/#)


<center>
<img src="docs/figures/pipeline.png" />
</center>

This project is currently under active development. Documentation, examples, and tutorial will be progressively detailed

## Requirements and dependencies
 **Setup project**: you can use either Conda or Virtualenv to create a virtual environment to run this program.
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


## Datasets description and preparation
### Datasets used
The datasets we used for benchmark are [Chalearn First Impression](https://chalearnlap.cvc.uab.cat/dataset/24/description/#) 
and [UDIVA](https://chalearnlap.cvc.uab.es/dataset/41/description/#). 

- The former contains  10, 000 video clips that come from 2, 764 YouTube users for apparent personality recognition(impression), 
where each video lasts for about 15 seconds with 30 fps. 

- The latter, for self-reported personality, records 188 dyadic 
interaction video clips between 147 voluntary participants, with total 90.5h of recordings. Every clip contains two audiovisual files, where each records a single participant’s behaviours. 

- Each video in both datasets is labelled with the Big-Five personality traits. 
### Data pre-processing
To meet various requirements from different models or experiments, we extract raw audio file and all frames from a video
and then extract face images from each full frame, termed as face frames. **Please find the process methods in 
[dataset preparation](datasets/README.md).**

### Pretrained weights
When extracting face images from frames, we used the pretrained models which can be found in 
[Google Drive](https://drive.google.com/drive/folders/1gxkjIkIt7jOk_3RJhzORUzIj9NkIaqT1?usp=sharing)
and those models should be simply placed in directory `pre_trained_weights`, shown as below:
```
pre_trained_weights/
├── shape_predictor_68_face_landmarks.dat
├── vgg16_bn-6c64b313.pth
└── vgg_m_face_bn_fer_dag.pth
```


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
For detailed usage or arguments description, please find more in **[command line interface file](docs/Command_line_interface.md)**.



## Usage for developing new personality computing models
We use config-pipe line files and registration mechanism to organize our experiments. If user want to add their own 
models or algorithms into this program please reference the **[Notebook tutorials]()**.




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


## History
- 2022/10/17 - Paper submission and make project publicly available.

## To Be Updated
- [ ] Detailed Data prepare pipeline description
- [ ] Pip install
- [ ] Description of adding new models
