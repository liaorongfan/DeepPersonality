# An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition
## Introduction
This is the official code repo of *An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition* (https://arxiv.org/abs/2210.09138).

The accepted supplementary material is provided in [Personality_benchmark_Supplementary_.pdf](https://github.com/liaorongfan/DeepPersonality/blob/add_new_data_loader/Personality_benchmark_Supplementary_.pdf)

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


## Installation
 **Setup project**: you can use either Conda or Virtualenv/pipenv to create a virtual environment to run this program.

```shell
# create and activate a virtual environment
virtualenv -p python38 venv
source venv/bin/activate
````

#### Installing from PyPI

```shell
pip install deep_personality
``` 

#### Installing from Github
```shell
# clone current repo
git clone DeepPersonality
cd DeepPersonality

# install required packages and dependencies
pip install -r requirements.txt
```


## Datasets 

The datasets we used for benchmark are [Chalearn First Impression](https://chalearnlap.cvc.uab.cat/dataset/24/description/#) 
and [UDIVA](https://chalearnlap.cvc.uab.es/dataset/41/description/#). 

- The former contains  10, 000 video clips that come from 2, 764 YouTube users for apparent personality recognition(impression), 
where each video lasts for about 15 seconds with 30 fps. 

- The latter, for self-reported personality, records 188 dyadic 
interaction video clips between 147 voluntary participants, with total 90.5h of recordings. Every clip contains two audiovisual files, where each records a single participant’s behaviours. 

- Each video in both datasets is labelled with the Big-Five personality traits. 


To meet various requirements from different models and experiments, we extract raw audio file and all frames from a video
and then extract face images from each full frame, termed as face frames. 

For quick start and demonstration, we provide a
**[tiny Chalearn 2016 dataset](https://drive.google.com/file/d/1S87nJFLz9ygzw2Ep_rJUXzzWFfdz15an/view?usp=sharing)** 
containing 100 videos within which 60 for training, 20 for validation and 20 for test. Please find the process methods in 
**[dataset preparation](datasets/README.md).**

For your convenience, we provide the processed **[face image frames dataset](https://drive.google.com/drive/folders/1UQQVQt5DZxgQPVI1seNbsqiPE436I06K?usp=share_link)** for Chalearn 2016 
since that dataset is publicly available, which indicates we can make our processed data open to the community.

## Experiments![](https://colab.research.google.com/assets/colab-badge.svg)
### Reproducing reported experiments
We employ a build-from-config manner to conduct an experiment. After setting up the environments and preparing the data needed,
we can have a quick start by the following command line:
```shell
# cd DeepPersonality # top directory 
python ./tools/run_exp.py --config path/to/exp_config.yaml 
```
For quick start with [tiny ChaLearn 2016 dataset](https://drive.google.com/file/d/1S87nJFLz9ygzw2Ep_rJUXzzWFfdz15an/view?usp=sharing),
if you prepare the data by the instructions in above section, the following command will launch an experiment for `bimodal-resnet18 model`.
```shell
# cd DeepPersonality # top directory
python ./tools/run_exp.py --config config/demo/bimodal_resnet18.yaml
```

Detailed arguments description are presented in  **[command line interface file](docs/Command_line_interface.md)**.


For quick start demonstration, please find the Colab Notebook:
**[QuickStart](https://colab.research.google.com/drive/1QgISDUbLN0UFFK78e5Zmwcyfn8xtpr3o?usp=sharing)**


For experiments start from raw video processing, please find this Colab Notebook:
**[StartFromDataProcessing](https://colab.research.google.com/drive/1_YThB_O1nO4y9zK6AHOUEx-a2zb_ba9y?usp=sharing)**


### Developing new personality computing models
We use pipeline config files and registration mechanism to organize our experiments. 

In the framework user can design and config their own spatial-temporal data preprocessing approaches. 

If user want to add their own models or algorithms for experiments, please reference the Colab Notebook
**[TrainYourModel](https://colab.research.google.com/drive/1lB3B0C9LgZ6NmZmsdblRKcfiY6nWoEcI?usp=sharing)**


## Data augmentation

To allow users to use different data augmentation strategies, the framework provides a command line argument 
"--set DATA_LOADER.TRANSFORM < augmentation strategy >" to choose the data augmentation strategy. The details are listed as follows:

- `standard_frame_transform`: the default data augmentation strategy.

- `strong_frame_transform`: a stronger data augmentation strategy with multiple image transforms.

- `customized_frame_transform`: a customized data augmentation strategy, which can be defined by users.


### Standard_frame_transform
Step 1: Resizing the input image (or cropped face image) as a 3-channel rectangle whose shorter edge is 256 pixles long. For example, if an image has a size of (width: 720, height: 512), the resized image will have the size of (width: 360, height: 256);

Step 2: the resized image will be filpped horizontally at a probablity of 0.5;

Step 3: a image patch of the size (width: 224, height: 244) is cropped from the filpped image;

Step 4: the pixel values (P) in the cropped image will be normalized to a numerical range of (-1, 1) by the formulation of $P = (p / 255 - mean) / std$,  
where the $mean$ and $std$ values for each image channel are [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] respectively.

### Strong_frame_transform
Besides the first two augmentaion steps in `standar_frame_transform` strategy ( `resize` and `horiziontally flip`), this strategy also sequentially employs `random rotation`, `color jitter`, and `random resized crop` followed by the same image normalization operation (Step 4) described in `standard_frame_tansform`.

`random rotation`: the image is rotated by a  randomly selected angle with the range of (-5, 5). 

`color jitter`: the brightness, contrast and saturation of the image are randomly changed, where the ranges of scaling/jittering factors for the brightness, contrast and saturation are (0.9, 1.1).

`random resized crop`: a patch (whose size is 0.8 to 1 of the original image) is randomly cropped from the image, which is then resized into the size of (width: 224, height: 224).

The command line for using the strong data augmentation strategy is:
```shell

python ./tools/run_exp.py --config config/demo/(model.yaml, e.g., bimodal_resnet18.yaml)  \
                          --set DATA_LOADER.TRANSFORM strong_frame_transform

````

### Customized_frame_transform

First, defining your own data transform function, and registering it to the registry. 
```python
# step 1: importing the registry for data transforms：
from dpcv.data.transforms.build import TRANSFORM_REGISTRY

# step 2: defining your own data transform function and registering it to the registry
@TRANSFORM_REGISTRY.register()
def my_frame_transform(cfg):
    """
    define your own data transform function
    """
    # take the transforms functions from torchvision for your reference
    import torchvision.transforms as transforms
    transforms = transforms.Compose([
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
    ])
    
    return transform

```
The command line for using the customized frame transform strategy is:

```shell

python ./tools/run_exp.py --config config/demo/bimodal_resnet18.yaml  \
                          --set DATA_LOADER.TRANSFORM my_frame_transform

````


## Developing models with metadata or additional data
This framework facilitates user to train models with metadata or additional data. The guideline is provided below:
- Step 1: Preparing the additional data
    - Preparing the metadata or additional data in a csv file.
    - The first column should be the video file name.

- Step 2: Modifing the training config file:
    ```yaml
  # file: config/demo/hrnet_use_other_data.yaml
    # modify the training config file
  DATA:
      ROOT: "datasets/chalearn2021"
      # the path to the metadata or additional data
      TRAIN_OTHER_DATA: "datasets/chalearn21_metadata/metadata_train.csv"
      VALID_OTHER_DATA: "datasets/chalearn21_metadata/metadata_valid.csv"
      TEST_OTHER_DATA: "datasets/chalearn21_metadata/metadata_test.csv"
  DATA_LOADER:
      # the name of the data_loader used for adding additional data
      NAME: "all_true_personality_with_other_data_loader"
  MODEL:
      # the name of the model used for training with additional data
      NAME: "hr_net_true_personality_with_meta_data" (please selecting and modifing your model)
      USE_OTHER_DATA: True
      # the dimension of the additional data
      OTHER_DATA_DIM: 3 (please customize the number here (the dimension of the metadata/additional data))
    ```
- Step 3: Training the model
    ```shell
    python ./tools/run_exp.py --config config/demo/hrnet_use_other_data.yaml
    ```



# Models
## On ChaLearn 2016 dataset
| Model                                                                       |    Modal    |                                                           ChaLearn2016 cfgs                                                            |                                                                                         ChaLearn2016 weights                                                                                         |
|-----------------------------------------------------------------------------|:-----------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [DAN](dpcv/modeling/networks/dan.py)                                        |   visual    |                                   [cfg](config/unified_frame_images/01_deep_bimodal_regression.yaml)                                   |                                                     [weight](https://drive.google.com/file/d/1r2Cn8tLje7otFHklU7aQHj5Wqlc2qS2u/view?usp=sharing)                                                     |
| [CAM-DAN<sup>+](dpcv/modeling/networks/interpret_dan.py)                    |   visual    |                                        [cfg](config/unified_frame_images/06_interpret_cnn.yaml)                                        |                                                     [weight](https://drive.google.com/file/d/1EvHIb0Bfk2mYqUlcpB8kFEVLSAFTHYd2/view?usp=sharing)                                                     |
| [ResNet](dpcv/modeling/module/resnet_tv.py)                                 |   visual    |                                   [cfg](config/unified_frame_images/03_bimodal_resnet18_visual.yaml)                                   |                                                     [weight](https://drive.google.com/file/d/1qEuFFGlLYdwweDgIVcsrG1_VEjbABsOw/view?usp=sharing)                                                     |
| [HRNet](dpcv/modeling/networks/hr_net_cls.py)                               |   visual    |            [cfg-frame](config/unified_frame_images/09_hrnet.yaml)/[cfg-face](config/unified_face_images/02_hrnet_face.yaml)            | [weight-frame](https://drive.google.com/file/d/1xkYWFz9gGWdtwaezQGsiojsZjHH7l-Uh/view?usp=sharing)/[weight-face](https://drive.google.com/file/d/1V-oVJU3GbMeWR5sEQvDpd36GAmeZQQXX/view?usp=sharing) |
| [SENet](dpcv/modeling/networks/se_net.py)                                   |   visual    |            [cfg-frame](config/unified_frame_images/08_senet.yaml)/[cfg-face](config/unified_face_images/01_senet_face.yaml)            |                                                                                              [weight]()                                                                                              |
| [3D-ResNet](dpcv/modeling/networks/resnet_3d.py)                            |   visual    |      [cfg-frame](config/unified_frame_images/11_resnet50_3d.yaml)/[cfg-face](config/unified_face_images/04_resnet50_3d_face.yaml)      |                                                                                              [weight]()                                                                                              |
| [Slow-Fast](dpcv/modeling/networks/slow_fast.py)                            |   visual    |        [cfg-frame](config/unified_frame_images/12_slow_fast.yaml)/[cfg-face](config/unified_face_images/05_slow_fast_face.yaml)        |                                                                                              [weight]()                                                                                              |
| [TPN](dpcv/modeling/networks/TSN2D.py)                                      |   visual    |              [cfg-frame](config/unified_frame_images/13_tpn.yaml)/[cfg-face](config/unified_face_images/06_tpn_face.yaml)              |                                                                                              [weight]()                                                                                              |
| [Swin-Transformer](dpcv/modeling/networks/swin_transformer.py)              |   visual    | [cfg-frame](config/unified_frame_images/10_swin_transformer.yaml)/[cfg-face](config/unified_face_images/03_swin_transformer_face.yaml) |                                                                                              [weight]()                                                                                              |
| [VAT](dpcv/modeling/networks/video_action_transformer.py)                   |   visual    |              [cfg-frame](config/unified_frame_images/14_vat.yaml)/[cfg-face](config/unified_face_images/07_vat_face.yaml)              |                                                                                              [weight]()                                                                                              |
| [Interpret Audio CNN](dpcv/modeling/networks/audio_interpretability_net.py) |    audio    |                                       [cfg](config/unified_frame_images/07_interpret_audio.yaml)                                       |                                                                                              [weight]()                                                                                              |
| [Bi-modal CNN-LSTM](dpcv/modeling/networks/bi_modal_lstm.py)                | audiovisual |                                        [cfg](config/unified_frame_images/02_bimodal_lstm.yaml)                                         |                                                                                              [weight]()                                                                                              |
| [Bi-modal ResNet](dpcv/modeling/networks/audio_visual_residual.py)          | audiovisual |                                      [cfg](config/unified_frame_images/03_bimodal_resnet18.yaml)                                       |                                                                                              [weight]()                                                                                              |
| [PersEmoN](dpcv/modeling/networks/sphereface_net.py)                        | audiovisual |                                          [cfg](config/unified_frame_images/05_persemon.yaml)                                           |                                                                                              [weight]()                                                                                              |
| [CRNet](dpcv/modeling/networks/cr_net.py)                                   | audiovisual |                                            [cfg](config/unified_frame_images/04_crnet.yaml)                                            |                                                     [weight](https://drive.google.com/file/d/1iNIJxIuW3XTKW_w48RoddY7-ZIKwGf7z/view?usp=sharing)                                                     |
| [Amb-Fac](dpcv/modeling/networks/multi_modal_pred_net.py)                   | audiovisual | [cfg-frame](config/unified_frame_images/15_multi_modal_pred.yaml), [cfg-face](config/unified_face_images/8_multi_modal_pred_face.yaml) |                                                                                              [weight]()                                                                                              |


## On ChaLearn 2021 dataset

| Model                                                                       |    Modal    |        Talk         |        Lego         |        Ghost        |       Animal        |
|-----------------------------------------------------------------------------|:-----------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
| [DAN](dpcv/modeling/networks/dan.py)                                        |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [CAM-DAN<sup>+](dpcv/modeling/networks/interpret_dan.py)                    |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [ResNet](dpcv/modeling/module/resnet_tv.py)                                 |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [HRNet](dpcv/modeling/networks/hr_net_cls.py)                               |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [SENet](dpcv/modeling/networks/se_net.py)                                   |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [3D-ResNet](dpcv/modeling/networks/resnet_3d.py)                            |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [Slow-Fast](dpcv/modeling/networks/slow_fast.py)                            |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [TPN](dpcv/modeling/networks/TSN2D.py)                                      |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [Swin-Transformer](dpcv/modeling/networks/swin_transformer.py)              |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [VAT](dpcv/modeling/networks/video_action_transformer.py)                   |   visual    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [Interpret Audio CNN](dpcv/modeling/networks/audio_interpretability_net.py) |    audio    | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [Bi-modal CNN-LSTM](dpcv/modeling/networks/bi_modal_lstm.py)                | audiovisual | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [Bi-modal ResNet](dpcv/modeling/networks/audio_visual_residual.py)          | audiovisual | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [PersEmoN](dpcv/modeling/networks/sphereface_net.py)                        | audiovisual | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [CRNet](dpcv/modeling/networks/cr_net.py)                                   | audiovisual | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |
| [Amb-Fac](dpcv/modeling/networks/multi_modal_pred_net.py)                   | audiovisual | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() | [cfg]()  [weight]() |

## Related papers 
The models that we have or will reproduced:

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


## Citation

If you use our code for a publication, please kindly cite it as:
```
@article{liao2022open,
  title={An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition},
  author={Liao, Rongfan and Song, Siyang and Gunes, Hatice},
  journal={arXiv preprint arXiv:2210.09138},
  year={2022}
}
```

## History
- 2022/10/17 - Paper submission and make project publicly available.

## To Be Updated

[//]: # (- [ ] Detailed Data prepare pipeline description)
- [ ] Test Pip install
- [ ] Description of adding new models
- [ ] Model zoo
- [ ] Notebook tutorials
