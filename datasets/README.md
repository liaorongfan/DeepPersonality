# Datasets
This file describes the data pre-processing and preparation for training.
## ChaLearn 2016 First Impression Dataset
### Data Pre-processing 
That dataset contains 10, 000 video clips which are seperated into train, valid and test splits. There are 6000, 2000 and 2000 
clips in train, valid and test split respectively. For each video clip, we extract all frames in it into a directory with 
the same name as the video, and then extract face images from those full frames. The extracted face images also organized 
in corresponding directories, shown as **ChaLearn 2016 Data Structure**

Following, We will demonstrate the pre-processing procedure by a [tiny Chalearn 2016 dataset](https://drive.google.com/file/d/1S87nJFLz9ygzw2Ep_rJUXzzWFfdz15an/view?usp=sharing) containing 100 videos within which 60 for 
training, 20 for validation and 20 for test.

### ChaLearn 2016 Data Structure
Expected dataset structure for Chalearn 2016 dataset. 
```
datasets/
|--- image_data/
|    |--- test_data/
|    |    |--- video_name_1/
|    |    |    |--- frame_1.jpg
|    |    |    |---  ...
|    |    |--- video_name_2/
|    |    |--- video_name_3/
|    |    |--- ...
|    |--- valid_data/
|    |    |--- video_name_1/
|    |    |    |--- frame_1.jpg
|    |    |    |---  ...
|    |    |--- video_name_2/
|    |    |--- ...
|    |--- train_data/
|    |    |--- video_name_1/
|    |    |---  ...
|    |--- test_data_face/
|    |    |--- video_name_1/
|    |    |    |--- face_1.jpg
|    |    |    |---  ...    
|    |--- valid_data_face/
|    |--- train_data_face/
|    |
|--- voice_data/
|    |--- voice_raw/
|    |    |--- test_data/
|    |    |    |--- video_name_1.wav
|    |    |    |--- video_name_2.wav  
|    |    |    |--- ...
|    |    |--- valid_data/
|    |    |--- train_data/   
|    |--- voice_librosa/
|    |    |--- test_data/
|    |    |--- valid_data/
|    |    |--- train_data/
|    |--- voice_logfbank/
|    |    |--- test_data/
|    |    |--- valid_data/
|    |    |--- train_data/

```
### Script for Video Frame Extraction

```shell
python dpcv/data/utils/video_to_image.py --video-dir /path/to/video_dir --output-dir /path/to/output_dir
```
This script will find all mp4 files in the `video directory` specified by `--video-dir` and extract frames in each video into a
`image directory`  with the same name as the video. All the `image directories` will be saved into the path assigned by `--output-dir`. For example, provided a directory with 2 mp4 videos
```
video_dir/
    |-- video_1.mp4
    |-- video_2.mp4
```
After frame extraction, it will be like this.

```
output_dir/
    |-- video_1/
    |   |-- frame_1.jpg
    |   |-- frame_2.jpg
    |   |-- ...
    |-- video_2/
```


### Script for Video Face Extraction

When extracting face images from frames, we used the pretrained models which can be found in
[Google Drive](https://drive.google.com/drive/folders/1gxkjIkIt7jOk_3RJhzORUzIj9NkIaqT1?usp=sharing)
and those models should be simply placed in directory `DeepPersonality/pre_trained_weights`, shown as below:
```
pre_trained_weights/
├── shape_predictor_68_face_landmarks.dat
├── vgg16_bn-6c64b313.pth
└── vgg_m_face_bn_fer_dag.pth
```

The following script will also find all mp4 files in the`directory` specified by `--video-dir` and extract face images from 
every frame in each video into a directory which shares the same name with the video. Those directories are saved in 
`path` specified by `--output-dir`

```shell
python dpcv/data/face_extract/face_img_extractor.py --video-dir /path/to/video/dir --output-dir /path/to/output/dir 
```

For example, provided a directory with 2 mp4 videos, and we run the following command line
```shell
python dpcv/data/face_extract/face_img_extractor.py  \
    --video-dir /path/to/train_videos_dir \ 
    --output-dir /path/to/train_face_img_dir 
```
```
train_video_dir/
    |-- video_1.mp4
    |-- video_2.mp4
```
After face image extraction, it will be like this.
```
train_face_img_dir/
    |-- video_1/
    |   |-- face_1.jpg
    |   |-- face_2.jpg
    |   |-- ...
    |-- video_2/
```


### Script for Audio Extraction

#### Extract raw audio files
The script below will find all mp4 files in the`directory` specified by `--video-dir` and extract `audio file` in `wav` 
format from each video into a directory which shares the same name with the video. Those directories are saved in 
`path` specified by `--output-dir`

```shell
python dpcv/data/utils/video_to_wave.py --video-dir /path/to/video/dir --output-dir /path/to/output/dir 

```

For example, provided a directory with 2 mp4 videos, and we run the following command line
```shell
python dpcv/data/utils/video_to_wave.py  \
    --video-dir /path/to/train_videos_dir \ 
    --output-dir /path/to/voice_data/voice_raw/train_data
```
```
train_video_dir/
    |-- video_1.mp4
    |-- video_2.mp4
```
After face image extraction, it will be like this.
```
voice_data/voice_raw/train_data/
                        |-- video_1.wav
                        |-- video_2.wav
```

#### Process raw audio files
The audio data processing methods we used include `librasa`, `logfbank`, `mfcc` and `opensmile`, Here we take `librasa` 
as an example.
```shell
python dpcv/data/utils/raw_audio_process.py --mode <librasa/logfbank/opensmile> \
    --audio-dir /path/to/audio_dir \
    --output-dir /path/to/output_dir 
```
In this script the arguments `--audio-dir` and `--output-dir` specify the directory path to audio file and directory to 
save the processed audio files.

## ChaLearn2021 True Personality Dataset
### Data Pre-processing
We employ the same data processing methods in ChaLearn2016 dataset descript above.

### ChaLearn 2021 Data Structure
```
datasets/
|    chalearn2021
|    ├── annotation
|    │   └── talk
|    ├── test
|    │   └── talk_test
|    ├── train
|    │   └── talk_train
|    └── valid
|        └── talk_valid


```