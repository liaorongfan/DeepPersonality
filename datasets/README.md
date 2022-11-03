# Datasets
This file describes the data pre-procession and preparation for training.
## ChaLearn 2016 First Impression Dataset
### Data Pre-processing 
That dataset contains 10, 000 video clips which are seperated into train, valid and test splits. There are 6000, 2000 and 2000 
clips in train, valid and test split respectively. For each video clip, we extract all frames in it into a directory with 
the same name as the video, and then extract face images from those full frames. The extracted face images also organized 
in corresponding directories, show as **ChaLearn 2016 Data Structure**

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
### Scrips for Video Frame Extraction

```shell
python dpcv/data/utils/video_to_image.py --video-path /path/to/video/directory
```
This script will find all mp4 files in the `directory` specified by `--video-path` and extract frames in each video into a
directory which shares the same path and name with the video. For example, provided a directory with 2 mp4 videos
```
video_dir/
    |-- video_1.mp4
    |-- video_2.mp4
```
After frame extraction, it will be like this.
```
video_dir/
    |-- video_1.mp4
    |-- video_1/
    |   |-- frame_1.jpg
    |   |-- frame_2.jpg
    |   |-- ...
    |-- video_2.mp4
    |-- video_2/
```