# An Open-source Benchmark of Deep Learning Models for Audio-visual Apparent and Self-reported Personality Recognition

<img src="docs/figures/pipeline.png"/>

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
The datasets we used for benchmark are `Chalearn First Impression` and `UDIVA`. 

The former contains  10, 000 video clips that come from 2, 764 YouTube users for apparent personality recognition(impression), 
where each video lasts for about 15 seconds with 30 fps. Each video is labelled with the Big-Five personality traits that
are annotated by human annotators.

The latter, UDIVA for self-reported personality, is released in 2021 and records 188 dyadic interaction video clips between 147 
voluntary participants, with total 90.5h of recordings. Each clip contains two audiovisual files, where each records a 
single participant’s behaviours. During the recordings, participants were asked to sit at 90 degrees to the conversational
partner around a table, and under the dyadic interactions based on five tasks: Talk, ’Animal games’, Lego building, 
“Ghost blitz” card game, and Gaze events.

To meet various requirements from different models or experiments, we first extract all frames from a video and then extract
all face images from each full frame, termed as face frames.

**Please find the process methods in [dataset preparation](datasets/README.md).**
## Usage for reproducing reproted experiments
We employ a build-from-config manner to set up an experiment

### Command line interface
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



## Usage for developing new personality computing models
We use config-pipe line files and registration mechanism to organize our experiments. If user want to add their own 
models or algorithms into this program please reference the config files in it.





## To Be Updated
(The code structure and README will be updated soon to facilitate all potential users)
