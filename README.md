# Benchmarking Deep Learning Models for Audio-visual Automatic Self-reported and Apparent Personality Recognition
# (The code structure and README will be updated soon for all potential users)


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





