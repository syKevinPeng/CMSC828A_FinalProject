
# Comparision Between Multi-task Learning and Continue Learning Using time-seriers data in terms of Performance, Efficacy and Transferbility

## introduction
In this research project, we will answer the following question: How do continual learning and multi-task learning approach compare in performance, computational efficiency, and robustness for human action recognition using smartwatch accelerometer data? Specifically, we aim to measure the overall performance of the two models in recognition accuracy, the training/inference time, and their transferability to other HAR datasets. We hope to gain insights into the strengths and weaknesses of the continual learning and multi-tasking learning method.

The project is divded into two phase:
 - Phase 1: Traning base models: Applying continuous learning and mutl-task learning paradiam on wisdm dataset.
 - Phase 2: Transfer/Personalize the model for data collected in Extrasensory




## File Structure
- **checkpoints**: where all weights/models are stored
- **datasets**: where datasets are located
    - **extrasensory**: where raw data are stored
    - **preprocessed**: where the preprocessed data stored
    - *extrasensory_download.sh*: automatic download script for extrasensory
- **src**: source code
    - **configs** : store your config here
    - **model**: where actual model script located
        - *inception.py*: the baseline model
    - **preprocess**: preprocessing script are located. We process the data off-line, instead of during the training, and store as csv File
        -e*xtrasensory.py* : preprocessing script of the extrasensory datasets
    - **utils**: utility scripts
        - *inception_constants.py*: inhereted from original repo, might not be used
        - *inception_utils.py*: inhereted from original repo, important for training
        - *utils.py*: our utils file. Contains function for custom callback (basically print training result after a epoch)
    - *dataloader.py*: load dataset to the memory
    - *main.py*: main script the of training, where the training starts and ends
    - *pretrain.py*: phase one script for pretraining on raw data
    - *transfer.py*: phase two script for transfer learning/personalization (TODO)
- *environment.yml*: contains information of used packages in virtual environment.
- *request_interactive.sh*: this file will return a shell with GPU attached for 10 minutes. Mainly for debugging use. After requesting a GPU, run "source ~/.profile" to activte virtual environment.
- *running.log*: training logs
- *train.sh*: this script would request GPU from the resource management module and training it based on your config and setting. 

## Training workflow
1. The training pipeline will start at main.py by calling train.sh
### CMD parameters:
- `--config`: required, path to your config file. Default is mine config file `configs/config_siyuan.yaml`
- `--train`:  not required, whether to include training part of the pipline.
2. After loading the config file, it will determine if it goes into training section (determined by your args `--train`).
3. In the prtrain.py, the pipline loads the data by initilizing `Dataloader` objects and get the corrsponding labels. `dataloader.load_pretrain_data()` is the actual place where the data get loaded. 
4. Then at line 40 of the pretrain.py file, call the model by adding the configs of the training
5. Line 47 of the pretrain.py file is the actual place where it starts training. 

## Config README
This part of the README is copied from `/src/configs/README.md`
### Introduction
For the config file, there are mainly three parts: Preprocessing, pretrain, experiment. Please pay attention to the experiment config. That's where you need to modify for your own experiments. 
### Config Details
- preprocessing: where the preprocessing config are stored (No need to modification)
- pretrain: where the pretrain model config is stored 
- experiment: training config stored. **IMPORTANT**
    - exp_name: the experiment prefix/name. It would be append to the output dir. **REQUIRED**
    - dataset: select dataset to use. one of ["ES", "SOME OTHER DATASET: TBD"]. This will be useful for phase 2.
    - force_preprocess: whether to force the pipeline re-preprocess the raw file. Usually we set it false, but if there is something wrong with preprocessing, we can set it true. 
    - training_epochs: total number of epochs to train
    - output_directory: where to output weights. default is './checkpoints'
    - batch_size: 8 # note: I tried 16, 32 and they all receive OM error (out of memory)
    - model_type: select one of ["baseline","CL", "MTL"]. CL stands for Continue Learning and MTL stands for Multi-task Learning. You might want to tweak this.
