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