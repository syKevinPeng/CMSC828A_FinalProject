import pathlib, datetime
from dataloader import PrepareDataLoader
from model import inception, inception_cl, KD_loss
from utils.utils import get_logger
import numpy as np
from tensorflow import keras
from pathlib import Path
class Trainer:
    def __init__(self, experiment_config, finetune_config):
        self.experiment_config = experiment_config
        self.finetuning_config = finetune_config
        self.output_dir = pathlib.Path(self.experiment_config["output_directory"])/self.experiment_config['exp_name']
        self.logger = get_logger(self.output_dir, "Trainer")
        self.model_type = self.experiment_config["model_type"]
        if self.model_type not in ['baseline', 'cl', 'mtl']:
            raise ValueError(f"Model type {self.model_type} not supported")
        self.universal_label = self.finetuning_config['universal_label']
        self.debug = self.experiment_config["debug"]
        self.learning_rate = self.experiment_config["learning_rate"]
        self.load_weights = self.experiment_config["load_weights"]
        if self.load_weights:
            self.weights_file = self.experiment_config["weights_file"]

    def train(self):
        if self.model_type in ['bl', 'baseline', 'Baseline']:
            self.ft_baseline()
        elif self.model_type in ['cl', 'CL','ContinueLearning']:
            self.ft_cl()
        elif self.model_type in ['mtl', 'MTL','MultitaskLearning']:
            self.ft_mtl()

    def ft_baseline(self):
        nb_epochs = self.experiment_config["training_epochs"]
        verbose = self.experiment_config["verbose"]
        # load data
        dataloader = PrepareDataLoader(self.finetuning_config, self.experiment_config)
        self.logger.info(f"Loading data ...")
        train_dataloader, valid_dataloader  = dataloader.load_pretrain_data(labels = self.universal_label, model_type = "baseline")
        if not self.output_dir.is_dir():
            self.logger.warning(f"Parent directory {self.output_dir} not found. Creating directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        nb_classes = len(self.universal_label)

        input_shape =(3,1)
        ## DO NOT SET BATCH SIZE HERE
        model = inception.Classifier_INCEPTION(self.output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                depth = 2,
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False,
                                                                lr = self.learning_rate
                                                                )
        # load pretrained weights
        if self.load_weights:
                self.logger.info(f"Loading weights from {self.experiment_config['weights_file']}")
                model.model.load_weights(self.experiment_config['weights_file'])
        self.logger.info("---- Start training ----") 
        model.fit(train_dataloader, valid_dataloader)
        self.logger.info("---- End training ----")
    
    def ft_cl(self):
        pass
    
    def ft_mtl(self):
        pass