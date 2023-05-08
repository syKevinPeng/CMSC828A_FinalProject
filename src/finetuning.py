import pathlib, datetime
from dataloader import PrepareDataLoader
from model import inception, inception_cl, KD_loss
from utils.utils import get_logger
import numpy as np
from tensorflow import keras

class Trainer:
    def __init__(self, experiment_config, finetune_config):
        self.experiment_config = experiment_config
        self.pretrain_config = finetune_config
        self.output_dir = pathlib.Path(self.experiment_config["output_directory"])/self.experiment_config['exp_name']
        self.logger = get_logger(self.output_dir, "Trainer")
        self.model_type = self.experiment_config["model_type"]
        if self.model_type not in ['baseline', 'cl', 'mtl']:
            raise ValueError(f"Model type {self.model_type} not supported")
        self.universal_label = self.pretrain_config['universal_label']
        self.debug = self.experiment_config["debug"]
        self.learning_rate = self.experiment_config["learning_rate"]

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
        dataloader = PrepareDataLoader(self.pretrain_config, self.experiment_config)
        self.logger.info(f"Loading data ...")
        train_dataloader, valid_dataloader  = dataloader.load_pretrain_data(labels = self.universal_label, model_type = "baseline")
        if not self.output_dir.is_dir():
            self.logger.warning(f"Parent directory {self.output_dir} not found. Creating directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        nb_classes = len(self.universal_label)
    
    def ft_cl(self):
        pass
    
    def ft_mtl(self):
        pass