import pandas as pd
import numpy as np

import tensorboard as tf
import sys

from regex import F
sys.path.insert(0,'../preprocess')
from preprocess import extrasensory
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from utils.utils import get_logger
from pathlib import Path
from utils.utils import get_logger


class Dataloader():
    def __init__(self, pretrain_config, experiment_config) -> None:
        self.preprocess_config = pretrain_config["preprocessing_config"]
        self.experiment_config = experiment_config
        self.dataset_list = self.experiment_config["dataset"]
        self.logger = get_logger("Dataloader")
        self.datasets = self.prepare_dataset()
    
    def prepare_dataset(self):
        all_dataset = []
        for dataset in self.dataset_list:
            if dataset in ["ES", "extrasensory"]:
                self.logger.info(f'Checking if Extrasensory is been preprocessed ...')
                preprocessed_file = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f"preprocessed_es.csv"
                if not (preprocessed_file).is_file():
                    self.logger.info("Extrasensory not found. Preprocessing")
                    es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                elif self.experiment_config["force_preprocess"]:
                    self.logger.warning("Extrasensory found but force_preprocess is set to True. Preprocessing")
                    es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                else:
                    self.logger.info("Extrasensory found. Loading")
                    es_df = pd.read_csv(preprocessed_file)
                dataset = es_df
            else:
                self.logger.error(f"Dataset {dataset} not found")
                raise ValueError("Dataset not found")
        return dataset
    
    def preprocess_es(self, save_df, dir):
        es_processor = extrasensory.ExtrasensoryProcessor(self.preprocess_config["extrasensory_preprocessor"])
        es_df = es_processor.preprocess()
        dir = Path(dir)
        # save the preprocessed data
        if not dir.parent.is_dir(): 
            self.logger.warning(f"Parent directory {Path(dir).parent} not found. Creating directory")
            dir.parent.mkdir(parents=True, exist_ok=True)
        if save_df: es_processor.save_df(es_df, save_dir = dir)
        return es_df
    
    def load_pretrain_data(self, model_type, label = None):
        # define sensor to train
        if model_type == "CL": # output dataset per label
            if label is None: raise ValueError("Please specify the label")
            return self.datasets[['x', 'y', 'z']], self.datasets[label]
            # return tf.data.Dataset.from_tensor_slices((self.datasets[['x', 'y', 'z']], self.datasets[label]))
        elif model_type == "MTL": # output all 
            return self.datasets[['x', 'y', 'z']], self.datasets