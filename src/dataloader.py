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
        self.output_dir =Path(self.experiment_config["output_directory"])/self.experiment_config['exp_name']
        self.logger = get_logger( self.output_dir, "Dataloader")
        self.train_type = self.experiment_config['model_type']
        self.datasets = self.prepare_dataset()
        self.universal_label = self.experiment_config['universal_label']
    
    def prepare_dataset(self):
        all_dataset = []
        for dataset in self.dataset_list:
            if dataset in ["ES", "extrasensory"]:
                self.logger.info(f'Checking if Extrasensory is been preprocessed ...')
                preprocessed_file = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f"preprocessed_es.csv"
                if not (preprocessed_file).is_file():
                    self.logger.info("Extrasensory not found. Preprocessing")
                    es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                    if self.train_type == 'CL': self.prepare_hear_selection_data(es_df)
                elif self.experiment_config["force_preprocess"]:
                    self.logger.warning("Extrasensory found but force_preprocess is set to True. Preprocessing")
                    es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                    if self.train_type == 'CL': self.prepare_hear_selection_data(es_df)
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
            # convert dataframe to numpy
            feature = self.datasets[['x', 'y', 'z']].to_numpy()
            labels = self.datasets[label].to_numpy()
            return feature, labels
            # return tf.data.Dataset.from_tensor_slices((self.datasets[['x', 'y', 'z']], self.datasets[label]))
        elif model_type == "MTL": # output all 
            return self.datasets[['x', 'y', 'z']], self.datasets
    
    # read the csv reserved data
    def load_reserved_data(self, labels):
        file_dir = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f"reserved_es.csv"
        if not file_dir.is_file(): raise ValueError(f"Reserved data: {file_dir} not found")
        feature = self.datasets[['x', 'y', 'z']].to_numpy()
        labels = self.datasets[labels].to_numpy()
        return feature, labels

    
    # get herd selection data
    def prepare_hear_selection_data(self, es_df):
        output_dir = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/'herd_samples.csv'
        extrasensory.herd_selection(es_df, output_dir, logger = self.logger)
