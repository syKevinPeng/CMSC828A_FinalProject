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
from .utils import convert_label


class Dataloader():
    def __init__(self, preprocessing_config, experiment_config) -> None:
        self.preprocess_config = preprocessing_config
        self.experiment_config = experiment_config
        self.datasets = self.prepare_dataset()
        self.dataset_list = self.experiment_config["dataset"]
    
    def prepare_dataset(self):
        all_dataset = []
        for dataset in self.dataset_list:
            if dataset in ["ES", "extrasensory"]:
                print(f'Checking if Extrasensory is been preprocessed ...')
                preprocessed_file = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f"preprocessed_es_{self.data_orgnize_opt}.csv"
                if not (preprocessed_file).is_file():
                    print("Extrasensory not found. Preprocessing")
                    es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                elif self.experiment_config["force_preprocess"]:
                    print("Extrasensory found but force_preprocess is set to True. Preprocessing")
                    es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                else:
                    print("Extrasensory found. Loading")
                    es_df = pd.read_csv(preprocessed_file)
                all_dataset.append(es_df)
        if len(all_dataset) == 0:
            raise ValueError("No dataset found")
        return pd.concat([i for i in all_dataset], axis = 0)
    
    def preprocess_es(self, save_df, dir):
        es_processor = extrasensory.ExtrasensoryProcessor(self.preprocess_config["extrasensory_preprocessor"])
        es_df = es_processor.preprocess()
        if save_df: es_processor.save_df(es_df, save_dir = dir)
        return es_df
    
    def load_pretrain_data_with_label(self, label, model_type):
        # define sensor to train
        sensor=self.experiment_config["sensor"]
        if model_type == "per_label":
            return self.datasets[['x', 'y', 'z']], self.datasets[label]
        elif model_type == "multitask":
            return self.datasets[['x', 'y', 'z']], self.datasets[label]