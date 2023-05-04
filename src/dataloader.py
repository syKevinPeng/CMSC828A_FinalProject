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
from utils.utils import get_logger
from pathlib import Path
# set random seed
np.random.seed(123)

class Dataloader():
    def __init__(self, pretrain_config, experiment_config) -> None:
        self.preprocess_config = pretrain_config["preprocessing_config"]
        self.experiment_config = experiment_config
        self.dataset_list = self.experiment_config["dataset"]
        self.output_dir =Path(self.experiment_config["output_directory"])/self.experiment_config['exp_name']
        self.logger = get_logger( self.output_dir, "Dataloader")
        self.train_type = self.experiment_config['model_type']
        self.datasets = self.prepare_dataset()
        self.universal_label = pretrain_config['universal_label']
        self.valid_ratio = self.experiment_config['valid_raio']
    
    def prepare_dataset(self):
        all_dataset = []
        for dataset in self.dataset_list:
            if dataset in ["ES", "extrasensory"]:
                self.logger.info(f'Checking if Extrasensory is been preprocessed ...')
                preprocessed_file = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f"preprocessed_es.csv"
                if not (preprocessed_file).is_file():
                    self.logger.info("Extrasensory not found. Preprocessing")
                    es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                    if self.train_type == 'CL': self.prepare_herd_selection_data(es_df)
                elif self.experiment_config["force_preprocess"]:
                    self.logger.warning("Extrasensory found but force_preprocess is set to True. Preprocessing")
                    es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                    if self.train_type == 'CL': self.prepare_herd_selection_data(es_df)
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
    
    def load_pretrain_data(self, labels:list, model_type):
        if model_type == 'baseline':
            train_df, valid_df = self.prepare_data_split()
            x_train = train_df[['x', 'y', 'z']].to_numpy()
            y_train = train_df[labels].to_numpy()
            x_valid = valid_df[['x', 'y', 'z']].to_numpy()
            y_valid = valid_df[labels].to_numpy()
            return x_train, y_train, x_valid, y_valid
        if model_type == "cl": # output dataset per label
            # get herd selection data
            # load heard_selection data
            herd_data, herd_index = self.load_reserved_data(labels)
            train_df, valid_df = self.prepare_data_split_with_herd(herd_index)
            # select rows where any of the "labels" columns is 1
            train_df = train_df[train_df[labels].any(axis=1)]
            valid_df = valid_df[valid_df[labels].any(axis=1)]
            herd_df = herd_data[herd_data[labels].any(axis=1)]

            # debug
            train_df = train_df.iloc[:1000]
            valid_df = valid_df.iloc[:1000]

            # simple combination strategy: need modificaiton later
            train_df = pd.concat([train_df, herd_df])
            x_train = train_df[['x', 'y', 'z']].to_numpy()
            y_train = train_df[labels].to_numpy()
            x_valid = valid_df[['x', 'y', 'z']].to_numpy()
            y_valid = valid_df[labels].to_numpy()
            return x_train, y_train, x_valid, y_valid
        elif model_type == "MTL": # output all 
            train_df, valid_df = self.prepare_data_split()
            x_train = train_df[['x', 'y', 'z']].to_numpy()
            y_train = train_df[labels].to_numpy()
            x_valid = valid_df[['x', 'y', 'z']].to_numpy()
            y_valid = valid_df[labels].to_numpy()
            return x_train, y_train, x_valid, y_valid
    
    # read the csv reserved data
    def load_reserved_data(self, labels):
        data_path = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f'herd_samples.csv'
        index_path = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f'herd_samples_index.npy'
        if not data_path.is_file(): raise ValueError(f"Reserved data: {data_path} not found. Try to set force_preprocess to True")
        if not index_path.is_file(): raise ValueError(f"index file: {index_path} not found. Try to set force_preprocess to True")
        reserved_df = pd.read_csv(data_path)
        reserved_index = np.load(index_path)
        return reserved_df, reserved_index

    
    # get herd selection data
    def prepare_herd_selection_data(self, es_df):
        output_dir = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])
        reserved_df, reserved_index = extrasensory.herd_selection(es_df, output_dir, logger = self.logger)
        reserved_df.to_csv(output_dir/'herd_samples.csv', index=False)
        np.save(output_dir/'herd_samples_index.npy', reserved_index)
        self.logger.info(f'herd select data is saved to {output_dir}/herd_samples.csv')
        self.logger.info(f'herd select data index is saved to {output_dir}/herd_samples_index.npy')
        return reserved_df, reserved_index

    # get train and valid data. We don't selelct reserved data for validation 
    def prepare_data_split_with_herd(self, reserved_index):
        # remove the reserved data with index
        es_df = self.datasets.drop(reserved_index)
        # for each universal label, select 10% of the data: 5% positive, 5% negative
        valid_index = []
        for label in self.universal_label:
            label_index = es_df[es_df[label] == 1].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
            label_index = es_df[es_df[label] == 0].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
        valid_df = es_df.loc[valid_index]
        # remove the valid index from the dataset
        train_df = es_df.drop(valid_index)
        # x_train = train_df[['x', 'y', 'z']]
        # y_train = train_df[self.universal_label]
        # x_valid = valid_df[['x', 'y', 'z']]
        # y_valid = valid_df[self.universal_label]
        return train_df, valid_df
    
    # get train and valid data without herd selection
    def prepare_data_split(self):
        # for each universal label, select 10% of the data: 5% positive, 5% negative
        valid_index = []
        for label in self.universal_label:
            label_index = self.datasets[self.datasets[label] == 1].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
            label_index = self.datasets[self.datasets[label] == 0].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
        valid_df = self.datasets.loc[valid_index]
        # remove the valid index from the dataset
        train_df = self.datasets.drop(valid_index)
        # x_train = train_df[['x', 'y', 'z']]
        # y_train = train_df[self.universal_label]
        # x_valid = valid_df[['x', 'y', 'z']]
        # y_valid = valid_df[self.universal_label]
        return train_df, valid_df


