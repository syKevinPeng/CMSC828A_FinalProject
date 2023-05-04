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
from tensorflow import keras

# build a data generator
class DataLoader(keras.utils.Sequence):
    def __init__(self,pretrain_config, experiment_config, nb_class, labels, partition) -> None:
        """
        Args:
            pretrain_config (dict): pretrain config
            experiment_config (dict): experiment config
            nb_class (int): number of classes
            labels (list): list of labels
            partition (str): train or valid
        """
        self.preprocess_config = pretrain_config["preprocessing_config"]
        self.experiment_config = experiment_config
        self.dataset_list = self.experiment_config["dataset"]
        self.output_dir =Path(self.experiment_config["output_directory"])/self.experiment_config['exp_name']
        self.logger = get_logger( self.output_dir, "Dataloader")
        self.train_type = self.experiment_config['model_type']
        self.universal_label = pretrain_config['universal_label']
        self.valid_ratio = self.experiment_config['valid_raio']
        self.debug = self.experiment_config['debug']
        self.partition = partition
        if self.partition not in ['train', 'valid']: raise ValueError("Partition must be either train or valid")

        self.batch_size = self.experiment_config['batch_size']
        self.n_channels = 3
        self.n_classes = nb_class
        self.labels = labels
        self.model_type = self.experiment_config['model_type']
        self.train_df, self.valid_df = self.prepare_dataset()


    def prepare_data_split(self, datasets):
        # for each universal label, select 10% of the data: 5% positive, 5% negative
        valid_index = []
        for label in self.universal_label:
            label_index = datasets[datasets[label] == 1].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
            label_index = datasets[datasets[label] == 0].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
        valid_df = datasets.loc[valid_index]
        # remove the valid index from the dataset
        train_df = datasets.drop(valid_index)
        return train_df, valid_df
    
    # get train and valid data. We don't selelct reserved data for validation 
    def prepare_data_split_with_herd(self, datasets, reserved_index):
        # remove the reserved data with index
        es_df = datasets.drop(reserved_index)
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
        return train_df, valid_df
    
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
    
    def prepare_dataset(self):
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
    
        if self.model_type == 'baseline':
            train_df, valid_df = self.prepare_data_split()
        if self.model_type == 'cl':
            self.reserved_df, reserved_index = self.load_reserved_data(self, self.labels)
            self.train_df, self.valid_df = self.prepare_data_split_with_herd(dataset, reserved_index)
        if self.model_type == 'mtl':
            train_df, valid_df = self.prepare_data_split()
        return train_df, valid_df
    
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

    def __len__(self):
        if self.partition == 'train': 
            return int(np.floor(len(self.train_df) / self.batch_size))
        elif self.partition == 'valid':
            return int(np.floor(len(self.valid_df) / self.batch_size))
        else:
            raise ValueError("Partition should be either train or valid")

    # generate a batch of data
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        if self.model_type == 'baseline':
            if self.partition == 'train':
                x, y = self.__data_generation(self.train_df, indexes)