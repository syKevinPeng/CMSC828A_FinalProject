from re import A
import pandas as pd
import numpy as np
from regex import B, D

import tensorflow as tf
import sys
sys.path.insert(0,'../preprocess')
from preprocess import extrasensory, wisdm
from pathlib import Path
import pandas as pd
import numpy as np
from utils.utils import get_logger
from pathlib import Path
# set random seed
np.random.seed(123)
from tensorflow import keras

class PrepareDataLoader():
    def __init__(self, pretrain_config, experiment_config) -> None:
        self.preprocess_config = pretrain_config["preprocessing_config"]
        self.pretrain_config = pretrain_config
        self.experiment_config = experiment_config
        self.dataset_name = self.pretrain_config["dataset"]
        self.output_dir =Path(self.experiment_config["output_directory"])/self.experiment_config['exp_name']
        self.logger = get_logger( self.output_dir, "Dataloader")
        self.train_type = self.experiment_config['model_type']
        self.universal_label = pretrain_config['universal_label']
        self.valid_ratio = self.experiment_config['valid_raio']
        self.debug = self.experiment_config['debug']
        self.datasets = self.prepare_dataset()
        
    
    def prepare_dataset(self):
        if self.dataset_name in ["ES", "extrasensory"]:
            self.logger.info(f'Checking if Extrasensory is been preprocessed ...')
            preprocessed_file = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f"preprocessed_es.csv"
            if not (preprocessed_file).is_file():
                self.logger.info("Extrasensory not found. Preprocessing")
                es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                if self.train_type == 'cl': self.prepare_herd_selection_data_es(es_df)
            elif self.experiment_config["force_preprocess"]:
                self.logger.warning("Extrasensory found but force_preprocess is set to True. Preprocessing")
                es_df = self.preprocess_es(save_df = True, dir = preprocessed_file)
                if self.train_type == 'cl': self.prepare_herd_selection_data_es(es_df)
            else:
                self.logger.info("Extrasensory found. Loading")
                es_df = pd.read_csv(preprocessed_file)
            dataset = es_df
        elif self.dataset_name in ['wisdm', 'WISDM']:
            print(f'Checking if WISDM is been preprocessed ...')
            preprocessed_file = Path(self.preprocess_config["wisdm_preprocessor"]["out"]['dir'])/f"preprocessed_wisdm.csv"
            if not (preprocessed_file).is_file():
                self.logger.info("WISDM not found. Preprocessing")
                wisdm_df = self.preprocess_wisdm(save_df = True, dir = preprocessed_file)
                if self.train_type == 'cl': self.prepare_herd_selection_data_wisdm(wisdm_df)
            elif self.experiment_config["force_preprocess"]:
                self.logger.warning("WISDM found but force_preprocess is set to True. Preprocessing")
                wisdm_df = self.preprocess_wisdm(save_df = True, dir = preprocessed_file)
                if self.train_type == 'cl': self.prepare_herd_selection_data_wisdm(wisdm_df)
            else: # loading dataframe
                self.logger.info("WISDM found. Loading")
                wisdm_df = pd.read_csv(preprocessed_file) 
            dataset = wisdm_df
        else:
            self.logger.error(f"Dataset {self.dataset_name} not found")
            raise ValueError("Dataset not found")
        
        # count how many entries per label
        for label in self.universal_label:
            self.logger.info(f"Number of {label} entries: {dataset[label].sum()}")
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
    
    def preprocess_wisdm(self, save_df, dir):
        processor = wisdm.WisdmProcessor(self.preprocess_config['wisdm_preprocessor'])
        df = processor.preprocess()
        if save_df: processor.save_df(df, save_dir = dir)
        return df

    
    def load_pretrain_data(self, labels:list, model_type, new_class = [None]):
        '''
        param:
        labels: the Y labels to be outputed (i.e. total number of class)
        model_type: specify one of ['baseline', 'cl', 'mtl']
        new_class: ONLY used in continue learning. This is the new class added to the training pipeline. The data generator should only output X in 'newclass', but y in all 'labels'
        '''
        if model_type == 'baseline':
            train_df, valid_df = self.prepare_data_split()
            # Convert data to batches
            dataloader_train = DataLoader(train_df, self.experiment_config, labels)
            dataloader_valid = DataLoader(valid_df, self.experiment_config, labels)

            return dataloader_train, dataloader_valid
        if model_type == "cl": # output dataset per label
            # load heard_selection data
            herd_data, herd_index = self.load_reserved_data(labels)
            train_df, valid_df = self.prepare_data_split_with_herd(herd_index)
            # select rows where "new_class" columns is 1
            train_df = train_df[train_df[new_class].any(axis=1)].reset_index()
            valid_df = valid_df[valid_df[labels].any(axis=1)]
            herd_df = herd_data[herd_data[labels].any(axis=1)]

            # debug
            if self.debug:
                train_df = train_df.iloc[:100]
                valid_df = valid_df.iloc[:100]

            # simple combination strategy: need modificaiton later
            cl_dataloader_train = CLDataLoader(train_df, herd_df, self.experiment_config, self.universal_label)
            cl_dataloader_valid = DataLoader(valid_df, self.experiment_config, self.universal_label)

            return cl_dataloader_train, cl_dataloader_valid
        elif model_type == "mtl":
            train_df, valid_df = self.prepare_data_split()
            # debug
            if self.debug:
                train_df = train_df.iloc[:100]
                valid_df = valid_df.iloc[:100]
            # Convert data to batches  Temporary solution to see if things work
            dataloader_train = MTLDataLoader(train_df, self.experiment_config, labels)
            dataloader_valid = MTLDataLoader(valid_df, self.experiment_config, labels)
            return dataloader_train, dataloader_valid
    
    def load_finetuning_data(self, labels, model_type, new_class = [None]):
        if model_type == 'baseline':
            train_df, valid_df = self.prepare_data_split()
            # Convert data to batches
            dataloader_train = DataLoader(train_df, self.experiment_config, labels)
            dataloader_valid = DataLoader(valid_df, self.experiment_config, labels)
            return dataloader_train, dataloader_valid
        elif model_type == 'cl':
            # load heard_selection data
            herd_data, herd_index = self.load_reserved_data(labels, dataset='es')
            train_df, valid_df = self.prepare_data_split_with_herd(herd_index)
            # select rows where "new_class" columns is 1
            train_df = train_df[train_df[new_class].any(axis=1)].reset_index()
            valid_df = valid_df[valid_df[labels].any(axis=1)]
            herd_df = herd_data[herd_data[labels].any(axis=1)]

            # debug
            if self.debug:
                train_df = train_df.iloc[:100]
                valid_df = valid_df.iloc[:100]

            # simple combination strategy: need modificaiton later
            cl_dataloader_train = CLDataLoader(train_df, herd_df, self.experiment_config, self.universal_label)
            cl_dataloader_valid = DataLoader(valid_df, self.experiment_config, self.universal_label)
            return cl_dataloader_train, cl_dataloader_valid
        elif model_type == 'mtl':
            train_df, valid_df = self.prepare_data_split()
            if self.debug:
                train_df = train_df.iloc[:100]
                valid_df = valid_df.iloc[:100]
            # Convert data to batches  Temporary solution to see if things work
            dataloader_train = MTLDataLoader(train_df, self.experiment_config, labels)
            dataloader_valid = MTLDataLoader(valid_df, self.experiment_config, labels)
            return dataloader_train, dataloader_valid
        

    # read the csv reserved data
    def load_reserved_data(self, labels, dataset = 'wisdm'):
        if dataset == 'wisdm':
            data_path = Path(self.preprocess_config["wisdm_preprocessor"]["out"]['dir'])/f'herd_samples.csv'
            index_path = Path(self.preprocess_config["wisdm_preprocessor"]["out"]['dir'])/f'herd_samples_index.npy'
        elif dataset == 'es':
            data_path = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f'herd_samples_es.csv'
            index_path = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])/f'herd_samples_index_es.npy'
        if not data_path.is_file(): raise ValueError(f"Reserved data: {data_path} not found. Try to set force_preprocess to True")
        if not index_path.is_file(): raise ValueError(f"index file: {index_path} not found. Try to set force_preprocess to True")
        reserved_df = pd.read_csv(data_path)
        reserved_index = np.load(index_path)
        return reserved_df, reserved_index

    
    # get herd selection data
    def prepare_herd_selection_data_wisdm(self, es_df):
        output_dir = Path(self.preprocess_config["wisdm_preprocessor"]["out"]['dir'])
        reserved_df, reserved_index = extrasensory.herd_selection(es_df, output_dir, logger = self.logger)
        reserved_df.to_csv(output_dir/'herd_samples.csv', index=False)
        np.save(output_dir/'herd_samples_index.npy', reserved_index)
        self.logger.info(f'herd select data is saved to {output_dir}/herd_samples.csv')
        self.logger.info(f'herd select data index is saved to {output_dir}/herd_samples_index.npy')
        return reserved_df, reserved_index

    def prepare_herd_selection_data_es(self, es_df):
        output_dir = Path(self.preprocess_config["extrasensory_preprocessor"]["out"]['dir'])
        reserved_df, reserved_index = extrasensory.herd_selection(es_df, output_dir, logger = self.logger)
        reserved_df.to_csv(output_dir/'herd_samples_es.csv', index=False)
        np.save(output_dir/'herd_samples_index_es.npy', reserved_index)
        self.logger.info(f'herd select data is saved to {output_dir}/herd_samples_es.csv')
        self.logger.info(f'herd select data index is saved to {output_dir}/herd_samples_index_es.npy')
        return reserved_df, reserved_index

    # get train and valid data. We don't selelct reserved data for validation 
    def prepare_data_split_with_herd(self, reserved_index):
        # remove the reserved data with index
        # self.datasets = self.datasets.reset_index()
        es_df = self.datasets.drop(reserved_index)
        # for each universal label, select 10% of the data: 5% positive, 5% negative
        valid_index = []
        for label in self.universal_label:
            label_index = es_df[es_df[label] == 1].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
            label_index = es_df[es_df[label] == 0].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
        valid_df = es_df.loc[valid_index]
        train_df = es_df.drop(valid_index)
        return train_df, valid_df
    
    # get train and valid data without herd selection
    def prepare_data_split(self):
        # for each universal label, select 10% of the data: 5% positive, 5% negative
        valid_index = []
        for label in self.universal_label:
            if len(self.datasets[self.datasets[label] == 1]) == 0:
                continue
            label_index = self.datasets[self.datasets[label] == 1].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
            label_index = self.datasets[self.datasets[label] == 0].sample(frac=self.valid_ratio/2).index
            valid_index.extend(label_index)
        valid_df = self.datasets.loc[valid_index]
        # remove the valid index from the dataset
        train_df = self.datasets.drop(valid_index)
        return train_df, valid_df

# dataloader for baseline training
class DataLoader(keras.utils.Sequence):
    def __init__(self, input_df:pd.DataFrame, experiment_config, labels):
        self.input_df = input_df
        self.batch_size = experiment_config['batch_size']
        self.labels = labels
        self.indexes = np.arange(len(self.input_df))
    
    def __len__(self):
        return int(np.floor(len(self.input_df)/self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.input_df.iloc[indexes]
        x = batch_df[['x', 'y', 'z']].to_numpy()
        y = batch_df[self.labels].to_numpy()
        if len(x.shape) == 2: 
            x = x.reshape((x.shape[0], x.shape[1], 1))
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        return x, y

# dataloader for baseline training
class MTLDataLoader(keras.utils.Sequence):
    def __init__(self, input_df:pd.DataFrame, experiment_config, labels):
        self.input_df = input_df
        self.batch_size = experiment_config['batch_size']
        self.labels = labels
        self.indexes = np.arange(len(self.input_df))
    
    def __len__(self):
        return int(np.floor(len(self.input_df)/self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.input_df.iloc[indexes]
        x = batch_df[['x', 'y', 'z']].to_numpy()
        y = batch_df[self.labels].to_numpy(dtype=np.float32)
        if len(x.shape) == 2: 
            x = x.reshape((x.shape[0], x.shape[1], 1))
        y = [y[:, i].reshape(-1,1) for i in range(len(y[0]))]
        names = [f"head{i+1}" for i in range(len(y))]
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        y = dict(zip(names, y))
        return x, y
    
# dataloader used for CL training, specifically it merge the reserved data and the current batch data
class CLDataLoader(keras.utils.Sequence):
    def __init__(self, input_df:pd.DataFrame, herd_df:pd.DataFrame,experiment_config, labels):
        self.input_df = input_df
        self.herd_df = herd_df.reset_index()
        self.reserve_indexes = np.arange(len(herd_df))
        self.total_reserve_len = len(herd_df)
        self.batch_size = experiment_config['batch_size']
        self.labels = labels
        self.indexes = np.arange(len(self.input_df))
    
    def __len__(self):
        return int(np.floor(len(self.input_df)/self.batch_size))
    
    def __getitem__(self, index):
        reserve_batch_size = 2
        old_batch_size = self.batch_size-reserve_batch_size
        old_indexes = self.indexes[index*old_batch_size:(index+1)*old_batch_size]
        curr_reserved_ind = index%self.total_reserve_len
        reserve_indexes = self.reserve_indexes[curr_reserved_ind*reserve_batch_size:(curr_reserved_ind+1)*reserve_batch_size]
        x, y = self.__data_generation_train(old_indexes)
        x_reserve, y_reserve = self.__data_generation_herd(reserve_indexes)
        x = np.concatenate((x, x_reserve), axis=0)
        y = np.concatenate((y, y_reserve), axis=0)
        if len(x.shape) == 2: 
            x = x.reshape((x.shape[0], x.shape[1], 1))
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        return x, y
    
    
    def __data_generation_train(self, indexes):
        index = self.input_df.index[indexes]
        x = self.input_df.loc[index, ['x', 'y', 'z']].to_numpy()
        y = self.input_df.loc[index, self.labels].to_numpy()
        return x, y
    def __data_generation_herd(self, indexes):
        index = self.input_df.index[indexes]
        x = self.herd_df.loc[index, ['x', 'y', 'z']].to_numpy()
        y = self.herd_df.loc[index, self.labels].to_numpy()
        return x, y