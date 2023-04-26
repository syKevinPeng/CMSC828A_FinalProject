import pathlib, datetime
from dataloader import Dataloader
from model import inception
from sklearn.model_selection import train_test_split
from utils.utils import get_logger
import numpy as np

class Trainer:
    def __init__(self, experiment_config, pretrain_config):
        self.experiment_config = experiment_config
        self.pretrain_config = pretrain_config
        self.logger = get_logger("Trainer")
    
    def train(self):
        # load data
        dataloader = Dataloader(self.pretrain_config, self.experiment_config)
        all_labels = self.pretrain_config['universal_label']
        self.logger.info(f"Loading data ...")
        X, y = dataloader.load_pretrain_data(label = all_labels, model_type = self.pretrain_config["model_type"])
        # convert to numpy array
        X = X.to_numpy()
        y = y.to_numpy()
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
        # convert one-hot to label
        y_true = np.argmax(y_test, axis=1)
        # TODO: Split data into balanced train and validation
        output_directory = pathlib.Path(self.experiment_config["output_directory"])
        if not output_directory.is_dir():
            self.logger.warning(f"Parent directory {output_directory} not found. Creating directory")
            output_directory.mkdir(parents=True, exist_ok=True)
        nb_classes = len(all_labels)
        batch_size = self.experiment_config["batch_size"]
        nb_epochs = self.experiment_config["training_epochs"]
        # add channel dimension if needed
        if len(X_train.shape) == 2: 
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        input_shape =X_train.shape[1:]
        model = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                                verbose=True, 
                                                                build=True, 
                                                                batch_size=batch_size,
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False)
        self.logger.info("---- Start training ----") 
        model.fit(X_train, y_train, X_test, y_test, y_true)
        self.logger.info("---- End training ----")



