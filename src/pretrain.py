import pathlib, datetime
from dataloader import Dataloader
from model import inception
from sklearn.model_selection import train_test_split
from utils.utils import get_logger

class Trainer:
    def __init__(self, experiment_config, pretrain_config):
        self.experiment_config = experiment_config
        self.pretrain_config = pretrain_config
        self.logger = get_logger("Trainer")
    
    def train(self):
        # load data
        dataloader = Dataloader(self.pretrain_config, self.experiment_config)
        all_labels = self.pretrain_config['universal_label']
        X, y = dataloader.load_pretrain_data(label = all_labels, model_type = self.pretrain_config["model_type"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
        # TODO: Split data into balanced train and validation
        output_directory = pathlib.Path(self.experiment_config["output_directory"])
        input_shape = self.pretrain_config["input_shape"]
        nb_classes = len(all_labels)
        batch_size = self.experiment_config["batch_size"]
        nb_epochs = self.experiment_config["training_epochs"]


        model = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                                verbose=False, build=True, batch_size=batch_size,nb_epochs = nb_epochs)
        self.logger.info("---- Start training ----")        
        model.fit(X_train, X_test, y_train, y_test)
        self.logger.info("---- End training ----")



