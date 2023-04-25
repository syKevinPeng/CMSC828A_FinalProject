import pathlib, datetime
from dataloader import Dataloader
from model import inception


class Trainer:
    def __init__(self, experiment_config, pretrain_config):
        self.experiment_config = experiment_config
        self.pretrain_config = pretrain_config
    
    def train(self):
        # load data
        dataloader = Dataloader(self.pretrain_config, self.experiment_config)
        all_labels = self.pretrain_config['universal_label']
        x, y = dataloader.load_pretrain_data(label = all_labels[0], model_type = self.pretrain_config["model_type"])
        # TODO: Split data into train and validation
        output_directory = pathlib.Path(self.experiment_config["output_directory"])
        nb_classes = self.pretrain_config["nb_classes"]
        input_shape = self.pretrain_config["input_shape"]
        model = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                                verbose=False, build=True)



