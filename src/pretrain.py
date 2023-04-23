import pathlib, datetime
from dataloader import Dataloader


class Trainer:
    def __init__(self, experiment_config, pretrain_config):
        self.experiement_config = experiment_config
        self.pretrain_config = pretrain_config
    
    def train(self):
        # load data
        dataloader = Dataloader(self.pretrain_config, self.experiment_config)
        x, y = dataloader.load_pretrain_data_with_label(label = self.pretrain_config["label"])


