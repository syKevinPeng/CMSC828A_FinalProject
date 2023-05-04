import pathlib, datetime
from dataloader import PrepareDataLoader
from model import inception
from utils.utils import get_logger
import numpy as np


class Trainer:
    def __init__(self, experiment_config, pretrain_config):
        self.experiment_config = experiment_config
        self.pretrain_config = pretrain_config
        self.output_dir = pathlib.Path(self.experiment_config["output_directory"])/self.experiment_config['exp_name']
        self.logger = get_logger(self.output_dir, "Trainer")
        self.model_type = self.experiment_config["model_type"]
        if self.model_type not in ['baseline', 'cl', 'mtl']:
            raise ValueError(f"Model type {self.model_type} not supported")
        self.universal_label = self.pretrain_config['universal_label']
    
    def train(self):
        if self.model_type in ['bl', 'baseline', 'Baseline']:
            self.train_baseline()
        elif self.model_type in ['cl', 'CL','ContinueLearning']:
            self.train_cl()
        elif self.model_type in ['mtl', 'MTL','MultitaskLearning']:
            self.train_mtl()
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

    # training code for baseline model
    def train_baseline(self):
        batch_size = self.experiment_config["batch_size"]
        nb_epochs = self.experiment_config["training_epochs"]
        verbose = self.experiment_config["verbose"]
        # load data
        dataloader = PrepareDataLoader(self.pretrain_config, self.experiment_config)
        self.logger.info(f"Loading data ...")
        train_dataloader, valid_dataloader  = dataloader.load_pretrain_data(labels = self.universal_label, model_type = "baseline")
        if not self.output_dir.is_dir():
            self.logger.warning(f"Parent directory {self.output_dir} not found. Creating directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        nb_classes = len(self.universal_label)

        input_shape =(3,1)
        model = inception.Classifier_INCEPTION(self.output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False)
        self.logger.info("---- Start training ----") 
        model.fit(train_dataloader, valid_dataloader)
        self.logger.info("---- End training ----")

    def train_cl(self):
        # initialize dataloader
        dataloader = PrepareDataLoader(self.pretrain_config, self.experiment_config)
        all_labels = self.pretrain_config['universal_label']
        batch_size = self.experiment_config["batch_size"]
        nb_epochs = self.experiment_config["training_epochs"]
        verbose = self.experiment_config["verbose"]

        seen_label = np.array([])
        # ---- first iter -----
        self.logger.info(f"---- Begin 1st iteration ----")
        # sedentary_sitting_other and sedentary_lying are two labels get train first as they have the most number of data
        seen_label = np.append(seen_label, ['sedentary_sitting_other', 'sedentary_lying'])
        # get train data for the first two labels
        self.logger.info(f"Loading data of label sedentary_sitting_other and sedentary_lying ...")
        train_df, valid_df = dataloader.load_pretrain_data(labels = seen_label, model_type = 'cl')
       
        if not self.output_dir.is_dir():
            self.logger.warning(f"Parent directory {self.output_dir} not found. Creating directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"---- Start training labels: {seen_label}----") 
        nb_classes = len(seen_label)

        # add channel dimension if needed
        input_shape =(3,1)
        output_dir = self.output_dir/"-".join(seen_label)
        model = inception.Classifier_INCEPTION(output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False)
        model.fit(train_df, valid_df)
        self.logger.info(f"---- End training : {seen_label}----")
        self.logger.info(f"---- End 1st iteration ----")


        # ---- second and following iter -----
        self.logger.info(f"---- Begin 2nd and following iteration ----")
        # get unseen label
        unseen_label = np.setdiff1d(all_labels, seen_label)
        for label in unseen_label:
            self.logger.info(f"---- Start training labels: {seen_label}----") 
            nb_classes = len(seen_label)
            weights_path = self.output_dir/"-".join(seen_label)/'last_model.hdf5'
            seen_label = np.append(seen_label, label)
            # load previously saved model:
            model = inception.Classifier_INCEPTION(output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                batch_size=batch_size,
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False)
            model.load_model_from_weights(weights_path)
            # get train data for the seen labels
            self.logger.info(f"Loading data of label {seen_label} ...")
            train_df, valid_df= dataloader.load_pretrain_data(labels = seen_label, model_type = 'cl')
            nb_classes = len(seen_label)
            # add channel dimension if needed
            input_shape =(3,1)
            # construct output file
            output_dir = self.output_dir/"-".join(seen_label)
            model = inception.Classifier_INCEPTION(output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False)
            model.fit(train_df, valid_df)
            self.logger.info(f"---- End training : {seen_label}----")
        self.logger.info(f"---- End 2nd and following iteration ----")
        self.logger.info("---- End training ----")


    
    # TODO train multitask learning
    def train_mtl(self):
        raise NotImplementedError





