import pathlib, datetime
from dataloader import Dataloader
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
        dataloader = Dataloader(self.pretrain_config, self.experiment_config)
        self.logger.info(f"Loading data ...")
        X_train, y_train, X_test, y_test = dataloader.load_pretrain_data(labels = self.universal_label, model_type = "baseline")
        # convert one-hot to label
        y_true = np.argmax(y_test, axis=1)
        if not self.output_dir.is_dir():
            self.logger.warning(f"Parent directory {self.output_dir} not found. Creating directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        nb_classes = len(self.universal_label)
        # add channel dimension if needed
        if len(X_train.shape) == 2: 
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        input_shape =X_train.shape[1:]
        model = inception.Classifier_INCEPTION(self.output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                batch_size=batch_size,
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False)
        self.logger.info("---- Start training ----") 
        model.fit(X_train, y_train, X_test, y_test, y_true)
        self.logger.info("---- End training ----")

    def train_cl(self):
        # initialize dataloader
        dataloader = Dataloader(self.pretrain_config, self.experiment_config)
        all_labels = self.pretrain_config['universal_label']
        batch_size = self.experiment_config["batch_size"]
        nb_epochs = self.experiment_config["training_epochs"]
        verbose = self.experiment_config["verbose"]

        seen_label = np.array([])
        # ---- first iter -----
        # sedentary_sitting_other and sedentary_lying are two labels get train first as they have the most number of data
        seen_label = np.append(seen_label, ['sedentary_sitting_other', 'sedentary_lying'])
        # get train data for the first two labels
        self.logger.info(f"Loading data of label sedentary_sitting_other and sedentary_lying ...")
        X_train, y_train, X_test, y_test = dataloader.load_pretrain_data(labels = seen_label, model_type = 'cl')
       
        # convert one-hot to label
        y_true = np.argmax(y_test, axis=1)
        if not self.output_dir.is_dir():
            self.logger.warning(f"Parent directory {self.output_dir} not found. Creating directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        nb_classes = len(seen_label)

        # add channel dimension if needed
        if len(X_train.shape) == 2: 
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        input_shape =X_train.shape[1:]
        output_dir = self.output_dir/"_".join(seen_label)
        model = inception.Classifier_INCEPTION(output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                batch_size=batch_size,
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False)
        self.logger.info(f"---- Start training labels: {seen_label}----") 
        model.fit(X_train, y_train, X_test, y_test, y_true)
        self.logger.info(f"---- End training : {seen_label}----")

        # ---- second and following iter -----
        # get unseen label
        unseen_label = np.setdiff1d(all_labels, seen_label)
        for label in unseen_label:
            nb_classes = len(seen_label)
            weights_path = self.output_dir/seen_label/'last_model.hdf5'
            seen_label = np.append(seen_label, label)
            # load previously saved model:
            model = inception.Classifier_INCEPTION(self.output_dir, input_shape, nb_classes, build=True)
            print(weights_path)
            model.load_model_from_weights(weights_path)
            # get train data for the seen labels
            self.logger.info(f"Loading data of label {seen_label} ...")
            X_train, y_train, X_test, y_test= dataloader.load_pretrain_data(labels = seen_label, model_type = 'cl')
            X_reserved, y_reserved = dataloader.load_reserved_data(label = seen_label)
            # combine new data with previous data
            X_train = np.concatenate((X_train, X_reserved), axis=0)
            y_train = np.concatenate((y_train, y_reserved), axis=0)

            # split data
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=42)
            # convert one-hot to label
            y_true = np.argmax(y_test, axis=1)
            nb_classes = len(seen_label)
            # add channel dimension if needed
            if len(X_train.shape) == 2: 
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            input_shape =X_train.shape[1:]
            # construct output file
            output_dir = self.output_dir/"_".join(seen_label)
            model = inception.Classifier_INCEPTION(output_dir, input_shape, nb_classes,
                                                                    verbose=verbose, 
                                                                    build=True, 
                                                                    batch_size=batch_size,
                                                                    nb_epochs = nb_epochs,
                                                                    use_bottleneck = False)
            self.logger.info(f"---- Start training labels: {seen_label}----") 
            model.fit(X_train, y_train, X_test, y_test, y_true)
            self.logger.info(f"---- End training : {seen_label}----")


    
    # TODO train multitask learning
    def train_mtl(self):
        raise NotImplementedError





