import pathlib, datetime
from dataloader import PrepareDataLoader
from model import inception, inception_cl, inception_mtl, KD_loss
from utils.utils import get_logger
import numpy as np
from tensorflow import keras


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
        self.debug = self.experiment_config["debug"]
        self.learning_rate = self.experiment_config["learning_rate"]
        self.load_weights = self.experiment_config["load_weights"]
        if self.load_weights:
            self.weights_file = self.experiment_config["weights_file"]
    
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
        ## DO NOT SET BATCH SIZE HERE
        model = inception.Classifier_INCEPTION(self.output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                depth = 2,
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False,
                                                                lr = self.learning_rate
                                                                )
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

        if self.weights_file:
            self.logger.info(f"Loading weights from {self.experiment_config['cl_weights_file']}")
            prev_model = keras.models.load_model(self.experiment_config['cl_weights_file'])
            prev_model.trainable = False
            seen_label = ['sedentary_sitting_other', 'upright_standing']
        else:
            # ---- first iter -----
            print('-'*10)
            self.logger.info(f"Begin 1st iteration")
            # sedentary_sitting_other and sedentary_lying are two labels get train first as they have the most number of data
            seen_label = ['sedentary_sitting_other', 'upright_standing']
            # get train data for the first two labels
            self.logger.info(f"Loading data of label sedentary_sitting_other and upright_standing ...")
            train_generator, valid_generator = dataloader.load_pretrain_data(
                                                                                        labels = self.universal_label, 
                                                                                        model_type = 'cl', 
                                                                                        new_class=seen_label)
            if self.debug:
                for x, y in train_generator:
                    self.logger.info(f'first iter: x shape: {x.shape}, y shape: {y.shape}')
                    break
        
            if not self.output_dir.is_dir():
                self.logger.warning(f"Parent directory {self.output_dir} not found. Creating directory")
                self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Start training labels: {seen_label}") 
            nb_classes = len(seen_label)

            # add channel dimension if needed
            input_shape =(3,1)
            output_dir = self.output_dir/"-".join(seen_label)
            model = inception_cl.InceptionWithCL(output_dir, input_shape, nb_classes = len(self.universal_label), # force the model to predict all labels
                                                                    verbose=verbose, 
                                                                    build=True, 
                                                                    nb_epochs = nb_epochs,
                                                                    use_bottleneck = False,
                                                                    add_CN = False) # do not use cosline normalziaton in first iter
            prev_model = model.fit(train_generator, valid_generator)
            self.logger.info(f"End training : {seen_label}")
            self.logger.info(f"End 1st iteration")
        # ---- second and following iter -----
        self.logger.info(f"Begin 2nd and following iteration:")
        # get unseen label
        # unseen_label = np.setdiff1d(all_labels, seen_label)
        unseen_label = ['upright_stepping']
        dataloader = PrepareDataLoader(self.pretrain_config, self.experiment_config)
        for label in unseen_label:
            print('-'*10)
            self.logger.info(f"Start training labels: {np.append(seen_label,label)}") 
            nb_classes = len(seen_label)
            prev_weights_path = self.output_dir/"-".join(seen_label)/'last_model.hdf5'
            input_shape =(3,1)
            # construct output file
            _add_CN = False if len(seen_label) == 2 else True
            seen_label = np.append(seen_label, label)
            output_dir = self.output_dir/"-".join(seen_label)  
            model = inception_cl.InceptionWithCL(output_dir, input_shape, nb_classes = len(self.universal_label), # force the model to predict all labels
                                                                verbose=verbose, 
                                                                build=True,
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False,
                                                                add_CN = _add_CN,
                                                                lr=self.learning_rate) # use cosline normalziaton in second and following iter
            model.load_model_from_weights(prev_weights_path)
            self.logger.info(f"Loading data of label {seen_label} ...")
            train_df, valid_df= dataloader.load_pretrain_data(
                                                                                    labels = seen_label, 
                                                                                    model_type = 'cl', 
                                                                                    new_class = [label])
            if self.debug:
                for x, y in train_generator:
                    self.logger.info(f'second and following iter: x shape: {x.shape}, y shape: {y.shape}')
                    break
            prev_model = model.fit_kd(train_df, valid_df, prev_model)
            self.logger.info(f"End training : {seen_label}")
        self.logger.info("---- End training ----")



    
    # TODO train multitask learning  Temporry using baseling to test
    def train_mtl(self):
        batch_size = self.experiment_config["batch_size"]
        nb_epochs = self.experiment_config["training_epochs"]
        verbose = self.experiment_config["verbose"]
        # load data
        dataloader = PrepareDataLoader(self.pretrain_config, self.experiment_config)
        self.logger.info(f"Loading data ...")
        train_dataloader, valid_dataloader  = dataloader.load_pretrain_data(labels = self.universal_label, model_type = "mtl")
        if not self.output_dir.is_dir():
            self.logger.warning(f"Parent directory {self.output_dir} not found. Creating directory")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        nb_classes = len(self.universal_label)
        input_shape =(3,1)

        model = inception_mtl.MTL_Classifier_INCEPTION(self.output_dir, input_shape, nb_classes,
                                                                verbose=verbose, 
                                                                build=True, 
                                                                nb_epochs = nb_epochs,
                                                                use_bottleneck = False,
                                                                lr=self.learning_rate
                                                                )
        if self.load_weights:
                self.logger.info(f"Loading weights from {self.experiment_config['weights_file']}")
                model.model.load_weights(self.experiment_config['weights_file'])
        self.logger.info("---- Start training ----") 
        model.fit(train_dataloader, valid_dataloader)
        self.logger.info("---- End training ----")






