# validate on trained model's performance
from pathlib import Path
from dataloader import PrepareDataLoader
from model import inception, inception_cl, inception_mtl, KD_loss
from utils.utils import get_logger
import numpy as np
from tensorflow import keras
from utils.utils import EvaluateCallback

class Evaluator:
    def __init__(self, experiment_config, evaluate_config):
        self.experiment_config = experiment_config
        self.evaluate_config = evaluate_config
        self.output_dir = Path(self.experiment_config["output_directory"])/self.experiment_config['exp_name']
        self.logger = get_logger(self.output_dir, "Trainer")
        self.weights_dir = Path(self.experiment_config['weights_dir'])
        self.train_type = self.experiment_config['model_type']
        self.universal_label = self.evaluate_config['universal_label']
        self.verbose = self.experiment_config['verbose']
        self.prep_resources()
    
    def prep_resources(self):
        if not self.weights_dir.is_dir():
            raise ValueError(f"weights_dir {self.weights_dir} not found")
        else:
            self.logger.info(f"Loading weights from {self.weights_dir}")
            all_weights = list(self.weights_dir.glob('*.h5'))
            # sort by epoch
            all_weights.sort(key=lambda x: int(x.stem.split('-')[1]))
            self.logger.info(f"Found {len(all_weights)} weights")
            self.weights = all_weights
    
    def evaluate(self):
        # load data
        dataloader = PrepareDataLoader(self.evaluate_config, self.experiment_config)
        train_dataloader, valid_dataloader  = dataloader.load_pretrain_data(labels = self.universal_label, model_type = self.train_type)
        nb_classes = len(self.universal_label)
        # get model
        input_shape =(3,1)
        ## DO NOT SET BATCH SIZE HERE
        model = inception.Classifier_INCEPTION(self.output_dir, input_shape, nb_classes,
                                                                verbose=self.verbose, 
                                                                build=True, 
                                                                depth = 2,
                                                                use_bottleneck = False
                                                                )
        
        # load model
        for epoch_num, weight_path in enumerate(self.weights):
            loaded_model = model.load_model_from_weights(weight_path)
            # set to evaluate mode
            loaded_model.trainable = False
            my_callback = EvaluateCallback(self.output_dir, "Training", epoch_num)
            loaded_model.evaluate(valid_dataloader, verbose = self.verbose, callbacks = [my_callback])
        self.logger.info("Evaluation complete")   





