# Modified based on https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

# resnet model
import pathlib
from sklearn.metrics import f1_score
from tensorflow import keras
import numpy as np
import time
import tensorflow as tf
from utils.inception_utils import save_logs
from utils.inception_utils import calculate_metrics
from utils.inception_utils import save_test_duration
from utils.utils import TrainingCallback, MTLTrainingCallback, get_logger
import tensorflow_addons as tfa
from pathlib import Path
class MTL_Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=1,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500, lr=0.0001,):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.logger = get_logger(self.output_directory, "INCEPTION")
        self.lr = lr
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.logger.info(self.model.summary())
            self.verbose = verbose
            self.model.save_weights(self.output_directory / 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        self.logger.info('Building model ...')
        # model archtecture
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        # output_layer = keras.layers.Dense(nb_classes, activation='relu')(gap_layer)

        heads = []
        for i in range(nb_classes):
            head = keras.layers.Dense(1, activation="sigmoid", name=f"head{i+1}")(gap_layer)
            heads.append(head)
        #output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=heads)

        #define loss dictionary
        loss_dict={}
        for i in range(nb_classes):
            loss_dict[f"head{i+1}"]='binary_crossentropy'
        
        # define the metrics.
        def get_metrics(name):
            metrics = [
            tf.keras.metrics.Precision(name=f'{name}_precision'),
            tf.keras.metrics.Recall(name=f'{name}_recall'),
            tf.keras.metrics.CategoricalAccuracy(name=f"{name}_accuracy"),
            tfa.metrics.F1Score(num_classes=1, average='macro', name=f'{name}_f1_score')
            ]
            return metrics

        metrics_dict = {}
        for i in range(nb_classes):
            metrics_dict[f"head{i+1}"] = get_metrics(f"head{i+1}")
            
        # define loss function
        model.compile(loss=loss_dict, optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=metrics_dict)
        # don't need to modify anything down below
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)
        weight_format = 'epoch-{epoch:02d}.h5'
        file_path = self.output_directory / weight_format

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=False)
        my_callback = MTLTrainingCallback(self.output_directory, "Training", metrics_dict=metrics_dict)
        self.callbacks = [reduce_lr, model_checkpoint, my_callback]

        return model

    def fit(self, train_dataloader, valid_dataloader, plot_test_acc=False, save_log = False):
        if not tf.test.is_built_with_cuda():
            raise Exception('error no gpu')
        start_time = time.time()
        hist = self.model.fit(x=train_dataloader, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=valid_dataloader, callbacks=self.callbacks)
        duration = time.time() - start_time
        self.logger.info(f"==== Training time: {duration} seconds ====")
        self.model.save(self.output_directory /'last_model.hdf5')
        keras.backend.clear_session()

        return self.model

        
    def load_model_from_weights(self, weights_path):
        self.model.load_weights(weights_path.as_posix())
        self.logger.info(f"Loading model from weights at {weights_path.as_posix()}")
        return self.model