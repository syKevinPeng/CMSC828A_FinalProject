# Modified based on https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

# resnet model
from math import log
import pathlib
from sklearn.metrics import f1_score
from tensorflow import keras
import numpy as np
import time
import tensorflow as tf
from utils.inception_utils import save_logs
from utils.inception_utils import calculate_metrics
from utils.inception_utils import save_test_duration
from utils.utils import TrainingCallback
import tensorflow_addons as tfa
from utils.utils import get_logger
from pathlib import Path
from .cosine_norm import CosineLinear
from .KD_loss import my_kd_loss

class InceptionWithCL:

    def __init__(self, output_directory, input_shape, nb_classes,verbose=False, build=True, batch_size=1,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500, add_CN=False, lr = 0.0001):
        '''
        :param output_directory: directory to save the model
        :param input_shape: [8, 3, 1]
        :param nb_classes
        :param verbose:
        :param build: if True, build model
        :param batch_size: should always to be 1. We supply a batch of data into the model
        :param nb_filters: number of filters in the convolutional layers
        :param use_residual: if True, use residual connection
        :param use_bottleneck: if True, use bottleneck architecture
        :param depth: number of inception modules per block
        :param kernel_size: kernel size of the convolutional layers
        :param nb_epochs: number of epochs
        :param add_CN: if True, add cosine normalization layer
        '''

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
        self.nb_classes = nb_classes
        self.lr = lr
        if build == True:
            self.model = self.build_model(input_shape, self.nb_classes, add_CN)
            if (verbose == True):
                self.logger.info(self.model.summary())
            self.verbose = verbose
            # self.model.save_weights(self.output_directory / 'model_init.hdf5')

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

    def build_model(self, input_shape, nb_classes, add_cosine_layer = False):
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

        if add_cosine_layer:
            output_layer = CosineLinear(in_features=gap_layer.shape[-1], 
                                    out_features = nb_classes)(gap_layer)
        else:
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        # define the metrics.
        metrics = [
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tfa.metrics.F1Score(num_classes=nb_classes, average='macro', name='f1_score')
        ]
        loss_func = tf.keras.losses.CategoricalCrossentropy()
        # define loss function
        model.compile(loss=loss_func, optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=metrics)
        # don't need to modify anything down below
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)
        weight_format = 'epoch-{epoch:02d}-val_acc-{val_accuracy:.4f}-train_acc-{accuracy:.4f}-precision-{precision:.4f}-recall-{recall:.4f}.h5'
        file_path = self.output_directory / weight_format

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=False)
        my_callback = TrainingCallback(self.output_directory, "Training")
        self.callbacks = [reduce_lr, model_checkpoint, my_callback]

        return model

    def fit(self, train_data_generator, valid_data_generator):
        '''
        train_data_generator: a generator that generates training data
        valid_data_generator: a generator that generates validation data
        '''

        if not tf.test.is_built_with_cuda():
            raise Exception('error no gpu')
        # x_val and y_val are only used to monitor the test loss and NOT for training


        start_time = time.time()
        hist = self.model.fit(x=train_data_generator, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=valid_data_generator, callbacks=self.callbacks)
        duration = time.time() - start_time
        self.logger.info(f"Training time: {duration} seconds")
        self.model.save(self.output_directory /f'last_model.hdf5')
        self.logger.info(f'current weights saved at {self.output_directory /"last_model.hdf5"}')

        keras.backend.clear_session()

        return self.model
    
    def fit_kd(self, train_data_generator, valid_data_generator, prev_model):
        # remove the last layer of the model
        gap_layer = self.model.layers[-2].output
        output_layer = CosineLinear(in_features=gap_layer.shape[-1], 
                                    out_features = self.nb_classes)(gap_layer)
        output_layer = keras.layers.Activation('softmax')(output_layer)
        
        self.model = keras.models.Model(inputs=self.model.input, outputs=output_layer)
        if not tf.test.is_built_with_cuda():
            raise Exception('error no gpu')
        
        # kd_loss:
        # kd_loss = KDLoss(prev_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        logs = {}
        # train the student model
        for epoch in range(self.nb_epochs):
            # define metrics
            train_loss_avg = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            train_recall = tf.keras.metrics.Recall(name='recall')
            train_precision = tf.keras.metrics.Precision(name='precision')
            train_f1 = tfa.metrics.F1Score(num_classes=self.nb_classes, average='macro', name='f1_score')
            def update_train_metrics(y_true, y_pred):
                train_loss_avg.update_state(y_true, y_pred)
                train_accuracy.update_state(y_true, y_pred)
                train_recall.update_state(y_true, y_pred)
                train_precision.update_state(y_true, y_pred)
                train_f1.update_state(y_true, y_pred)

            # for _, (x_batch, y_batch) in enumerate(train_data_generator):
            for (x_batch, y_batch) in (train_data_generator):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch, training=True)
                    # loss_value = my_kd_loss(y_true=y_batch, y_pred=logits, inputs = x_batch, teacher_model=prev_model)
                    loss_value = tf.keras.losses.categorical_crossentropy(y_batch, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                # update metrics
                update_train_metrics(y_batch, logits)


            #  -------------------------
            # Validation loop
            val_loss_avg = tf.keras.metrics.Mean(name='train_loss')
            val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            val_recall = tf.keras.metrics.Recall(name='recall')
            val_precision = tf.keras.metrics.Precision(name='precision')
            val_f1 = tfa.metrics.F1Score(num_classes=self.nb_classes, average='macro', name='f1_score')
            def update_val_metrics(y_true, y_pred):
                val_loss_avg.update_state(y_true, y_pred)
                val_accuracy.update_state(y_true, y_pred)
                val_recall.update_state(y_true, y_pred)
                val_precision.update_state(y_true, y_pred)
                val_f1.update_state(y_true, y_pred)

            for x_val, y_val in valid_data_generator:
                val_logits = self.model(x_val, training=False)
                loss_value = my_kd_loss(y_true=y_val, y_pred=val_logits, inputs = x_val, teacher_model=prev_model)
                # update metrics
                update_val_metrics(y_val, val_logits)
            logs[f'epoch:{epoch}'] = {
                'train_loss': train_loss_avg.result().numpy(),
                'train_accuracy': train_accuracy.result().numpy(),
                'train_recall': train_recall.result().numpy(),
                'train_precision': train_precision.result().numpy(),
                'train_f1': train_f1.result().numpy(),
                'val_loss': val_loss_avg.result().numpy(),
                'val_accuracy': val_accuracy.result().numpy(),
                'val_recall': val_recall.result().numpy(),
                'val_precision': val_precision.result().numpy(),
                'val_f1': val_f1.result().numpy()
            }
            self.on_epoch_end(epoch, logs)
        self.model.save(self.output_directory / 'last_model.hdf5')
        tf.keras.backend.clear_session()
        return self.model
    
    def on_epoch_end(self, epoch, logs):
        # get the metrics
        metrics = logs[f'epoch:{epoch}']
        weight_format = f"epoch-{epoch:02d}-acc-{metrics['val_accuracy']:.4f}-precision-{metrics['val_precision']:.4f}-recall-{metrics['val_recall']:.4f}.h5"
        file_path = self.output_directory / weight_format
        self.model.save(file_path)
        message = [f'{metrics}: {logs[f"epoch:{epoch}"][metrics]}' for metrics in logs[f'epoch:{epoch}']]
        self.logger.info(f"Epoch {epoch} : " + " - ".join(message))
    
    def load_model_from_weights(self, weights_path):
        self.logger.info(f"Loading model from weights at {weights_path.as_posix()}")
        self.model.load_weights(weights_path.as_posix())
        return self.model