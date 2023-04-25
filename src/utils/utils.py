import logging
from tensorflow import keras

def get_logger(name, logname="running.log"):
    assert len(name) <= 12, "name should be less than 12 characters"
    formatter = logging.Formatter(
        f"%(asctime)s {name:12} %(levelname)s\t%(message)s"
    )
    logging.basicConfig(
        filename=logname,
        filemode="w",
        format=formatter._fmt or "",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    logger = logging.getLogger(name)
    if not logger.handlers:
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    return logger

class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, name):
        self.logger = get_logger(name)

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(f"Epoch {epoch} - loss: {logs['loss']} - acc: {logs['accuracy']} - F1: {logs['f1_score']}")

    # def on_train_end(self, logs=None):
    #     self.logger.info(f"Training finished. Time elapsed: {logs['time']}")

