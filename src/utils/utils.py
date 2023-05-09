import logging
from tensorflow import keras
from pathlib import Path

def get_logger(output_dir, name,logname="running.log"):
    # check if output_dir is a PAth object
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    logname = output_dir/logname
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    assert len(name) <= 12, "name should be less than 12 characters"
    formatter = logging.Formatter(
        f"%(asctime)s {name:12} %(levelname)s\t%(message)s",
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
    def __init__(self, output_dir, name):
        self.logger = get_logger(output_dir, name)

    def on_epoch_end(self, epoch, logs=None):

        self.logger.info(f"Epoch {epoch} : train_loss: {logs['loss']} - train_accuracy: {logs['accuracy']} - train_recall: {logs['recall']} - train_precision: {logs['recall']} - train_F1: {logs['f1_score']} - val_loss: {logs['val_loss']} - val_accuracy: {logs['val_accuracy']} - val_recall: {logs['val_recall']} - val_precision: {logs['val_precision']} - val_f1: {logs['val_f1_score']}")

    # def on_train_end(self, logs=None):
    #     self.logger.info(f"Training finished. Time elapsed: {logs['time']}")

