import argparse
import logging
import yaml
import tensorflow as tf
import time
import pretrain
from utils.utils import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_siyuan.yaml")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    log = get_logger("main")

    assert args.config, "Please specify the config file"
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    log.info(f"==== Experiment Config ====\n {config['experiment']}")

    if args.train:
        log.info(f"==== Training Start ====")
        start_time = time.time()
        config["pretrain"]["start_time"] = start_time
        trainer = pretrain.Trainer(config["experiment"], config["pretrain"])
        trainer.train()


if __name__ == "__main__":
    main()
