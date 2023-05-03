import argparse
import logging
import yaml
import tensorflow as tf
import time
import pretrain
from utils.utils import get_logger
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_siyuan.yaml")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    assert args.config, "Please specify the config file"
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    output_dir = Path(config['experiment']["output_directory"])/config['experiment']['exp_name']
    log = get_logger(output_dir, "main")
    log.info(f"==== Experiment Config ====\n {config['experiment']}")

    if args.train:
        log.info(f"==== Training Phase ====")
        start_time = time.time()
        config["pretrain"]["start_time"] = start_time
        trainer = pretrain.Trainer(config["experiment"], config["pretrain"])
        trainer.train()


if __name__ == "__main__":
    main()
