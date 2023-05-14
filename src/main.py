import argparse
import logging
import yaml
import tensorflow as tf
import time
import pretrain, finetuning, evaluate
from utils.utils import get_logger
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_siyuan.yaml")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(f'--finetuning', action='store_true')
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    assert args.config, "Please specify the config file"
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.train:
        exp_config = config['pretrain_exp']
        output_dir = Path(exp_config["output_directory"])/exp_config['exp_name']
        log = get_logger(output_dir, "main")
        log.info(f"==== Experiment Config ====\n {exp_config}")
        log.info(f"==== Training Phase ====")
        if exp_config['model_type'] not in ['baseline', 'cl', 'mtl']:
            raise ValueError(f"Model type {exp_config['model_type']} not supported. Please choose from ['baseline', 'cl', 'mtl']")
        
        start_time = time.time()
        config["pretrain"]["start_time"] = start_time
        trainer = pretrain.Trainer(exp_config, config["pretrain"])
        trainer.train()
        end_time = time.time()
        log.info(f"Overall Training time: {end_time - start_time}")
    if args.finetuning:
        exp_config = config['finetuning_exp']
        output_dir = Path(exp_config["output_directory"])/exp_config['exp_name']
        log = get_logger(output_dir, "main")
        log.info(f"==== Experiment Config ====\n {exp_config}")
        log.info(f"==== Finetuning Phase ====")
        
        start_time = time.time()
        config["finetuning"]["start_time"] = start_time
        trainer = finetuning.Trainer(exp_config, config["finetuning"])
        trainer.train()
        end_time = time.time()
        log.info(f"Overall Finetuning time: {end_time - start_time}")
    if args.evaluate:
        exp_config = config['evaluate_exp']
        output_dir = Path(exp_config["output_directory"])/exp_config['exp_name']
        log = get_logger(output_dir, "main")
        log.info(f"==== Experiment Config ====\n {exp_config}")
        log.info(f"==== Evaluation Phase ====")

        evaluator = evaluate.Evaluator(exp_config, config["evaluate"])
        evaluator.evaluate()
if __name__ == "__main__":
    main()
