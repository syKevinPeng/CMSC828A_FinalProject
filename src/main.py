import argparse
import yaml
import tensorflow as tf
import time
import pretrain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_siyuan.yaml')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()


    assert args.config, 'Please specify the config file'
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader())

    if args.train:
        start_time = time.time()
        config["pretrain"]["start_time"] = start_time
        trainer = pretrain.Trainer(config['experiment'],config['pretrain'] )