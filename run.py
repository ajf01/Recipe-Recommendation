#!/usr/bin/env python

from os import listdir, path, makedirs
import sys
import json
from src.data import etl
from src.models import train, evaluate, rmse
from src.baselines import surprise, linReg, meanMovie


def main(targets):
    """ Runs data pipeline to parse all the data into these folders and turn movie title data into a csv"""

    if targets == 'test':
        filepath = 'config/test_params.json'
        with open(filepath) as file:
            configs = json.load(file)

        etl.main(configs)
        train.main(configs)
        evaluate.main(configs)
        rmse.main(configs)
        
        print("####################")
        meanMovie.main(configs)
        linReg.main(configs)
        surprise.main(configs) 
        print("####################")

    if targets == 'data' or targets == 'all':
        filepath = 'config/etl_params.json'
        with open(filepath) as file:
            configs = json.load(file)
            
        etl.main(configs)
        
    if targets == 'train' or targets == 'all':
        filepath = 'config/train_eval_params.json'
        with open(filepath) as file:
            configs = json.load(file)
        
        train.main(configs)
            
    if targets == 'eval' or targets == 'all':
        filepath = 'config/train_eval_params.json'
        with open(filepath) as file:
            configs = json.load(file)
            
        evaluate.main(configs)
        
    if targets == 'rmse' or targets == 'all':
        filepath = 'config/rmse_params.json'
        with open(filepath) as file:
            configs = json.load(file)        
        
        rmse.main(configs)
        
    if targets == 'baselines' or targets == 'all':
        filepath = 'config/baselines_params.json'
        with open(filepath) as file:
            configs = json.load(file)        
        
        print("####################")
        meanMovie.main(configs)
        linReg.main(configs)
        surprise.main(configs) 
        print("####################")

    return None


if __name__ == '__main__':
    targets = sys.argv[1]
    main(targets)
