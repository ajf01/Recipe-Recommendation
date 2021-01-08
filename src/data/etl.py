# Copyright (c) 2017 NVIDIA Corporation
from os import listdir, path, makedirs
import kaggle
import random
import sys

def save_data_to_file(data, filename):
    with open(filename, 'w') as out:
        for userId in data:
            for record in data[userId]:
                out.write("{}\t{}\t{}\n".format(userId, record[0], record[1]))

def main(configs):
    folder = configs['original_loc']
    out_folder = configs['output_location']
    # create necessary folders:
    makedirs(folder, exist_ok=True)
    
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('Food.com Recipes and Interactions', path=folder, unzip=True)

if __name__ == "__main__":
    main(sys.argv)
