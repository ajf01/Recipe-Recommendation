# Copyright (c) 2017 NVIDIA Corporation
from os import listdir, path, makedirs
import kaggle
import pandas as pd
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
    makedirs(out_folder, exist_ok=True)
    
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('shuyangli94/food-com-recipes-and-user-interactions', path=folder, unzip=True)
    
    
    raw_r = pd.read_csv(configs['recipe'])
    raw_i = pd.read_csv(configs['inter'])

    raw_r = raw_r[raw_r['minutes'] < 300000]
    combined = pd.merge(raw_r,raw_i,how='inner',left_on='id',right_on='recipe_id')
    combined.to_csv(out_folder+'/combined.csv',index=False)

if __name__ == "__main__":
    main(sys.argv)
