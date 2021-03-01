import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

def main(configs):
    recipes = configs['final'][configs['size']]
    recipes['calories'] =  [eval(x)[0] for x in recipes['nutrition']]
    recipes = recipes[['name','minutes','ingredients','n_ingredients','calories','mean_rating','cuisine']]
    
    test = recipes['ingredients'].apply(lambda x: eval(x))
    
    mlb = MultiLabelBinarizer()
    mlb.fit(test)
    
    input_vector = eval(config['sampleInput'])
    
    ingredients_transformed = mlb.transform(test)
    recipe_test_trans = mlb.transform(input_vector)

    sims = []
    for recipe in ingredients_transformed:
        sim = cosine_similarity(recipe_test_trans,recipe.reshape(-1,len(recipe)))
        sims.append(sim)

    recipes['sim'] = [x[0][0] for x in sims]
    return recipes.set_index('name')[:5]
    
if __name__ == '__main__':
    main(sys.argv)