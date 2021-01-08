import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

def RMSE(predictions, labels):
    """Function that takes in predictions and real values and computes RMSE."""
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return np.sqrt(sum(differences)/len(differences))

def main(configs):
    # Get the training and test data
    train_path = configs['train_data'] + "/" + configs['train_file']
    test_path = configs['test_data'] + "/" + configs['test_file']
    
    # Change train and test data in pandas dataframes
    train_data = pd.read_csv(train_path,sep='\t',names=['User ID', 'Movie ID', 'Rating'])
    test_data = pd.read_csv(test_path,sep='\t',names=['User ID', 'Movie ID', 'Rating'])
    
    # Split data between features and y values
    train_x = train_data[['User ID', 'Movie ID']]
    train_y = train_data['Rating']
    test_x = test_data[['User ID', 'Movie ID']]
    test_y = test_data['Rating']
    
    # Turn train data into a dictionary
    train_dict = train_data.groupby('Movie ID').mean().drop(columns=['User ID']).T.to_dict('list')
    
    # Get test predictions by getting mean value per movie as the prediction for the movie user pair
    test_pred = [train_dict[d][0] for d in test_x['Movie ID']]
    
    # Calculate RMSE
    mean_movie_rmse = RMSE(test_pred,test_y)

    print("Mean Movie RMSE: {}".format(mean_movie_rmse))
    
    
if __name__ == '__main__':
    main(sys.argv)