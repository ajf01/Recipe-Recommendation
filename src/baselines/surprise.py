from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import PredefinedKFold
from surprise import accuracy


def main(configs):
    epochs = configs['EPOCHS']
    # Path to train file
    train_file_path = configs['train_data'] + "/" + configs['train_file']

    # 'User Item Rating', separated by '\t' characters.
    reader = Reader(line_format='user item rating', sep='\t')

    # Path to test file
    test_file_path = configs['test_data'] + "/" + configs['test_file']

    # Put the file paths into a list, and gets the data from those paths
    folds_files = [(train_file_path,test_file_path)]
    data = Dataset.load_from_folds(folds_files, reader=reader)
    
    # Create pre-defined folds and split data into train and test Trainset objects
    pkf = PredefinedKFold()
    trainset, testset = next(pkf.split(data))

    # Use SVD model popularized during Netflix Prize challenge and fit training data
    algo = SVD(epochs)
    algo.fit(trainset)
    
    # Get test predictions using SVD model
    predictions = algo.test(testset)
    
    # Get RMSE for the predictions made on the test data
    surprise_rmse = accuracy.rmse(predictions, verbose=False)
    
    print("Surprise RMSE: {}".format(surprise_rmse))
    
    
if __name__ == '__main__':
    main(sys.argv)