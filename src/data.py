from imports import *


# A function to load the dataset

def load_data(file_name):
    """
       data loading
       file_name: Show the name of file to load
       """
    data = pd.read_csv('../data/{}'.format(file_name))

    return data


if __name__ == "__main__":
    # load train dataset
    file_name = 'Test.csv'

    data = load_data(file_name)

    print("data shape: {}".format(data.shape))
    print("data columns: {}".format(data.columns))

