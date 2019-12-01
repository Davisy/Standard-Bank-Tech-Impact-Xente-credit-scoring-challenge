# import important packages

from imports import *
from data import *
from classifiers import *
from logs import log
from features_engineering import featuresEngineering

# set a logger file
logger = log("../logs/cv_logs_zindi")

# function to train the models
def fit_model(models = models ):

    # load the dataset
    train = load_data(file_name='Train.csv')
    test = load_data(file_name='Test.csv')

    # perform feature engineering
    y, X, X_test = featuresEngineering(train, test)

    # perform feature selection

    logger.info("fitting model")

    for model_name, model in models.items():

        logger.info("Train {}".format(model_name))

        # cross_val_score for each classifier
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

        logger.info("List of scores for {}: {}".format(model_name, scores))
        logger.info("The mean score for {}: {}".format(model_name, scores.mean()))

        logger.info("-------------------------------")


def main():
    fit_model(models=models)


if __name__ == "__main__":
    main()




