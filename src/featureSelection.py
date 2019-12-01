# import important packagesr
from imports import *
from data import *
from visualize import *
from logs import log
from features_engineering import *


logger = log("../logs/featureSelection")

# function to train the models
def selectBestFeatures(features, target, no_features):

    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=no_features)
    fit = bestfeatures.fit(features, target)

    dfscores = pd.DataFrame(fit.scores_)

    dfcolumns = pd.DataFrame(features.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    best_features = featureScores.nlargest(no_features, 'Score')
    columns = list(best_features.Specs)

    if __name__ == "__main__":
        print(featureScores.nlargest(no_features, 'Score'))  # print 10 best features
        print("best features: {}".format(best_features.Specs))

    return columns

def feature_importance(features, target, classifier, no_features):

    logger.info("fitting model")

    model = classifier()
    model.fit(features, target)

    dfscores = pd.DataFrame(model.feature_importances_)

    dfcolumns = pd.DataFrame(features.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    best_features = featureScores.nlargest(no_features, 'Score')
    columns = list(best_features.Specs)

    if __name__ == "__main__":
        print(featureScores.nlargest(no_features, 'Score'))  # print 10 best features
        print(columns)

    return columns

def main():

    # load dataset
    train = load_data(file_name='Train.csv')
    test = load_data(file_name='Test.csv')

    y, X, X_test = featuresEngineering(train,test)

    mylist = feature_importance(features=X, target=y, classifier=ExtraTreesClassifier, no_features=20)

if __name__ == "__main__":
    main()
