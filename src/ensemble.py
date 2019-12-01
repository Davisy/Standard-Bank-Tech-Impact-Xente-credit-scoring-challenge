# import important packages
from imports import *

from features_engineering import *
from data import *
from visualize import *
from logs import log

logger = log("../logs/ensemble_log_zindi")

# function to train the models
def fit_model(features, target, class_names, model_name):

    # split into train and val 
    X_Train, X_val, y_Train, y_val = train_test_split(features, target, stratify=target, test_size=0.1, random_state=42)

    logger.info("fitting model")

    # Voting classifier
    brc_classifier = BalancedRandomForestClassifier()
    xgb_classifier = XGBClassifier()
    eec_classifier = EasyEnsembleClassifier()
    bbc_classifier = BalancedBaggingClassifier()

    #xgb_classifier = XGBClassifier(booster='gbtree', max_depth=10, n_estimators=100)

    #eec_classifier = EasyEnsembleClassifier(base_estimator=RandomForestClassifier(),n_estimators=30,sampling_strategy='all')

    #brc_classifier = BalancedRandomForestClassifier(class_weight='balanced', n_estimators=50, sampling_strategy='all')

    #bbc_classifier = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=100, sampling_strategy='all')



    classifiers = [
        ("eec", eec_classifier),
        ("xgb", xgb_classifier),
        ("bbc",bbc_classifier),
    ]

    voting_clf = VotingClassifier(estimators=classifiers)

    logger.info("Train {}".format("{}".format(voting_clf)))

    sm = SMOTEENN(random_state=42)
    X_train, y_train = sm.fit_resample(X_Train, y_Train)

    # convert into Dataframe

    features = list(features.columns)

    X_train = pd.DataFrame(X_train, columns=features)

    voting_clf.fit(X_train, y_train)

    y_pred = voting_clf.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix : \n", cm)

    accuracy = roc_auc_score(y_val, y_pred)
    logger.info("roc auc score: {:.3f}".format(accuracy))
    clf_report = classification_report(y_val, y_pred)
    print(clf_report)
    #plot_confusion_matrix(y_val, y_pred, class_names, title=model_name + "-cm")
    #plt.savefig("../figures/test_figures/{}_{}_cm.pdf".format(model_name, "results"),bbox_inches="tight",)
    #plt.close()
    # save the models
    joblib.dump(voting_clf, "../models/{}--roc-{:.3f}.pkl".format(model_name, accuracy))
    logger.info("-------------------------------")


def main():

    # load dataset
    train = load_data(file_name='Train.csv')
    test = load_data(file_name='Test.csv')

    #feature engineering
    y, X, X_test = featuresEngineering(train, test)

    #features selection
    features = ['ProductId_14', 'before_due_mean', 'Number_Of_Split_Payments',
                'CustomerId', 'before_due_min', 'Value', 'inc_value_date',
                'max_cus_transac', 'before_due_max', 'ProductCategory_6',
                'Day_in_month', 'min_cus_transac', 'mean_cus_transac',
                'std_cus_transac', 'Day_Of_Week', 'ProductCategory_1',
                'Cnt_missed_payment', 'ProductId_4', 'Count_Rejected_Loans',
                'before_due_std', 'ProductId_8', 'ProductId_1', 'ProductId_7',
                'ProductCategory_4', 'ProductCategory_2']

    X = X[features]

    # name of the classes
    classes = ["1", "0"]

    model = "Voting Classifier "
    fit_model(features=X, target=y, class_names=classes, model_name=model)


if __name__ == "__main__":
    main()
