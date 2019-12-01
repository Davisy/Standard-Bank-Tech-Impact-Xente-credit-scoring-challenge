from imports import *

# GridSearch: Set parameters values for each classifier

# for KNN
k_range = range(10, 31, 3)


parameters = {
    "MLP": [{"hidden_layer_sizes": [(400,), (500,), (600,)]}],
    "KNN": [
        {
            "n_neighbors": k_range,
            "weights": ["uniform", "distance"],
            "n_jobs": [1],
        }
    ],
    "GB": [
        {
            "learning_rate": [0.1, 0.01],
            "n_estimators": [10, 30, 50, 100, 150],
        }
    ],
    "RF": [
        {
            "n_estimators": [5, 10, 30, 50, 100],
            "max_features": ["auto", "log2", None],
            "class_weight": ["balanced_subsample", "balanced", None],
            "min_samples_leaf": [0.2, 0.4, 1],
        }
    ],
    "DTC": [
        {
            "max_depth": [None, 1, 5, 10],
            "min_samples_split": [2, 4, 6, 8, 10, 12],
            "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7],
        }
    ],
    "SVC": [
        {
            "C": [1.0, 10.0],
            "kernel": ["linear", "rbf", "sigmoid"],
        }
    ],
    "BC": [{"n_estimators": [10, 50, 100, 120, 150]}],
    "XGB": [
        {
            "booster": ["gbtree", "dart"],
            "max_depth": [10, 30, 50, 70, 100],
            "n_estimators": [10, 20, 30, 50, 100],
        }
    ],
    "EXT": [{"n_estimators": [5, 10, 50, 100]}],
    "LG": [{"penalty": ["l1", "l2"]}],
    "BBC": [
        {
            "base_estimator": [
                None,
                DecisionTreeClassifier(),
                RandomForestClassifier(),
            ],
            "n_estimators": [10, 30, 50, 100],
            "sampling_strategy": ["auto", "all"],
        }
    ],
    "BRC": [
        {
            "n_estimators": [10, 30, 50, 100, 150],
            "sampling_strategy": ["all", "auto"],
            "class_weight": ["balanced_subsample", "balanced", None],
        }
    ],
    "EEC": [
        {
            "n_estimators": [10, 30, 50, 100],
            "base_estimator": [
                RandomForestClassifier(),
                BalancedRandomForestClassifier(),
            ],
            "sampling_strategy": ["all", "auto"],
        }
    ],
}

# List of Classifiers to
models = {
    "KNN": KNeighborsClassifier(),
    "RF": RandomForestClassifier(),
    "GB": GradientBoostingClassifier(),
    "DTC": DecisionTreeClassifier(),
    "MLP": MLPClassifier(),
    "SVC": svm.SVC(),
    "BC": BaggingClassifier(),
    "XGB": XGBClassifier(),
    "EXT": ExtraTreesClassifier(),
    "LG": LogisticRegression(),
    "ADB": AdaBoostClassifier(),
    "BBC": BalancedBaggingClassifier(),
    "BRC": BalancedRandomForestClassifier(),
    "EEC": EasyEnsembleClassifier(),
}
# NOTE: not all classifiers are selected in different ML experiments
