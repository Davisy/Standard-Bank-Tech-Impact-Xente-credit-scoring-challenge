# import important packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import (
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
)

#PREPROCESSING AND METRICS
from sklearn.externals import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import category_encoders as ce
from sklearn.feature_selection import chi2, f_classif

#SPECIAL TOOLS
import itertools
import os
import logging
from datetime import date

#IMBALANCE OF DATASET
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import (
    RandomUnderSampler,
    ClusterCentroids,
    TomekLinks,
    NeighbourhoodCleaningRule,
    NearMiss,
)
import warnings

warnings.filterwarnings("ignore")
np.random.seed(0)