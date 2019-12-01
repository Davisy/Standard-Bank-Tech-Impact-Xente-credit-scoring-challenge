# import important packages
from classifiers import *
from logs import log
from features_engineering import *

# set a logger file
logger = log("../logs/train_logs_grid_zindi")

## feaures selected
features = ['ProductId_14', 'before_due_mean', 'Number_Of_Split_Payments',
			'CustomerId', 'before_due_min', 'Value', 'inc_value_date',
			'max_cus_transac', 'before_due_max', 'ProductCategory_6',
			'Day_in_month', 'min_cus_transac', 'mean_cus_transac',
			'std_cus_transac', 'Day_Of_Week', 'ProductCategory_1',
			'Cnt_missed_payment', 'ProductId_4', 'Count_Rejected_Loans',
			'before_due_std', 'ProductId_8', 'ProductId_1', 'ProductId_7',
			'ProductCategory_4', 'ProductCategory_2']


# function to train the models
def fit_model(parameters=parameters, models=models, best_features=features):
	# name of the classes
	class_names = ['1', '0']

	# load the dataset
	train = load_data(file_name='Train.csv')
	test = load_data(file_name='Test.csv')

	# features engineering

	y, X, X_test = featuresEngineering(train, test)

	# feature selection
	logger.info("features selected {}".format(best_features))

	X = X[best_features]
	X_test = X_test[best_features]

	# split into train and val
	X_Train, X_val, y_Train, y_val = train_test_split(X, y, stratify=y,
													  test_size=0.15, random_state=42)

	# control data imbalance

	sm = SMOTEENN(random_state=42)
	X_train, y_train = sm.fit_resample(X_Train, y_Train)

	# convert into Dataframe

	features = list(X.columns)

	X_train = pd.DataFrame(X_train, columns=features)


	logger.info("fitting model")

	for model_name, model in models.items():
		logger.info("Train {}".format(model_name))

		# gridsearch for each classifier
		clf = GridSearchCV(model, parameters[model_name], cv=5, scoring='roc_auc')
		clf.fit(X_train, y_train)

		print(clf.best_params_)
		logger.info("Best parameters values:{}".format(clf.best_params_))

		y_pred = clf.predict(X_val)
		cm = confusion_matrix(y_val, y_pred)
		print('Confusion Matrix : \n', cm)

		accuracy_result = roc_auc_score(y_val, y_pred)

		logger.info("Accuracy Score: {:.3f}".format(accuracy_result))
		clf_report = classification_report(y_val, y_pred)
		print(clf_report)

		# save the model
		joblib.dump(clf.best_estimator_, '../models/{}-roc-{:.3f}.pkl'.format(model_name, accuracy_result))
		logger.info("-------------------------------")


def main():
	fit_model(parameters=parameters, models=models, best_features=features)


if __name__ == "__main__":
	main()
