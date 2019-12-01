# import category encoder 
from imports import *
from data import *
import warnings
warnings.filterwarnings("ignore")

def featuresEngineering(train=None, test=None):

	## Transform dates types from 'object' to 'datetime'
	train.TransactionStartTime = pd.to_datetime(train.TransactionStartTime)
	test.TransactionStartTime = pd.to_datetime(test.TransactionStartTime)
	train.IssuedDateLoan = pd.to_datetime(train.IssuedDateLoan)
	test.IssuedDateLoan = pd.to_datetime(test.IssuedDateLoan)
	train.PaidOnDate = pd.to_datetime(train.PaidOnDate)
	train.DueDate = pd.to_datetime(train.DueDate)

	## creating variables to transfer the information contained in the rows of the same transaction.
	train['Number_Of_Split_Payments'] = 0

	## this is a count on the number of payments on the same loan. It will take a 0 for singled-rowed transactions, 1+ for multi-row transacs.

	test['Number_Of_Split_Payments'] = 0

	## creating the feature : number of split payments on a loan.
	train['Number_Of_Split_Payments'] = train['TransactionId'].map(
		train.groupby('TransactionId').agg('count')['Number_Of_Split_Payments'])

	test['Number_Of_Split_Payments'] = test['TransactionId'].map(
		test.groupby('TransactionId').agg('count')['Number_Of_Split_Payments'])

	train.drop(
		train[(train.TransactionId == 'TransactionId_703') | (train.TransactionId == 'TransactionId_927')].index,
		axis=0, inplace=True)

	## Lets drop the duplicate rows with the same transaction ID and keep the last one. (as in with the latest payment installment )

	train.drop_duplicates(subset=['TransactionId'], keep='last', inplace=True)
	test.drop_duplicates(subset=['TransactionId'], keep='last', inplace=True)
	train.drop(['CountryCode', 'Currency', 'CurrencyCode', 'SubscriptionId', 'ProviderId', 'ChannelId'], axis=1,
			   inplace=True)
	test.drop(['CountryCode', 'CurrencyCode', 'SubscriptionId', 'ProviderId', 'ChannelId'], axis=1, inplace=True)

	# Feature Engineering
	train['Count_Rejected_Loans'] = train['CustomerId'].map(
		train[train.TransactionStatus == 0].groupby('CustomerId').LoanId.size())
	test['Count_Rejected_Loans'] = test['CustomerId'].map(
		train[train.TransactionStatus == 0].groupby('CustomerId').LoanId.size())

	## then we should impute the columns of customers that were not found in the rejected list with 0 as in they have never been rejected.
	train.Count_Rejected_Loans.fillna(value=0, inplace=True)
	test.Count_Rejected_Loans.fillna(value=0, inplace=True)

	## group train/test together to perform cumulative count
	all_data = pd.concat((train, test))

	## Initialize and compute values for the new feature
	all_data['Cumulative_Reject'] = 0
	all_data.loc[all_data.TransactionStatus == 0, 'Cumulative_Reject'] = all_data[
		all_data.TransactionStatus == 0].groupby('CustomerId').cumcount()

	## Separate all_data into train and test

	train1 = all_data[:len(train)]
	test1 = all_data[len(train):]
	train['Cumulative_Reject'] = 0
	test['Cumulative_Reject'] = 0
	train['Cumulative_Reject'] = train1['Cumulative_Reject']
	test['Cumulative_Reject'] = test1['Cumulative_Reject']

	purchasestats = train[train.TransactionStatus == 0].groupby('CustomerId').Value.agg(('mean', 'std', 'min', 'max'))
	train['prchs_mean'] = train['CustomerId'].map(purchasestats['mean'])
	train['prchs_std'] = train['CustomerId'].map(purchasestats['std'])
	train['prchs_max'] = train['CustomerId'].map(purchasestats['max'])
	train['prchs_min'] = train['CustomerId'].map(purchasestats['min'])
	test['prchs_mean'] = test['CustomerId'].map(purchasestats['mean'])
	test['prchs_std'] = test['CustomerId'].map(purchasestats['std'])
	test['prchs_max'] = test['CustomerId'].map(purchasestats['max'])
	test['prchs_min'] = test['CustomerId'].map(purchasestats['min'])

	valuegroups = train.groupby('CustomerId').Value.agg(('mean', 'std', 'min', 'max'))
	train['mean_cus_transac'] = train['CustomerId'].map(valuegroups['mean'])
	train['std_cus_transac'] = train['CustomerId'].map(valuegroups['std'])
	train['min_cus_transac'] = train['CustomerId'].map(valuegroups['min'])
	train['max_cus_transac'] = train['CustomerId'].map(valuegroups['max'])
	test['mean_cus_transac'] = test['CustomerId'].map(valuegroups['mean'])
	test['std_cus_transac'] = test['CustomerId'].map(valuegroups['std'])
	test['min_cus_transac'] = test['CustomerId'].map(valuegroups['min'])
	test['max_cus_transac'] = test['CustomerId'].map(valuegroups['max'])

	train['Day_Of_Week'] = train.TransactionStartTime.dt.weekday
	test['Day_Of_Week'] = test.TransactionStartTime.dt.weekday
	train['Day_in_month'] = train.TransactionStartTime.dt.day
	test['Day_in_month'] = test.TransactionStartTime.dt.day

	datemin = date(2018, 9, 21)
	datemax = date(2019, 7, 17)
	(datemax - datemin).days
	datesinc = pd.DataFrame(columns=['date', 'inc_value'])
	datesinc.loc[0, 'inc_value'] = 1
	datesinc.loc[0, 'date'] = datemin
	from datetime import timedelta
	for i in range(2, 301):
		datesinc.loc[i - 1, 'inc_value'] = i
		datesinc.loc[i - 1, 'date'] = datemin + timedelta(days=i - 1)
	train['inc_value_date'] = train.TransactionStartTime.dt.date.map(datesinc.set_index('date').inc_value)
	test['inc_value_date'] = test.TransactionStartTime.dt.date.map(datesinc.set_index('date').inc_value)

	train.inc_value_date = train.inc_value_date.astype(np.int64)
	test.inc_value_date = test.inc_value_date.astype(np.int64)

	aa = train[(train.TransactionStatus == 1) & (train.TransactionStartTime < train.DueDate)].groupby('CustomerId').agg(
		('count', 'mean', 'std', 'min', 'max')).Value
	train['before_due_mean'] = train['CustomerId'].map(aa['mean'])
	train['before_due_std'] = train['CustomerId'].map(aa['std'])
	train['before_due_min'] = train['CustomerId'].map(aa['min'])
	train['before_due_max'] = train['CustomerId'].map(aa['max'])
	test['before_due_mean'] = test['CustomerId'].map(aa['mean'])
	test['before_due_std'] = test['CustomerId'].map(aa['std'])
	test['before_due_min'] = test['CustomerId'].map(aa['min'])
	test['before_due_max'] = test['CustomerId'].map(aa['max'])

	train['Cnt_missed_payment'] = 0
	train.loc[train.DueDate < train.PaidOnDate, 'Cnt_missed_payment'] = train[train.DueDate < train.PaidOnDate].groupby(
		'CustomerId').cumcount()
	test['Cnt_missed_payment'] = test['CustomerId'].map(train.groupby('CustomerId').agg('max').Cnt_missed_payment)

	train = train[train.IsDefaulted.notnull()]

	features = ['CustomerId','Value', 'ProductId','ProductCategory','Number_Of_Split_Payments', 'Count_Rejected_Loans','mean_cus_transac', 'std_cus_transac', 'min_cus_transac', 'max_cus_transac',
			'Day_Of_Week', 'Day_in_month', 'inc_value_date', 'before_due_mean', 'before_due_std',
			'before_due_min', 'before_due_max','Cnt_missed_payment']

	oce = ce.OneHotEncoder(cols=['ProductId', 'ProductCategory'])
	tce = ce.TargetEncoder(cols=['CustomerId'], smoothing=40, min_samples_leaf=3)
	X = train[features]

	# handle missing values in training set
	X = X.fillna(X.median())

	X_test = test[features]

	# handle missing values in training set

	X_test = X_test.fillna(X_test.median())

	y = train.IsDefaulted.copy()
	X = oce.fit_transform(X)
	X = tce.fit_transform(X, y)
	X_test = oce.transform(X_test)
	X_test = tce.transform(X_test)

	return y, X, X_test


if __name__ == "__main__":

    # load train and test dataset

	train = load_data(file_name='Train.csv')
	test = load_data(file_name='Test.csv')

	y, X, X_test = featuresEngineering(train,test)

	print("Train features shape: {}".format(X.shape))
	print("Test features shape: {}".format(X_test.shape))
	print("Train columns: {}".format(X.columns))

