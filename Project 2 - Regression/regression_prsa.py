import pandas as pd
import numpy as np
import os
from math import sqrt



LR_state = True
SVR_state = False
RFR_state = True
MLP_state = False

# read csvs and join them in a single dataset
data = pd.read_csv("PRSA_data_2010.1.1-2014.12.31_dum.csv").fillna(0)

data = data.drop(data[data.pm25 == 0].index)

# get rid of the key
X = data.drop(['No', 'pm25', 'cbwd'], axis = 1)

# save the labels in y
y = data['pm25']

# split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = pd.DataFrame(X_train, columns=X.columns.values.tolist()).rename(columns={'TEMP': 'Temp','PRES':'Pres'})
X_test = pd.DataFrame(X_test, columns=X.columns.values.tolist()).rename(columns={'TEMP': 'Temp','PRES':'Pres'})
y_train = pd.DataFrame(y_train, columns=['pm25'])
y_test = pd.DataFrame(y_test, columns=['pm25'])

print(X_train.columns.values.tolist())
print(X_test.columns.values.tolist())
print(type(X_train))


feat_set = "BeijingPM2.5"

# free-up memory
del data

#preparing results file
#file = open("results_train_test.csv","w")
#file.write(','.join(list(training)))
#file.write("\n")


# minmax scalling
##from sklearn.preprocessing import MinMaxScaler

##scalermm = MinMaxScaler()
##scalermm.fit(X_train)

##X_train = scalermm.transform(X_train)
##X_test = scalermm.transform(X_test)


# DT to see importance
##from sklearn.tree import DecisionTreeClassifier

# DT manually adjusted to avoid overfit 
##dtree = DecisionTreeClassifier(min_samples_leaf=1,max_depth=9)
##dtree.fit(X_train,y_train)

##print('\nfeatures importance')
##feat_imp = np.multiply(dtree.feature_importances_,100)
##print(feat_imp)
##feat_imp.tofile(file, sep=",", format="%.3f")
##file.write("\n")

# features selection, choose index of features
##indices = np.where(dtree.feature_importances_ == 0)
##X_train = np.delete(X_train, indices[0], axis=1)
##X_test = np.delete(X_test, indices[0], axis=1)

##print('\nirrelevant features (deleted)')
##print(indices[0])

# center the data (- mean)
##from sklearn.preprocessing import StandardScaler

##scalerm = StandardScaler(with_std = False)
##scalerm.fit(X_train)

##X_train = scalerm.transform(X_train)
##X_test = scalerm.transform(X_test)

# apply PCA
##from sklearn.decomposition import PCA

##pca = PCA()
##pca.fit(X_train)

##X_train_pca = pca.transform(X_train)
##X_test_pca = pca.transform(X_test)

##print('\nPCA variance')
##print(np.multiply(pca.explained_variance_ratio_,100))


# DT to see importance after PCA
##from sklearn.tree import DecisionTreeClassifier
##from sklearn.model_selection import cross_validate

##dtree = DecisionTreeClassifier(min_samples_leaf=1,max_depth=9)
##dtree.fit(X_train_pca,y_train)
##predict_train = dtree.predict(X_train_pca)
##predict = dtree.predict(X_test_pca)


##print('\nimportance of feaatures after PCA')
##print(np.multiply(dtree.feature_importances_,100))

# feature reduction (select feature index)
##indices = np.where(dtree.feature_importances_ == 0)
##X_train_pca = np.delete(X_train_pca, indices[0], axis=1)
##X_test_pca = np.delete(X_test_pca, indices[0], axis=1)
##num_features=len(X_train[0])

##print('\nirrelevant features (deleted)')
##print(indices[0])

# importing metrics and X-val
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
# DT training

if LR_state:
	print('\n####################### LinearRegression#######################')
	from sklearn.linear_model import LinearRegression
	param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
	regr = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
	regr.fit(X_train, y_train)
	# Prediction
	y_pred = regr.predict(X_test)
	print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
	print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
	print('R2: {}'.format(r2_score(y_test, y_pred)))
	print('RMSE: {}'.format(sqrt(mean_squared_error(y_test, y_pred))))
	print('MeAE: {}'.format(median_absolute_error(y_test, y_pred)))
	#Print best params
	print('LR Params: {}'.format(regr.best_params_))


if SVR_state:
	print('\n####################### SVR Regression#######################')
	from sklearn.svm import SVR
	svr = GridSearchCV(SVR(kernel='rbf'), cv=5, n_jobs=-1, verbose=2, param_grid={"C": np.logspace(-5, 5, num=5, base=2), "gamma": np.logspace(-5, 5, num=5, base=2)})
	svr.fit(X_train,y_train)
	#Prediction	
	y_pred = svr.predict(X_test)
	print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
	print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
	print('R2: {}'.format(r2_score(y_test, y_pred)))
	print('RMSE: {}'.format(sqrt(mean_squared_error(y_test, y_pred))))
	print('MeAE: {}'.format(median_absolute_error(y_test, y_pred)))

	# Print best params
	print('SVR Params: {}'.format(svr.best_params_))

if RFR_state:
	print('\n####################### RF Regression#######################')
	from sklearn.ensemble import RandomForestRegressor
	param_grid = {
		"n_estimators": [100, 150, 200, 500, 1000, 1500],
		"max_features": ["auto"], #, "sqrt", "log2"
		"min_samples_split": [2], #, 4, 8, 10, 12, 14, 16
		"bootstrap": [True], #, False
    }

    # GridSearch in param_grid
	rfr = GridSearchCV(RandomForestRegressor(), cv=5, n_jobs=-1, verbose=2, param_grid=param_grid)
	#Fit Model
	rfr.fit(X_train, y_train)

	#Prediction	
	y_pred = rfr.predict(X_test)
	print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
	print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
	print('R2: {}'.format(r2_score(y_test, y_pred)))
	print('RMSE: {}'.format(sqrt(mean_squared_error(y_test, y_pred))))
	print('MeAE: {}'.format(median_absolute_error(y_test, y_pred)))

    # Print best params
	print('RandomForestsRegressor Params: {}'.format(rfr.best_params_))

if MLP_state:
	print('\n####################### MLP Regression#######################')
	from sklearn.neural_network import MLPRegressor
	param_grid = {'solver': ['lbfgs'], 
				'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 
				'alpha': 10.0 ** -np.arange(1, 10), 
				'hidden_layer_sizes':np.arange(10, 15), 
				'random_state':[0,1,2,3,4,5,6,7,8,9]}
	mlpr = GridSearchCV(MLPClassifier(), cv=5, param_grid=param_grid, n_jobs=-1)	

	#Fit Model
	mlpr.fit(X_train, y_train)

	#Prediction	
	y_pred = mlpr.predict(X_test)
	print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
	print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
	print('R2: {}'.format(r2_score(y_test, y_pred)))
	print('RMSE: {}'.format(sqrt(mean_squared_error(y_test, y_pred))))
	print('MeAE: {}'.format(median_absolute_error(y_test, y_pred)))

    # Print best params
	print('RandomForestsRegressor Params: {}'.format(mlpr.best_params_))


