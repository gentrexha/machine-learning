import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

LR_state = True
SVR_state = True
RFR_state = True
MLP_state = True

# read csvs and join them in a single dataset
data = pd.read_csv('data/processed/StudentPerformance.shuf.train_dum.csv').fillna(0)
test_data = pd.read_csv('data/processed/StudentPerformance.shuf.test_dum.csv').fillna(0)
ids_list = test_data['id']
test_data = test_data.drop(['id', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                            'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                            'romantic'], axis=1)

# get rid of the key
X = data.drop(['id', 'Grade', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
               'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], axis=1)

# save the labels in y
y = data['Grade']

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# free-up memory
del data

# preparing results file
# file = open("results_train_test.csv","w")
# file.write(','.join(list(training)))
# file.write("\n")


# minmax scalling
from sklearn.preprocessing import MinMaxScaler

scalermm = MinMaxScaler()
scalermm.fit(X_train)

X_train = scalermm.transform(X_train)
X_test = scalermm.transform(X_test)
test_data = scalermm.transform(test_data)


# DT to see importance
from sklearn.tree import DecisionTreeClassifier

# DT manually adjusted to avoid overfit 
dtree = DecisionTreeClassifier(min_samples_leaf=1, max_depth=11)
dtree.fit(X_train, y_train)

##print('\nfeatures importance')
##feat_imp = np.multiply(dtree.feature_importances_,100)
##print(feat_imp)
##feat_imp.tofile(file, sep=",", format="%.3f")
##file.write("\n")

# features selection, choose index of features
indices = np.where(dtree.feature_importances_ == 0)
X_train = np.delete(X_train, indices[0], axis=1)
X_test = np.delete(X_test, indices[0], axis=1)
test_data = np.delete(test_data, indices[0], axis=1)

print('\nirrelevant features (deleted)')
print(indices[0])

# center the data (- mean)
from sklearn.preprocessing import StandardScaler

scalerm = StandardScaler(with_std = False)
scalerm.fit(X_train)

X_train = scalerm.transform(X_train)
X_test = scalerm.transform(X_test)

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
# DT training

if LR_state:
    print('\n####################### LinearRegression#######################')

    param_grid = {
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'copy_X': [True, False]
    }
    regression = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    regression.fit(X_train, y_train)

    y_prediction = regression.predict(X_test)

    print('MAE: {}'.format(mean_absolute_error(y_test, y_prediction)))
    print('MSE: {}'.format(mean_squared_error(y_test, y_prediction)))
    print('R2: {}'.format(r2_score(y_test, y_prediction)))
    print('RMSE: {}'.format(sqrt(mean_squared_error(y_test, y_prediction))))
    print('MeAE: {}'.format(median_absolute_error(y_test, y_prediction)))

    print('LR Params: {}'.format(regression.best_params_))

    test_prediction = regression.predict(test_data)
    prediction_dataframe = pd.DataFrame({'id': ids_list, 'Grade': test_prediction})

    prediction_dataframe.to_csv('prediction-LRStudentPerformanceTest.csv', index=False, columns=['id', 'Grade'])
    print('Saved LR file!')

if SVR_state:
    print('\n####################### SVR Regression#######################')

    param_grid = {
        "C": np.logspace(-5, 5, num=5, base=2),
        "gamma": np.logspace(-5, 5, num=5, base=2)
    }
    svr = GridSearchCV(SVR(kernel='rbf'), cv=5, n_jobs=-1, verbose=1, param_grid=param_grid)
    svr.fit(X_train, y_train)

    y_prediction = svr.predict(X_test)
    print('MAE: {}'.format(mean_absolute_error(y_test, y_prediction)))
    print('MSE: {}'.format(mean_squared_error(y_test, y_prediction)))
    print('R2: {}'.format(r2_score(y_test, y_prediction)))
    print('RMSE: {}'.format(sqrt(mean_squared_error(y_test, y_prediction))))
    print('MeAE: {}'.format(median_absolute_error(y_test, y_prediction)))

    print('SVR Params: {}'.format(svr.best_params_))

    test_prediction = svr.predict(test_data)
    prediction_dataframe = pd.DataFrame({'id': ids_list, 'Grade': test_prediction})

    prediction_dataframe.to_csv('prediction-SVRStudentPerformanceTest.csv', index=False, columns=['id', 'Grade'])
    print('Saved SVR file!')

if RFR_state:
    print('\n####################### RF Regression#######################')

    param_grid = {
        "n_estimators": [1500],
        "max_features": ["auto"],  # , "sqrt", "log2"
        "min_samples_split": [2],  # , 4, 8, 10, 12, 14, 16
        "bootstrap": [True],  # , False
    }

    # GridSearch in param_grid
    rfr = GridSearchCV(RandomForestRegressor(), cv=5, n_jobs=-1, verbose=1, param_grid=param_grid)

    rfr.fit(X_train, y_train)

    y_prediction = rfr.predict(X_test)

    print('MAE: {}'.format(mean_absolute_error(y_test, y_prediction)))
    print('MSE: {}'.format(mean_squared_error(y_test, y_prediction)))
    print('R2: {}'.format(r2_score(y_test, y_prediction)))
    print('RMSE: {}'.format(sqrt(mean_squared_error(y_test, y_prediction))))
    print('MeAE: {}'.format(median_absolute_error(y_test, y_prediction)))

    # Print best params
    print('RandomForests Regressor Params: {}'.format(rfr.best_params_))

    test_prediction = rfr.predict(test_data)
    prediction_dataframe = pd.DataFrame({'id': ids_list, 'Grade': test_prediction})

    prediction_dataframe.to_csv('prediction-RFRStudentPerformanceTest.csv', index=False, columns=['id', 'Grade'])
    print('Saved RFR file!')

if MLP_state:
    print('\n####################### MLP Regression#######################')

    param_grid = {'solver': ['lbfgs'],
                  'max_iter': [1300, 1400, 1500],
                  'alpha': 10.0 ** -np.arange(2, 3),
                  'hidden_layer_sizes': np.arange(6, 13)
                  }
    mlp_regression = GridSearchCV(MLPRegressor(), cv=5, param_grid=param_grid, n_jobs=-1, verbose=1)
    mlp_regression.fit(X_train, y_train)

    y_prediction = mlp_regression.predict(X_test)

    print('MAE: {}'.format(mean_absolute_error(y_test, y_prediction)))
    print('MSE: {}'.format(mean_squared_error(y_test, y_prediction)))
    print('R2: {}'.format(r2_score(y_test, y_prediction)))
    print('RMSE: {}'.format(sqrt(mean_squared_error(y_test, y_prediction))))
    print('MeAE: {}'.format(median_absolute_error(y_test, y_prediction)))

    # Print best params
    print('RandomForests Regressor Params: {}'.format(mlp_regression.best_params_))

    test_prediction = mlp_regression.predict(test_data)
    prediction_dataframe = pd.DataFrame({'id': ids_list, 'Grade': test_prediction})

    prediction_dataframe.to_csv('prediction-MLPStudentPerformanceTest.csv', index=False, columns=['id', 'Grade'])
    print('Saved MLP file!')
