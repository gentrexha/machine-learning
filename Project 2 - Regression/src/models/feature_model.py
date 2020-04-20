import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import numpy as np

from utils import print_measurement_scores


def predict_StandardScaler():
    """

    :return:
    """
    # Load Dataset, 0=
    df = pd.read_csv('C:/Projects/University/Machine Learning/ml-regression/data/raw/Crime/crimedata_feature_0.csv')

    X = df.drop(['ViolentCrimesPerPop'], axis=1)
    y = df['ViolentCrimesPerPop']
    # # Validation set should be the test set. But we don't have the solutions for that one
    # X_validation = df.loc[2018:].drop(['weekly_infections'], axis=1)
    # y_validation = df.loc[2018:]['weekly_infections']

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # # # # # # # # # # #
    # LinearRegression  #
    # # # # # # # # # # #

    param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
    regr = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    regr.fit(X_train, y_train)

    # Predict test set
    y_pred = regr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'LinearRegression', 'Crime Dataset. Feature Selected & StandardScaler.')

    # # # # # # # # #
    # SVR           #
    # # # # # # # # #

    #
    svr = GridSearchCV(SVR(kernel='rbf'), cv=5, n_jobs=-1, verbose=1,
                       param_grid={"C": np.logspace(-5, 5, num=2, base=2), "gamma": np.logspace(-5, 5, num=2, base=2)})
    # Fit Model
    svr.fit(X_train, y_train)

    # Predict test set
    y_pred = svr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'SVR', 'Crime Dataset. Feature Selected & StandardScaler.')
    # Print params
    print('SVR Params: {}'.format(svr.best_params_))

    # # # # # # # # # # # # # #
    # RandomForestRegressor   #
    # # # # # # # # # # # # # #

    param_grid = {
        "n_estimators": [10, 20, 30, 50, 100],
        "min_samples_split": [2, 4, 8, 10, 12, 14, 16],
    }

    # GridSearch in param_grid
    rfr = GridSearchCV(RandomForestRegressor(), cv=5, n_jobs=-1, verbose=1, param_grid=param_grid)
    # Fit Model
    rfr.fit(X_train, y_train)

    # Predict test set
    y_pred = rfr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'RandomForestRegressor', 'Crime Dataset. Feature Selected & StandardScaler.')

    # Print best params
    print('RandomForestsRegressor Params: {}'.format(rfr.best_params_))

    # # # # # # # # # # # # # #
    # MLPRegressor             #
    # # # # # # # # # # # # # #

    param_grid = {'solver': ['lbfgs'],
                  'max_iter': [1000, 1500, 2000],
                  'alpha': 10.0 ** -np.arange(1, 10),
                  'hidden_layer_sizes': np.arange(10, 15)}
    mlpr = GridSearchCV(MLPRegressor(), cv=5, param_grid=param_grid, n_jobs=-1, verbose=1)

    # Fit Model
    mlpr.fit(X_train, y_train)

    # Prediction
    y_pred = mlpr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'MLPRegressor', 'Crime Dataset. Feature Selected & StandardScaler.')

    # Print best params
    print('RandomForestsRegressor Params: {}'.format(mlpr.best_params_))


def predict_MinMaxScaler():
    """

    :return:
    """
    # Load Dataset
    df = pd.read_csv('C:/Projects/University/Machine Learning/ml-regression/data/raw/Crime/crimedata_feature_1.csv')

    X = df.drop(['ViolentCrimesPerPop'], axis=1)
    y = df['ViolentCrimesPerPop']
    # # Validation set should be the test set. But we don't have the solutions for that one
    # X_validation = df.loc[2018:].drop(['weekly_infections'], axis=1)
    # y_validation = df.loc[2018:]['weekly_infections']

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # # # # # # # # # # #
    # LinearRegression  #
    # # # # # # # # # # #

    param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
    regr = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    regr.fit(X_train, y_train)

    # Predict test set
    y_pred = regr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'LinearRegression', 'Crime Dataset. Feature selected & MinMaxScaled.')

    # # # # # # # # #
    # SVR           #
    # # # # # # # # #

    #
    svr = GridSearchCV(SVR(kernel='rbf'), cv=5, n_jobs=-1, verbose=1,
                       param_grid={"C": np.logspace(-5, 5, num=2, base=2), "gamma": np.logspace(-5, 5, num=2, base=2)})
    # Fit Model
    svr.fit(X_train, y_train)

    # Predict test set
    y_pred = svr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'SVR', 'Crime Dataset. Feature selected & MinMaxScaled.')
    # Print params
    print('SVR Params: {}'.format(svr.best_params_))

    # # # # # # # # # # # # # #
    # RandomForestRegressor   #
    # # # # # # # # # # # # # #

    param_grid = {
        "n_estimators": [10, 20, 30, 50, 100],
        "min_samples_split": [2, 4, 8, 10, 12, 14, 16],
    }

    # GridSearch in param_grid
    rfr = GridSearchCV(RandomForestRegressor(), cv=5, n_jobs=-1, verbose=1, param_grid=param_grid)
    # Fit Model
    rfr.fit(X_train, y_train)

    # Predict test set
    y_pred = rfr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'RandomForestRegressor', 'Crime Dataset. Feature selected & MinMaxScaled.')

    # Print best params
    print('RandomForestsRegressor Params: {}'.format(rfr.best_params_))

    # # # # # # # # # # # # # #
    # MLPRegressor             #
    # # # # # # # # # # # # # #

    param_grid = {'solver': ['lbfgs'],
                  'max_iter': [1000, 1500, 2000],
                  'alpha': 10.0 ** -np.arange(1, 10),
                  'hidden_layer_sizes': np.arange(10, 15)}
    mlpr = GridSearchCV(MLPRegressor(), cv=5, param_grid=param_grid, n_jobs=-1, verbose=1)

    # Fit Model
    mlpr.fit(X_train, y_train)

    # Prediction
    y_pred = mlpr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'MLPRegressor', 'Crime Dataset. Feature selected & MinMaxScaled.')

    # Print best params
    print('RandomForestsRegressor Params: {}'.format(mlpr.best_params_))


def predict_bike():
    df = pd.read_csv('C:/Projects/University/Machine Learning/ml-regression/data/raw/BikeSharing/bikesharing_feature.csv')

    X = df.drop(['cnt'], axis=1)
    y = df['cnt']
    # # Validation set should be the test set. But we don't have the solutions for that one
    # X_validation = df.loc[2018:].drop(['weekly_infections'], axis=1)
    # y_validation = df.loc[2018:]['weekly_infections']

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # # # # # # # # # # #
    # LinearRegression  #
    # # # # # # # # # # #

    param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
    regr = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    regr.fit(X_train, y_train)

    # Predict test set
    y_pred = regr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'LinearRegression', 'Bike Sharing Dataset. Raw Data.')

    # Print params
    print('LinearRegression Params: {}'.format(regr.best_params_))

    # # Predict validation set
    # print('-' * 40)
    # print('Validation scores:')
    # y_val_pred = regr.predict(X_validation)
    # print_measurement_scores(y_validation, y_val_pred)

    # # # # # # # # #
    # SVR           #
    # # # # # # # # #

    #
    svr = GridSearchCV(SVR(kernel='rbf'), cv=5, n_jobs=-1, verbose=1,
                       param_grid={"C": np.logspace(-5, 5, num=5, base=2), "gamma": np.logspace(-5, 5, num=5, base=2)})
    # Fit Model
    svr.fit(X_train, y_train)

    # Predict test set
    y_pred = svr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'SVR', 'Bike Sharing Dataset. Raw Data.')
    # Print params
    print('SVR Params: {}'.format(svr.best_params_))

    # # # # # # # # # # # # # #
    # RandomForestRegressor   #
    # # # # # # # # # # # # # #

    param_grid = {
        "n_estimators": [10, 20, 30, 50, 100],
        "min_samples_split": [2, 4, 8, 10, 12, 14, 16],
    }

    # GridSearch in param_grid
    rfr = GridSearchCV(RandomForestRegressor(), cv=5, n_jobs=-1, verbose=1, param_grid=param_grid)
    # Fit Model
    rfr.fit(X_train, y_train)

    # Predict test set
    y_pred = rfr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'RandomForestRegressor', 'Bike Sharing Dataset. Raw Data.')

    # Print best params
    print('RandomForestsRegressor Params: {}'.format(rfr.best_params_))

    # # # # # # # # # # # # # #
    # MLPRegressor             #
    # # # # # # # # # # # # # #

    param_grid = {'solver': ['lbfgs'],
                  'max_iter': [1000, 1500, 2000],
                  'alpha': 10.0 ** -np.arange(1, 10),
                  'hidden_layer_sizes': np.arange(10, 15)}
    mlpr = GridSearchCV(MLPRegressor(), cv=5, param_grid=param_grid, n_jobs=-1, verbose=2)

    # Fit Model
    mlpr.fit(X_train, y_train)

    # Prediction
    y_pred = mlpr.predict(X_test)
    print_measurement_scores(y_test, y_pred, 'MLPRegressor', 'Bike Sharing Dataset. Raw Data.')

    # Print best params
    print('RandomForestsRegressor Params: {}'.format(mlpr.best_params_))


def main():
    # predict_MinMaxScaler()
    # predict_StandardScaler()
    predict_bike()


if __name__ == '__main__':
    main()
    sys.exit(0)
