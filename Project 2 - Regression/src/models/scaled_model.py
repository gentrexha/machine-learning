import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pandas as pd
from pathlib import Path

from utils import load_raw_data, print_measurement_scores, DATASETS
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def predict_bike_sharing():
    """
    Predicts the training data for the bike_sharing dataset.
    Techniques used: LinearRegression, RandomForest Regression, SVR, MLPRegressor
    Scalings used: Standard, MinMax
    """

    # Load and split data
    df = load_raw_data('bike_sharing')
    X = df.drop(['cnt', 'dteday', 'id'], axis=1)
    y = df['cnt']

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Prepare Kaggle test set
    kaggle_df = pd.read_csv(DATASETS['bike_sharing']['test'])
    X_kaggle = kaggle_df.drop(['dteday', 'id'], axis=1)

    # Scale methods to try
    scalers = [MinMaxScaler()]

    # # # # # # # # # # #
    # LinearRegression  #
    # # # # # # # # # # #

    for scaler in scalers:
        # Scale train and test data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        # X_validation_scaled = scaler.fit_transform(X_validation)

        # LinearRegression
        param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
        regr = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        regr.fit(X_train_scaled, y_train)

        # Predict test set
        y_pred = regr.predict(X_test_scaled)
        print_measurement_scores(y_test, y_pred, 'LinearRegression', 'Bike Sharing Dataset. ' +
                                 'StandardScaler' if scalers.index(scaler) == 0 else 'MinMaxScaler')

        # Print params
        print('LinearRegression Params: {}'.format(regr.best_params_))

        # # Predict validation set
        # print('-' * 40)
        # print('Validation scores:')
        # y_val_pred = regr.predict(X_validation_scaled)
        # print_measurement_scores(y_validation, y_val_pred)

    # # # # # # # # #
    # SVR           #
    # # # # # # # # #

    for scaler in scalers:
        # Scale train and test data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        # X_validation_scaled = scaler.fit_transform(X_validation)

        # GridSearch in np.logspace
        svr = GridSearchCV(SVR(kernel='rbf'), cv=5, n_jobs=-1, verbose=1,
                           param_grid={"C": np.logspace(-5, 5, num=5, base=2),
                                       "gamma": np.logspace(-5, 5, num=5, base=2)})
        # Fit Model
        svr.fit(X_train_scaled, y_train)

        # Predict test set
        y_pred = svr.predict(X_test_scaled)
        print_measurement_scores(y_test, y_pred, 'SVR', 'Bike Sharing Dataset. ' +
                                 'StandardScaler' if scalers.index(scaler) == 0 else 'MinMaxScaler')

        # # Predict validation set
        # print('-' * 40)
        # print('Validation scores:')
        # y_val_pred = regr.predict(X_validation_scaled)
        # print_measurement_scores(y_validation, y_val_pred)

        # Print best params
        print('SVR Params: {}'.format(svr.best_params_))

    # # # # # # # # # # # # # #
    # RandomForestRegressor   #
    # # # # # # # # # # # # # #

    for scaler in scalers:
        # Scale train and test data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        X_kaggle_scaled = scaler.fit_transform(X_kaggle)

        # default LinearRegression Model without any parameter tuning
        param_grid = {
            "n_estimators": [100, 200],  # [100, 200, 300, 400, 500]
            "min_samples_split": [2, 4, 8, 10],  # [2, 4, 8, 10, 12, 14, 16]
        }
        rfr = GridSearchCV(RandomForestRegressor(), cv=5, n_jobs=-1, verbose=1, param_grid=param_grid)
        rfr.fit(X_train_scaled, y_train)

        # Predict test set
        y_pred = rfr.predict(X_test_scaled)
        print_measurement_scores(y_test, y_pred, 'RandomForestRegressor', 'Bike Sharing Dataset. ' +
                                 'StandardScaler' if scalers.index(scaler) == 0 else 'MinMaxScaler')

        # Predict kaggle set
        y_kaggle_pred = rfr.predict(X_kaggle_scaled)

        # Save id and predictions to CSV
        # TODO: Export ID and y_kaggle_pred to CSV
        kaggle_predictions = pd.DataFrame({'id': kaggle_df['id'], 'cnt': y_kaggle_pred})
        kaggle_predictions[['id', 'cnt']].to_csv(Path('C:/Projects/University/Machine Learning/ml-regression/data/external/kaggle_predictions_'+str(scalers.index(scaler))+'.csv'), index=False)
        print('Saved kaggle predictions succesfully!')

        # Print best params
        print('RandomForestsRegressor Params: {}'.format(rfr.best_params_))

    # # # # # # # # # # # # # #
    # MLPRegressor             #
    # # # # # # # # # # # # # #

    for scaler in scalers:
        # Scale train and test data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        # X_validation_scaled = scaler.fit_transform(X_validation)

        param_grid = {'solver': ['lbfgs'],
                      'max_iter': [1000, 1500, 2000],
                      'alpha': 10.0 ** -np.arange(1, 10),
                      'hidden_layer_sizes': np.arange(10, 15)}
        mlpr = GridSearchCV(MLPRegressor(), cv=5, param_grid=param_grid, n_jobs=-1, verbose=1)

        # Fit Model
        mlpr.fit(X_train_scaled, y_train)

        # Prediction
        y_pred = mlpr.predict(X_test_scaled)
        print_measurement_scores(y_test, y_pred, 'MLPRegressor',
                                 'Bike Sharing Dataset. ' + 'StandardScaler'
                                 if scalers.index(scaler) == 0 else 'MinMaxScaler')

        # Print best params
        print('RandomForestsRegressor Params: {}'.format(mlpr.best_params_))


def predict_crime():
    """
    Predicts the training data for the crime dataset.
    Techniques used: LinearRegression, RandomForest Regression, SVR, MPLPRegression
    Scalings used: Standard, MinMax
    """

    df = pd.read_csv(DATASETS['crime']['filled'])

    X = df.drop(['ViolentCrimesPerPop'], axis=1)
    y = df['ViolentCrimesPerPop']
    # # Validation set should be the test set. But we don't have the solutions for that one
    # X_validation = df.loc[2018:].drop(['weekly_infections'], axis=1)
    # y_validation = df.loc[2018:]['weekly_infections']

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Scale methods to try
    scalers = [MinMaxScaler()]

    # # # # # # # # # # #
    # LinearRegression  #
    # # # # # # # # # # #

    # for scaler in scalers:
    #     # Scale train and test data
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.fit_transform(X_test)
    #     # X_validation_scaled = scaler.fit_transform(X_validation)
    #
    #     # LinearRegression
    #     param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
    #     regr = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    #     regr.fit(X_train_scaled, y_train)
    #
    #     # Predict test set
    #     y_pred = regr.predict(X_test_scaled)
    #     print_measurement_scores(y_test, y_pred, 'LinearRegression', 'Crime dataset. ' +
    #                              'StandardScaler' if scalers.index(scaler) == 0 else 'MinMaxScaler')
    #
    #     # Print params
    #     print('LinearRegression Params: {}'.format(regr.best_params_))
    #
    #     # # Predict validation set
    #     # print('-' * 40)
    #     # print('Validation scores:')
    #     # y_val_pred = regr.predict(X_validation_scaled)
    #     # print_measurement_scores(y_validation, y_val_pred)

    # # # # # # # # #
    # SVR           #
    # # # # # # # # #

    # for scaler in scalers:
    #     # Scale train and test data
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.fit_transform(X_test)
    #     # X_validation_scaled = scaler.fit_transform(X_validation)
    #
    #     # GridSearch in np.logspace
    #     svr = GridSearchCV(SVR(kernel='rbf'), cv=5, n_jobs=-1, verbose=1,
    #                        param_grid={"C": np.logspace(-5, 5, num=5, base=2),
    #                                    "gamma": np.logspace(-5, 5, num=5, base=2)})
    #     # Fit Model
    #     svr.fit(X_train_scaled, y_train)
    #
    #     # Predict test set
    #     y_pred = svr.predict(X_test_scaled)
    #     print_measurement_scores(y_test, y_pred, 'SVR', 'Crime Dataset. ' +
    #                              'StandardScaler' if scalers.index(scaler) == 0 else 'MinMaxScaler')
    #
    #     # # Predict validation set
    #     # print('-' * 40)
    #     # print('Validation scores:')
    #     # y_val_pred = regr.predict(X_validation_scaled)
    #     # print_measurement_scores(y_validation, y_val_pred)
    #
    #     # Print best params
    #     print('SVR Params: {}'.format(svr.best_params_))

    # # # # # # # # # # # # # #
    # RandomForestRegressor   #
    # # # # # # # # # # # # # #

    # for scaler in scalers:
    #     # Scale train and test data
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.fit_transform(X_test)
    #     # X_validation_scaled = scaler.fit_transform(X_validation)
    #
    #     # default LinearRegression Model without any parameter tuning
    #     param_grid = {
    #         "n_estimators": [100, 200],
    #         "min_samples_split": [2, 4, 8, 10],
    #     }
    #     rfr = GridSearchCV(RandomForestRegressor(), cv=5, n_jobs=-1, verbose=1, param_grid=param_grid)
    #     rfr.fit(X_train_scaled, y_train)
    #
    #     # Predict test set
    #     y_pred = rfr.predict(X_test_scaled)
    #     print_measurement_scores(y_test, y_pred, 'RandomForestRegressor', 'Crime Dataset. ' +
    #                              'StandardScaler' if scalers.index(scaler) == 0 else 'MinMaxScaler')
    #
    #     # # Predict validation set
    #     # print('-' * 40)
    #     # print('Validation scores:')
    #     # y_val_pred = regr.predict(X_validation_scaled)
    #     # print_measurement_scores(y_validation, y_val_pred)
    #
    #     # Print best params
    #     print('RandomForestsRegressor Params: {}'.format(rfr.best_params_))

    # # # # # # # # # # # # # #
    # MLRegressor             #
    # # # # # # # # # # # # # #

    for scaler in scalers:
        # Scale train and test data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        # X_validation_scaled = scaler.fit_transform(X_validation)

        param_grid = {'solver': ['lbfgs'],
                      'max_iter': [1000, 1500, 2000],
                      'alpha': 10.0 ** -np.arange(1, 10),
                      'hidden_layer_sizes': np.arange(10, 15)}
        mlpr = GridSearchCV(MLPRegressor(), cv=5, param_grid=param_grid, n_jobs=-1, verbose=1)

        # Fit Model
        mlpr.fit(X_train_scaled, y_train)

        # Prediction
        y_pred = mlpr.predict(X_test_scaled)
        print_measurement_scores(y_test, y_pred, 'MLPRegressor', 'Crime Dataset. ' +
                                                                 'StandardScaler' if scalers.index(scaler) == 0 else 'MinMaxScaler')

        # Print best params
        print('RandomForestsRegressor Params: {}'.format(mlpr.best_params_))


def main():
    # predict_bike_sharing()
    predict_crime()


if __name__ == '__main__':
    main()
    sys.exit(0)
