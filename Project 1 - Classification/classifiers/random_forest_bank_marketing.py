import datetime
import sys
import time
from pprint import pprint

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

from config import *
from utils import get_k_best_features, get_best_features_with_decision_tress, get_dataset_scaled, \
    get_dataset_normalized, get_not_correlated_features, get_dataset_min_max_scaled, get_perf


def random_forests_on_breast_cancer(preprocess_method: str=None,
                                    feature_selection: str=None,
                                    feature_param: float=None,
                                    search_params: str=None):
    dataset_name = 'bank_marketing'
    timestamp = str(datetime.datetime.utcnow().replace(microsecond=0)).replace(' ', '').replace('-', '')

    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, dataset.columns.get_loc('class')].map(lambda x: 1 if x == 'B' else 0)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True)

    columns_without_ID = list(x_train.columns)
    columns_without_ID.remove('ID')

    # variable used to make the names of csvs more descriptive
    more_info_name = ''
    # if no preprocessing set, use raw dataset
    if not preprocess_method:
        print('Using raw dataset for breast cancer random_forests...')
        more_info_name += '_raw'
        x_train = x_train[columns_without_ID]
        x_test = x_test[columns_without_ID]

    elif preprocess_method == 'normalize':
        print('Using normalized dataset for breast cancer random_forests...')
        more_info_name += '_normalized'
        x_train = get_dataset_normalized(x_train[columns_without_ID])
        x_test = get_dataset_normalized(x_test[columns_without_ID])

    elif preprocess_method == 'scale':
        print('Using scaled dataset for breast cancer knn...')
        more_info_name += '_scaled'
        x_train = get_dataset_scaled(x_train[columns_without_ID])
        x_test = get_dataset_scaled(x_test[columns_without_ID])

    elif preprocess_method == 'min_max_scale':
        print('Using min_max_scaled dataset for breast cancer knn...')
        more_info_name += '_minmaxscaled'
        x_train = get_dataset_min_max_scaled(x_train[columns_without_ID])
        x_test = get_dataset_min_max_scaled(x_test[columns_without_ID])
    else:
        raise ValueError('Unknown pre-processing method!')

    use_features = []

    # if no feature selection set, use the default columns
    if not feature_selection:
        print('No feature selection. Using columns (excluding ID)')
        more_info_name += '_allfeatures'
        # do not include ID on the model (it makes no sense)
        use_features = columns_without_ID

    elif feature_selection == 'k_best':
        number_of_features = int(feature_param) if feature_param else 10
        more_info_name += '_{}best'.format(number_of_features)

        use_features = get_k_best_features(x_train[columns_without_ID], y_train, K=number_of_features)

    elif feature_selection == 'random_forests':
        more_info_name += '_decisiontrees_importance'
        use_features = get_best_features_with_decision_tress(
            x_train[columns_without_ID], y_train,
            dataset_name=breast_cancer_data_folder / 'breast_cancer_random_forests')

    elif feature_selection == 'drop_correlated':
        if feature_param > 1:
            raise ValueError('Threshold not allowed to be more than 1.0')
        more_info_name += '_drop_noncorrelated{}'.format(feature_param).replace('.', 'dot')

        use_features = get_not_correlated_features(x_train[columns_without_ID], threshold=feature_param)

    dt_classifier = RandomForestClassifier()

    if not search_params:
        print('No search params. Applying n_neighbors and weight only')
        # if no method defined, use parameters
        dt_classifier = RandomForestClassifier()
        start_time = time.time()

        print('Fitting/training the dt model...')
        dt_classifier.fit(x_train[use_features], y_train)

        print('Predicting the training data...')
        predictions = dt_classifier.predict(x_test[use_features])

        elapsed_time = time.time() - start_time  # gives you the duration in seconds

        get_perf(y_test.values, predictions)
        print('It took {} seconds in total'.format(elapsed_time))

    elif search_params == 'grid_search':
        n_estimators = [int(x) for x in pd.np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in pd.np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        grid_search = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap
                       }
        pprint(grid_search)
        print('Starting Grid search with cv=5 and the parameters mentioned just before. Sorting by accuracy...')

        # Grid search of parameters, using 5 fold cross validation,
        random_forests_grid_search = GridSearchCV(estimator=dt_classifier,
                                                  param_grid=grid_search,
                                                  cv=5,
                                                  verbose=0,
                                                  refit='accuracy',
                                                  n_jobs=-1,
                                                  scoring=['accuracy', 'recall', 'precision', 'f1', 'roc_auc'],
                                                  return_train_score=True)

        # Fit the grid search model
        print('Fitting/training the grid search in decision trees ...')
        random_forests_grid_search.fit(x_train[use_features], y_train)

        try:
            prediction_dataframe = pd.DataFrame(random_forests_grid_search.cv_results_)
            prediction_dataframe.to_csv(
                breast_cancer_data_folder / 'grid_search_random_forests_breast_cancer_{}{}.csv'.format(more_info_name,
                                                                                                       timestamp),
                index=False)
            print('Stored file as: grid_search_random_forests_breast_cancer_{}{}.csv'.format(more_info_name, timestamp))
        except Exception as e:
            print('An error occurred while storing grind search results')
            pass

        predictions = random_forests_grid_search.best_estimator_.predict(x_test[use_features])
        get_perf(y_test.values, predictions)
        print(
            'Best parameters found by grid_search: {}'.format(random_forests_grid_search.best_estimator_.get_params()))
        return
    else:
        raise ValueError('Unknown search parameter')


def random_forests_on_breast_cancer_randomized_search():
    dataset_name = 'breast_cancer'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('class')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    # Number of trees in random forest
    n_estimators = [int(x) for x in pd.np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in pd.np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap
                   }
    pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=30, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    # do not include ID on the model (it makes no sense)
    columns_without_ID = list(x_train.columns)
    columns_without_ID.remove('ID')

    # k_best_columns = get_k_best_features(x_train[columns_without_ID], y_train, K=10)
    k_best_columns = get_best_features_with_decision_tress(x_train[columns_without_ID], y_train,
                                                           dataset_name=breast_cancer_data_folder / 'breast_cancer')

    # Fit the random search model
    print('Fitting/training the random forests model...')
    rf_random.fit(x_train[k_best_columns], y_train)

    print('Predicting the training data...')
    predictions = rf_random.best_estimator_.predict(dataset_test[k_best_columns])

    print('Storing random forests results to csv predictions for {}...'.format(dataset_name))
    # store prediction
    prediction_dataframe = pd.DataFrame({'ID': dataset_test['ID'], 'class': predictions})
    prediction_dataframe.to_csv(breast_cancer_data_folder / 'prediction-random-forests.csv', index=False)


def random_forests_on_breast_cancer_grid_search():
    dataset_name = 'breast_cancer'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('class')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    # Number of trees in random forest
    n_estimators = [int(x) for x in pd.np.linspace(start=100, stop=2000, num=8)]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in pd.np.linspace(4, len(x_train.columns), num=5)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    criterion = ['gini', 'entropy']

    # Create the grid
    grid_search = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'criterion': criterion
                   }
    pprint(grid_search)

    # Use the random grid to search for best hyper parameters
    rf = RandomForestClassifier()

    # Random search of parameters, using 3 fold cross validation,
    rf_grid_search = GridSearchCV(estimator=rf, param_grid=grid_search, cv=3, verbose=2, refit='accuracy',
                                  n_jobs=-1, scoring=['accuracy', 'recall', 'precision', 'f1', 'roc_auc'])

    # do not include ID on the model (it makes no sense)
    columns_without_ID = list(x_train.columns)
    columns_without_ID.remove('ID')

    # k_best_columns = get_k_best_features(x_train[columns_without_ID], y_train, K=10)
    # k_best_columns = get_k_best_features(x_train[columns_without_ID], y_train, K=10)
    x_train = get_not_correlated_features(x_train[columns_without_ID], 0.8)
    # k_best_columns = columns_without_ID

    # Fit the random search model
    print('Fitting/training the random forests model...')
    rf_grid_search.fit(x_train, y_train.map(lambda x: 1 if x == 'B' else 0))

    try:
        prediction_dataframe = pd.DataFrame(rf_grid_search.cv_results_)
        prediction_dataframe.to_csv(breast_cancer_data_folder / 'grid_search_random_forest.csv', index=False)
    except Exception as e:
        print('An erro occurred while storing grind search results')
        pass

    print('Predicting the training data...')
    predictions = rf_grid_search.best_estimator_.predict(dataset_test[x_train.columns.values])

    print('Best parameters found by grid_search: {}'.format(rf_grid_search.best_estimator_.get_params()))

    print('Storing random forests results to csv predictions for {}...'.format(dataset_name))
    # store prediction
    prediction_dataframe = pd.DataFrame({'ID': dataset_test['ID'],
                                         'class': ['B' if val == 1 else 'M' for val in predictions]})

    prediction_dataframe.to_csv(breast_cancer_data_folder / 'prediction-random-forests.csv', index=False)


@click.command()
@click.option('--preprocess_method', '-p', help='Whether to preprocess dataset or not', required=False)
@click.option('--feature_selection', '-fs', help='What feature selection to use', required=False)
@click.option('--feature_param', '-fp', help='Feature param for feature selection. '
                                             'K for k_best, and threshold for drop_not_correlated', required=False)
@click.option('--search_params', '-s', help='Whether to use search params or not', required=False, default=None)
def main(preprocess_method, feature_selection, feature_param, search_params):
    # shit to make shell scripting run
    if preprocess_method == 'None':
        preprocess_method = None

    if feature_selection == 'None':
        feature_selection = None

    if feature_param == 'None':
        feature_param = None

    if search_params == 'None':
        search_params = None

    print('Starting the traning of dataset BREAST_CANCER with '
          ' preprocess: {}, feature_Selection: {}, '
          'feature_number: {} ,search_params {}'.format(preprocess_method, feature_selection,
                                                        feature_param, search_params))

    random_forests_on_breast_cancer(preprocess_method=preprocess_method,
                                    feature_selection=feature_selection,
                                    feature_param=float(feature_param) if feature_param else None,
                                    search_params=search_params)

    print('Done!')


if __name__ == '__main__':
    main()
    sys.exit(0)
