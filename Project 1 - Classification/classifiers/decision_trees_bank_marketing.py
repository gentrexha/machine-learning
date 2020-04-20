import datetime
import sys
import time
from pprint import pprint

import click
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from config import *
from utils import get_k_best_features, get_dataset_normalized, get_dataset_scaled, \
    get_best_features_with_decision_tress, get_not_correlated_features, get_dataset_min_max_scaled, get_perf


def perform_decision_trees_on_bank_marketing(preprocess_method: str=None,
                                             feature_selection: str=None,
                                             feature_param: float=None,
                                             search_params: str=None):
    dataset_name = 'bank_marketing'
    timestamp = str(datetime.datetime.utcnow().replace(microsecond=0)).replace(' ', '').replace('-', '')

    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path'])

    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, dataset.columns.get_loc('y')].map(lambda x: 1 if x == 'yes' else 0)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True)

    # variable used to make the names of csvs more descriptive
    more_info_name = ''
    # if no preprocessing set, use raw dataset
    if not preprocess_method:
        print('Using raw dataset for bank_marketing decision_trees...')
        more_info_name += '_raw'

    elif preprocess_method == 'normalize':
        print('Using normalized dataset for bank_marketing decision_trees...')
        more_info_name += '_normalized'
        x_train = get_dataset_normalized(x_train)
        x_test = get_dataset_normalized(x_test)

    elif preprocess_method == 'scale':
        print('Using scaled dataset for bank_marketing decision trees...')
        more_info_name += '_scaled'
        x_train = get_dataset_scaled(x_train)
        x_test = get_dataset_scaled(x_test)

    elif preprocess_method == 'min_max_scale':
        print('Using min_max_scaled dataset for bank_marketing decision trees...')
        more_info_name += '_minmaxscaled'
        x_train = get_dataset_min_max_scaled(x_train)
        x_test = get_dataset_min_max_scaled(x_test)
    else:
        raise ValueError('Unknown pre-processing method!')

    use_features = []

    # if no feature selection set, use the default columns
    if not feature_selection:
        print('No feature selection. Using columns (excluding ID)')
        more_info_name += '_allfeatures'
        # do not include ID on the model (it makes no sense)
        use_features = x_train.columns.values

    elif feature_selection == 'k_best':
        number_of_features = int(feature_param) if feature_param else 10
        more_info_name += '_{}best'.format(number_of_features)

        use_features = get_k_best_features(x_train, y_train, K=number_of_features)

    elif feature_selection == 'decision_trees':
        more_info_name += '_decisiontrees_importance'
        use_features = get_best_features_with_decision_tress(
            x_train, y_train, dataset_name=breast_cancer_data_folder / 'breast_cancer_decision_trees')

    elif feature_selection == 'drop_correlated':
        if feature_param > 1:
            raise ValueError('Threshold not allowed to be more than 1.0')
        more_info_name += '_drop_noncorrelated{}'.format(feature_param).replace('.', 'dot')

        use_features = get_not_correlated_features(x_train, threshold=feature_param)

    dt_classifier = DecisionTreeClassifier()

    if not search_params:
        print('No search params. Applying n_neighbors and weight only')
        # if no method defined, use parameters
        dt_classifier = DecisionTreeClassifier(criterion='gini')
        start_time = time.time()

        print('Fitting/training the dt model...')
        dt_classifier.fit(x_train[use_features], y_train)

        print('Predicting the training data...')
        predictions = dt_classifier.predict(x_test[use_features])

        elapsed_time = time.time() - start_time  # gives you the duration in seconds

        get_perf(y_test.values, predictions)
        print('It took {} seconds in total'.format(elapsed_time))

    elif search_params == 'grid_search':

        # Create the grid
        grid_search = {
            'max_depth': [int(x) for x in range(1, len(x_train.columns), 1)],
            'max_features': [int(x) for x in range(1, len(x_train.columns), 1)],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }

        pprint(grid_search)
        print('Starting Grid search with cv=5 and the parameters mentioned just before. Sorting by accuracy...')

        # Grid search of parameters, using 5 fold cross validation,
        decision_trees_grid_search = GridSearchCV(estimator=dt_classifier,
                                                  param_grid=grid_search,
                                                  cv=5,
                                                  verbose=0,
                                                  refit='accuracy',
                                                  n_jobs=-1,
                                                  scoring=['accuracy', 'recall', 'precision', 'f1', 'roc_auc'],
                                                  return_train_score=True)

        # Fit the grid search model
        print('Fitting/training the grid search in decision trees ...')
        decision_trees_grid_search.fit(x_train[use_features], y_train)

        try:
            prediction_dataframe = pd.DataFrame(decision_trees_grid_search.cv_results_)
            prediction_dataframe.to_csv(
                breast_cancer_data_folder / 'grid_search_decision_trees_bank_marketing_{}{}.csv'.format(more_info_name,
                                                                                                        timestamp),
                index=False)
            print(
                'Stored file as: grid_search_decision_trees_bank_marketing_{}{}.csv'.format(more_info_name, timestamp))
        except Exception as e:
            print('An error occurred while storing grind search results')
            pass

        predictions = decision_trees_grid_search.best_estimator_.predict(x_test[use_features])
        get_perf(y_test.values, predictions)
        print('Best parameters found by grid_search: {}'.format(decision_trees_grid_search.best_estimator_.get_params()))
        return
    else:
        raise ValueError('Unknown search parameter')


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

    perform_decision_trees_on_bank_marketing(preprocess_method=preprocess_method,
                                             feature_selection=feature_selection,
                                             feature_param=float(feature_param) if feature_param else None,
                                             search_params=search_params)

    print('Done!')


if __name__ == '__main__':
    main()
    sys.exit(0)
