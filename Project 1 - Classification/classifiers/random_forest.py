import sys
from pprint import pprint

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

from config import *
from utils import get_k_best_features, get_best_features_with_decision_tress, get_dataset_scaled, \
    get_dataset_normalized, get_not_correlated_features


def random_forests_on_bank_marketing(n_estimators: int, max_depth: int):
    dataset_name = 'bank_marketing'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path'])

    # do not include the target column (which is the last one) on data dataframe
    data = dataset_train.iloc[:, :-1]
    labels = dataset_train.iloc[:, dataset_train.columns.get_loc('y')]

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

    classifier: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    print('Fitting/training the random forests model...')
    classifier.fit(x_train, y_train)

    accuracy = classifier.score(x_test, y_test) * 100

    print('Classifiers accuracy: {0:.3f}%'.format(accuracy))


def random_forests_on_image_segmentation(n_estimators: int, max_depth: int):
    dataset_name = 'image_segmentation'

    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])
    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    x_train = dataset_train.iloc[:, 1:]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('CLASS')]

    x_test = dataset_test.iloc[:, 1:]
    y_test = dataset_test.iloc[:, dataset_test.columns.get_loc('CLASS')]

    classifier: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    print('Fitting/training the naive bayes model')
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test) * 100

    print('Classifiers accuracy: {0:.3f}%'.format(accuracy))


def random_forests_on_breast_cancer(n_estimators: int, max_depth: int):
    dataset_name = 'breast_cancer'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('class')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    classifier: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # do not include ID on the model (it makes no sense)
    columns_without_ID = list(x_train.columns)
    columns_without_ID.remove('ID')

    print('Fitting/training the decision trees model...')
    classifier.fit(x_train[columns_without_ID], y_train)

    print('Predicting the training data...')
    predictions = classifier.predict(dataset_test[columns_without_ID])

    print('Storing decision tree csv predictions for {}...'.format(dataset_name))
    # store prediction
    prediction_dataframe = pd.DataFrame({'ID': dataset_test['ID'], 'class': predictions})
    prediction_dataframe.to_csv(breast_cancer_data_folder / 'prediction-random-forests.csv', index=False)


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
@click.option('--dataset_name', '-d', help='Dataset on which you want to run the classification algorithm',
              required=False, default='breast_cancer')
@click.option('--n_estimators', '-n', help='Number of samples', required=False, default=100)
@click.option('--max_depth', '-m', help='Maximum depth for decision trees in forest', required=False, default=4)
def main(dataset_name, n_estimators, max_depth):
    print('Starting the traning of dataset {} with n_estimators={} and max_depth={}'.format(dataset_name.upper(),
                                                                                            n_estimators,
                                                                                            max_depth))

    if dataset_name == 'bank_marketing':
        random_forests_on_bank_marketing()
    elif dataset_name == 'breast_cancer':
        random_forests_on_breast_cancer_grid_search()
    elif dataset_name == 'image_segmentation':
        random_forests_on_image_segmentation(n_estimators, max_depth)

    print('Done!')


if __name__ == '__main__':
    main()
    sys.exit(0)
