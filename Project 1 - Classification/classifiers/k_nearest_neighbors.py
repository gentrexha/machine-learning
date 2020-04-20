import sys

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from config import *
from utils import store_correlation_heatmap, get_k_best_features, get_dataset_normalized


def perform_knn_on_bank_marketing(n_neighbors: int, weight: str):
    dataset_name = 'bank_marketing'
    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path'])

    # do not include the target column (which is the last one) on data dataframe
    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, dataset.columns.get_loc('y')]

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

    x_train = get_dataset_normalized(x_train)
    x_test = get_dataset_normalized(x_test)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)

    print('Fitting/training the knn model')
    knn.fit(x_train, y_train)
    accuracy = knn.score(x_test, y_test) * 100

    print('Classifiers accuracy: {0:.3f}%'.format(accuracy))


def perform_knn_on_image_segmentation(n_neighbors: int, weight: str):
    dataset_name = 'image_segmentation'

    train_dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the first one) on data dataframe
    x_train = train_dataset.iloc[:, 1:]
    y_train = train_dataset.iloc[:, train_dataset.columns.get_loc('CLASS')]

    test_dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    x_test = test_dataset.iloc[:, 1:]
    y_test = test_dataset.iloc[:, train_dataset.columns.get_loc('CLASS')]

    x_train = get_dataset_normalized(x_train)
    x_test = get_dataset_normalized(x_test)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)

    print('Fitting/training the knn model')
    knn.fit(x_train, y_train)
    accuracy = knn.score(x_test, y_test) * 100

    print('Classifiers accuracy: {0:.3f}%'.format(accuracy))


def perform_knn_on_breast_cancer(n_neighbors: int, weight: str):
    """
    Performs knn on the breast_cancer dataset and stores the results as csv in the format that kaggle accepts.

    Procedure:
    - Fetches both training and testing datasets
    - Normalize both datasets (but do not include ID's, since that is something we do not use in our model - and need it
      untouched for mapping)
    - Train model on train_dataset_normalized
    - Predict on test_dataset_normalized
    - Store predictions as csv
    """
    dataset_name = 'breast_cancer'
    train_dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = train_dataset.iloc[:, :-1]
    y_train = train_dataset.iloc[:, train_dataset.columns.get_loc('class')]

    test_dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    # do not include ID on the model (it makes no sense)
    columns_without_ID = list(x_train.columns)
    columns_without_ID.remove('ID')

    store_correlation_heatmap(dataset=test_dataset[columns_without_ID],
                              figure_name=breast_cancer_data_folder / 'breast_cancer_rawdata')

    best_k_features = get_k_best_features(x_train[columns_without_ID], y_train, K=10)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)

    # normalize values in train and test datasets (but do not include ID)
    x_train_normalized_no_ids = get_dataset_normalized(x_train[best_k_features])
    test_dataset_normalized_no_ids = get_dataset_normalized(test_dataset[best_k_features])

    print('Fitting/training the knn model...')
    knn.fit(x_train_normalized_no_ids, y_train)

    print('Predicting the training data...')
    predictions = knn.predict(test_dataset_normalized_no_ids)

    print('Storing the predictions csv file...')
    # store prediction
    prediction_dataframe = pd.DataFrame({'ID': test_dataset['ID'], 'class': predictions})
    prediction_dataframe.to_csv(breast_cancer_data_folder / 'prediction-knn.csv', index=False)


@click.command()
@click.option('--dataset_name', '-d', help='Dataset on which you want to run the classification algorithm',
              required=False, default='breast_cancer')
@click.option('--n_neighbors', '-k', help='Number of neighbors to consider', required=False, default=1)
@click.option('--weight', '-w', help='Weight function for distance', required=False, default='uniform',
              type=click.Choice(['uniform', 'distance']))
def main(dataset_name, n_neighbors, weight):
    print('Starting the traning of dataset {} with k={} and weight_function={}'.format(dataset_name.upper(),
                                                                                       n_neighbors,
                                                                                       weight))
    if dataset_name == 'bank_marketing':
        perform_knn_on_bank_marketing(n_neighbors, weight)
    elif dataset_name == 'breast_cancer':
        perform_knn_on_breast_cancer(n_neighbors, weight)
    elif dataset_name == 'image_segmentation':
        perform_knn_on_image_segmentation(n_neighbors, weight)

    print('Done!')


if __name__ == '__main__':
    main()
    sys.exit(0)
