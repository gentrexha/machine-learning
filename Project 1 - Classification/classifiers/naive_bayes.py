import sys

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from config import *


def perform_naive_bayes_on_bank_marketing(classifier):
    dataset_name = 'bank_marketing'
    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path'])

    # do not include the target column (which is the last one) on data dataframe
    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, dataset.columns.get_loc('y')]

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

    print('Fitting/training the naive bayes model')
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test) * 100

    print('Classifiers accuracy: {0:.3f}%'.format(accuracy))


def perform_naive_bayes_on_image_segmentation(classifier):
    dataset_name = 'image_segmentation'

    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])
    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    x_train = dataset_train.iloc[:, 1:]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('CLASS')]

    x_test = dataset_test.iloc[:, 1:]
    y_test = dataset_test.iloc[:, dataset_test.columns.get_loc('CLASS')]

    print('Fitting/training the naive bayes model')
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test) * 100

    print('Classifiers accuracy: {0:.3f}%'.format(accuracy))


def perform_naive_bayes_on_breast_cancer(classifier):
    """
    Performs naive bayes on the breast_cancer dataset and stores the results as csv in the format that kaggle accepts.
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

    print('Fitting/training the naive bayes model...')
    classifier.fit(x_train[columns_without_ID], y_train)

    print('Predicting the training data...')
    predictions = classifier.predict(test_dataset[columns_without_ID])

    print('Storing the predictions csv file...')
    # store prediction
    prediction_dataframe = pd.DataFrame({'ID': test_dataset['ID'], 'class': predictions})
    prediction_dataframe.to_csv(breast_cancer_data_folder / 'prediction-naive-bayes.csv', index=False)


def perform_naive_bayes_on_kdd(classifier):
    """
    Performs naive bayes on the kdd dataset and stores the results as csv in the format that kaggle accepts.
    """
    dataset_name = 'kdd'
    train_dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = train_dataset.iloc[:, :-1]
    y_train = train_dataset.iloc[:, train_dataset.columns.get_loc('TARGET_B')]

    test_dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    # do not include ID on the model (it makes no sense)
    columns_without_ID = list(x_train.columns)
    columns_without_ID.remove('CONTROLN')

    print('Fitting/training the naive bayes model...')
    classifier.fit(x_train[columns_without_ID], y_train)

    print('Predicting the training data...')
    predictions = classifier.predict(test_dataset[columns_without_ID])

    print('Storing the predictions csv file...')
    # store prediction
    prediction_dataframe = pd.DataFrame({'CONTROLN': test_dataset['CONTROLN'], 'TARGET_B': predictions})
    prediction_dataframe.to_csv(kdd_data_folder / 'prediction-naive-bayes.csv', index=False)


@click.command()
@click.option('--dataset_name', '-d', help='Dataset on which you want to run the classification algorithm',
              required=False, default='image_segmentation')
@click.option('--classifier_name', '-c', help='Which naive bayes classifier to use',
              required=False, type=click.Choice(['gaussian', 'bernuolli']), default='gaussian')
def main(dataset_name, classifier_name):
    print('Starting the traning of dataset {} with classifier {}'.format(dataset_name.upper(), classifier_name))

    if classifier_name == 'gaussian':
        classifier = GaussianNB()
    elif classifier_name == 'bernuolli':
        classifier = BernoulliNB()
    else:
        raise ValueError('Invalid naive bayes classifier given')

    if dataset_name == 'bank_marketing':
        perform_naive_bayes_on_bank_marketing(classifier)
    elif dataset_name == 'breast_cancer':
        perform_naive_bayes_on_breast_cancer(classifier)
    elif dataset_name == 'kdd':
        perform_naive_bayes_on_kdd(classifier)
    elif dataset_name == 'image_segmentation':
        perform_naive_bayes_on_image_segmentation(classifier)

    print('Done!')


if __name__ == '__main__':
    main()
    sys.exit(0)
