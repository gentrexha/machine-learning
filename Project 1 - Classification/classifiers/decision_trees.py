import sys

import click
import graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from config import *


def decision_trees_on_bank_marketing():
    """
        sk-learn implementation of decision trees does not accept categorical data, therefore we have
        to use the preprocessed version of the dataset.
        Normally, decision trees do not require categorical data to be encoded
    """
    dataset_name = 'bank_marketing'
    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path'])

    # do not include the target column (which is the last one) on data dataframe
    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, dataset.columns.get_loc('y')]

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
    classifier: DecisionTreeClassifier = DecisionTreeClassifier()

    print('Fitting/training the decision tree')
    classifier = classifier.fit(x_train, y_train)

    accuracy = classifier.score(x_test, y_test) * 100

    print('Classifiers accuracy: {0:.3f}%'.format(accuracy))

    print('Creating decision tree layout... (This will create a {dataset_name}.pdf on the main directory)')
    # output the tree visualized
    dot_data = export_graphviz(classifier, out_file=None, feature_names=data.columns)
    graph = graphviz.Source(dot_data)
    graph.render(dataset_name)

    print('Done!')


def decision_trees_on_image_segmentation():
    """
        sk-learn implementation of decision trees does not accept categorical data, therefore we have
        to use the preprocessed version of the dataset.
        Normally, decision trees do not require categorical data to be encoded
    """
    dataset_name = 'image_segmentation'

    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])
    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, 1:]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('CLASS')]

    x_test = dataset_test.iloc[:, 1:]
    y_test = dataset_test.iloc[:, dataset_test.columns.get_loc('CLASS')]

    classifier: DecisionTreeClassifier = DecisionTreeClassifier()

    print('Fitting/training the decision tree')
    classifier = classifier.fit(x_train, y_train)

    accuracy = classifier.score(x_test, y_test) * 100

    print('Classifiers accuracy: {0:.3f}%'.format(accuracy))
    print('Done!')


def decision_trees_on_breast_cancer():
    dataset_name = 'breast_cancer'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('class')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    classifier: DecisionTreeClassifier = DecisionTreeClassifier(criterion='gini')

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
    prediction_dataframe.to_csv(breast_cancer_data_folder / 'prediction-decision-trees.csv', index=False)


def decision_trees_on_kdd():
    dataset_name = 'kdd'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('TARGET_B')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    classifier: DecisionTreeClassifier = DecisionTreeClassifier(criterion='gini')

    # do not include ID on the model (it makes no sense)
    columns_without_ID = list(x_train.columns)
    columns_without_ID.remove('CONTROLN')

    print('Fitting/training the decision trees model...')
    classifier.fit(x_train[columns_without_ID], y_train)

    print('Predicting the training data...')
    predictions = classifier.predict(dataset_test[columns_without_ID])

    print('Storing decision tree csv predictions for {}...'.format(dataset_name))
    # store prediction
    prediction_dataframe = pd.DataFrame({'CONTROLN': dataset_test['CONTROLN'], 'TARGET_B': predictions})
    prediction_dataframe.to_csv(kdd_data_folder / 'prediction-decision-trees.csv', index=False)


@click.command()
@click.option('--dataset_name', '-d', help='Dataset on which you want to run the classification algorithm',
              required=False, default='image_segmentation')
def main(dataset_name):
    if dataset_name == 'bank_marketing':
        decision_trees_on_bank_marketing()
    elif dataset_name == 'breast_cancer':
        decision_trees_on_breast_cancer()
    elif dataset_name == 'kdd':
        decision_trees_on_kdd()
    elif dataset_name == 'image_segmentation':
        decision_trees_on_image_segmentation()

    print('Done!')


if __name__ == '__main__':
    main()
    sys.exit(0)
