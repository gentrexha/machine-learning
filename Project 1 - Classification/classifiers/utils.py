import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from typing import Optional
from sklearn import metrics

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import *


def get_dataset_normalized(dataset: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe after normalizing it's values"""
    print('Normalizing dataset...')
    return pd.DataFrame(preprocessing.normalize(dataset), columns=dataset.columns.values)


def get_dataset_scaled(dataset: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe after scaling it's values"""
    print('Scaling dataset...')
    return pd.DataFrame(preprocessing.scale(dataset), columns=dataset.columns.values)


def get_dataset_min_max_scaled(dataset: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe after scaling it's values"""
    print('Min-max scaling dataset...')
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(dataset)
    return pd.DataFrame(scaled_df, columns=dataset.columns.values)


def store_correlation_heatmap(dataset: pd.DataFrame, figure_name: str, fig_size: Optional[dict] = None):
    print('Storing correlation heatmap...')
    timestamp = str(datetime.datetime.utcnow().replace(microsecond=0)).replace(' ', '').replace('-', '')

    sns.set(rc={'figure.figsize': (20.7, 17.27)} or fig_size)
    sns.heatmap(data=dataset.corr(),
                cmap=sns.diverging_palette(220, 10, as_cmap=True)).get_figure().savefig('{}{}.png'.format(figure_name,
                                                                                                          timestamp))
    print('Correlation heatmap stored as {}.png'.format(figure_name))


def get_not_correlated_features(dataset: pd.DataFrame, threshold: float):
    """ Drops features with correlation greater than the given threshold """
    print('Droping features with correlation greater than {}...'.format(threshold))
    # Create correlation matrix
    corr_matrix = dataset.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print('Dropped features: {}'.format(to_drop))
    dataset.drop(to_drop, axis=1, inplace=True)
    print('Left features after dropping highly correlated: {}'.format(dataset.columns.values))

    return dataset.columns.values


def get_k_best_features(X: pd.DataFrame, Y: pd.DataFrame, K: int):
    """
    :param X: data for training
    :param Y: corresponding labels
    :param K: number of features to return
    :return: k best features based on the chi2 function
    """
    print('Extracting {} best features with chi2...'.format(K))
    # feature extraction
    selector = SelectKBest(score_func=chi2, k=K).fit(X, Y)

    mask = selector.get_support()  # list of booleans whether feature i was included or not
    new_features = []  # The list of your K best features

    for bool, feature in zip(mask, X.columns.values):
        if bool:
            new_features.append(feature)

    print('Chose these top {} best features: {}'.format(K, new_features))
    return new_features


def get_best_features_with_decision_tress(X: pd.DataFrame, Y: pd.DataFrame, dataset_name: str):
    print('Getting features from decision tree classifier that have importance > 0...')

    timestamp = str(datetime.datetime.utcnow().replace(microsecond=0)).replace(' ', '').replace('-', '')

    tree_classifier = DecisionTreeClassifier(min_samples_leaf=1, max_depth=len(X.columns))
    tree_classifier.fit(X, Y)

    zero_importance_features = pd.np.where(tree_classifier.feature_importances_ == 0)
    new_features = []  # The list of your K best features

    for idx, feature in enumerate(X.columns.values):
        if idx not in zero_importance_features[0]:
            new_features.append(feature)

    importances = tree_classifier.feature_importances_
    std = np.std([tree_classifier.feature_importances_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")

    # TODO: correctly write column names vertically
    plt.xticks(range(X.shape[1]), [X.columns[x] for x in indices])

    plt.xlim([-1, X.shape[1]])
    plt.savefig('{}_feature_importances_dc_{}.png'.format(dataset_name, timestamp))
    # plt.show()

    print('Decision trees chose these features with non-zero importance: {}'.format(new_features))
    return new_features


def get_perf(y, y_pred, multiclass: bool = False):
    """
    This method outputs several performance metrics for classification.
    """

    # Gets Confusion Matrix
    # conf_matrix = metrics.confusion_matrix(y_true = y, y_pred = y_pred)

    # if multclass true, use the total values for metrics
    average = 'binary' if not multiclass else 'micro'

    # Gets Accuracy
    accuracy = metrics.accuracy_score(y_true=y, y_pred=y_pred)

    # Gets Recall
    recall = metrics.recall_score(y_true=y, y_pred=y_pred, average=average)

    # Gets Precision
    precision = metrics.precision_score(y_true=y, y_pred=y_pred, average=average)

    # F1
    f1 = metrics.fbeta_score(y_true=y, y_pred=y_pred, beta=1, average=average)

    if not multiclass:
        tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_pred).ravel()

        TPR = tp / (tp + fn)
        TNR = tn / (tn + fp)
        roc_auc = ((TPR + TNR) / 2)
    else:
        print('Total number of predictions: {}'.format(len(y)))
        print('accuracy: {}, recall: {}, precision: {}, F1Beta: {}'.format(accuracy, recall, precision, f1))
        return

    print('Total number of predictions: {}'.format(len(y)))
    print('accuracy: {}, recall: {}, precision: {}, F1Beta: {}, roc_auc: {} '
          ' TP: {}, TN: {}, FP: {}, FN:{} '.format(accuracy, recall, precision, f1, roc_auc, tp, tn, fp, fn))

    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'F1Beta': f1}


def classify_all(data: pd.DataFrame, target_feature: str, cv: int, filename: str):
    """
    Classifies the data with the four classifers with default parameters.
    :param cv:
    :param target_feature: target feature name
    :param data: input dataset
    :return:
    """

    # put target column as the last column
    class_column = data[target_feature]
    data.drop(columns=target_feature, inplace=True)
    data.insert(len(data.columns), target_feature, class_column)

    # do not include the target column (which is the last one) on data dataframe
    X = data.iloc[:, :-1]
    y = data.iloc[:, data.columns.get_loc(target_feature)]

    # Splitting the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # KNN
    clf_knn: KNeighborsClassifier = KNeighborsClassifier()
    scores_knn = cross_validate(clf_knn, X, y, cv=cv, scoring=['f1', 'accuracy', 'recall', 'precision', 'roc_auc'],
                                n_jobs=-1, return_train_score=False)
    print("F1 Scores for KNN")
    print(scores_knn)
    # Get mean of 5 test times for measurements
    scores_mean_knn = {'fit_time': [scores_knn['fit_time'].mean()],
                       'score_time': [scores_knn['score_time'].mean()],
                       'test_f1': [scores_knn['test_f1'].mean()],
                       'test_accuracy': [scores_knn['test_accuracy'].mean()],
                       'test_recall': [scores_knn['test_recall'].mean()],
                       'test_precision': [scores_knn['test_precision'].mean()],
                       'test_roc_auc': [scores_knn['test_roc_auc'].mean()]}
    scores_mean_knn.update({'classifier': 'KNN'})
    all_scores = pd.DataFrame(data=scores_mean_knn, index=[filename])

    # Naive-Bayes
    # Binary Features <- Bernoulli Naive Bayes
    # Continuous Features <- Gaussian Naive Bayes
    clf_gnb = GaussianNB()
    scores_nb = cross_validate(clf_gnb, X, y, cv=cv, scoring=['f1', 'accuracy', 'recall', 'precision', 'roc_auc'],
                               n_jobs=-1, return_train_score=False)
    print("F1 Scores for GNB")
    print(scores_nb)
    # Get mean of 5 test times for measurements
    scores_mean_nb = {'fit_time': [scores_nb['fit_time'].mean()],
                      'score_time': [scores_nb['score_time'].mean()],
                      'test_f1': [scores_nb['test_f1'].mean()],
                      'test_accuracy': [scores_nb['test_accuracy'].mean()],
                      'test_recall': [scores_nb['test_recall'].mean()],
                      'test_precision': [scores_nb['test_precision'].mean()],
                      'test_roc_auc': [scores_nb['test_roc_auc'].mean()]}
    scores_mean_nb.update({'classifier': 'Naive Bayes'})
    all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_nb, index=[filename])))

    # Decision Trees
    clf_dt: DecisionTreeClassifier = DecisionTreeClassifier(max_depth=20)
    scores_dt = cross_validate(clf_dt, X, y, cv=cv, scoring=['f1', 'accuracy', 'recall', 'precision', 'roc_auc'],
                               n_jobs=-1, return_train_score=False)
    print("F1 Scores for DT")
    print(scores_dt)
    # Get mean of 5 test times for measurements
    scores_mean_dt = {'fit_time': [scores_dt['fit_time'].mean()],
                      'score_time': [scores_dt['score_time'].mean()],
                      'test_f1': [scores_dt['test_f1'].mean()],
                      'test_accuracy': [scores_dt['test_accuracy'].mean()],
                      'test_recall': [scores_dt['test_recall'].mean()],
                      'test_precision': [scores_dt['test_precision'].mean()],
                      'test_roc_auc': [scores_dt['test_roc_auc'].mean()]}
    scores_mean_dt.update({'classifier': 'Decision Tree'})
    all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_dt, index=[filename])))

    # Random Forest
    clf_rf: RandomForestClassifier = RandomForestClassifier(n_estimators=500, max_depth=10, verbose=1, n_jobs=-1)
    scores_rf = cross_validate(clf_rf, X, y, cv=cv, scoring=['f1', 'accuracy', 'recall', 'precision', 'roc_auc'],
                               n_jobs=-1, return_train_score=False)
    print("F1 Scores for RF")
    print(scores_rf)
    # Get mean of 5 test times for measurements
    scores_mean_rf = {'fit_time': [scores_rf['fit_time'].mean()],
                      'score_time': [scores_rf['score_time'].mean()],
                      'test_f1': [scores_rf['test_f1'].mean()],
                      'test_accuracy': [scores_rf['test_accuracy'].mean()],
                      'test_recall': [scores_rf['test_recall'].mean()],
                      'test_precision': [scores_rf['test_precision'].mean()],
                      'test_roc_auc': [scores_rf['test_roc_auc'].mean()]}
    scores_mean_rf.update({'classifier': 'Random Forest'})
    all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_rf, index=[filename])))

    # Save to CSV
    all_scores.to_csv(path_or_buf='{}{}'.format(filename, '.csv'))


def classify_all_multiple(data: pd.DataFrame, target_feature: str, cv: int, filename: str):
    """
    Classifies the data with the four classifers with default parameters.
    :param cv:
    :param target_feature: target feature name
    :param data: input dataset
    :return:
    """

    # put target column as the last column
    class_column = data[target_feature]
    data.drop(columns=target_feature, inplace=True)
    data.insert(len(data.columns), target_feature, class_column)

    # do not include the target column (which is the last one) on data dataframe
    X = data.iloc[:, :-1]
    y = data.iloc[:, data.columns.get_loc(target_feature)]

    # Splitting the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # KNN
    clf_knn: KNeighborsClassifier = KNeighborsClassifier()
    scores_knn = cross_validate(clf_knn, X, y, cv=cv, n_jobs=-1, return_train_score=False, scoring=['accuracy'])
    print("F1 Scores for KNN")
    print(scores_knn)
    # Get mean of 5 test times for measurements
    scores_mean_knn = {'fit_time': [scores_knn['fit_time'].mean()],
                       'score_time': [scores_knn['score_time'].mean()],
                       # 'test_score': [scores_knn['test_score'].mean()]}
                       'test_accuracy': [scores_knn['test_accuracy'].mean()]}
    #                    'test_recall': [scores_knn['test_recall'].mean()],
    #                    'test_precision': [scores_knn['test_precision'].mean()],
    #                    'test_roc_auc': [scores_knn['test_roc_auc'].mean()]}
    scores_mean_knn.update({'classifier': 'KNN'})
    all_scores = pd.DataFrame(data=scores_mean_knn, index=[filename])

    # Naive-Bayes
    # Binary Features <- Bernoulli Naive Bayes
    # Continuous Features <- Gaussian Naive Bayes
    clf_gnb = GaussianNB()
    scores_nb = cross_validate(clf_gnb, X, y, cv=cv, n_jobs=-1, return_train_score=False, scoring=['accuracy'])
    print("F1 Scores for GNB")
    print(scores_nb)
    # Get mean of 5 test times for measurements
    scores_mean_nb = {'fit_time': [scores_nb['fit_time'].mean()],
                      'score_time': [scores_nb['score_time'].mean()],
                      # 'test_score': [scores_nb['test_score'].mean()]}
                      'test_accuracy': [scores_nb['test_accuracy'].mean()]}
    #                   'test_recall': [scores_nb['test_recall'].mean()],
    #                   'test_precision': [scores_nb['test_precision'].mean()],
    #                   'test_roc_auc': [scores_nb['test_roc_auc'].mean()]}
    scores_mean_nb.update({'classifier': 'Naive Bayes'})
    all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_nb, index=[filename])))

    # Decision Trees
    clf_dt: DecisionTreeClassifier = DecisionTreeClassifier(max_depth=20)
    scores_dt = cross_validate(clf_dt, X, y, cv=cv, n_jobs=-1, return_train_score=False, scoring=['accuracy'])
    print("F1 Scores for DT")
    print(scores_dt)
    # Get mean of 5 test times for measurements
    scores_mean_dt = {'fit_time': [scores_dt['fit_time'].mean()],
                      'score_time': [scores_dt['score_time'].mean()],
                      # 'test_score': [scores_dt['test_score'].mean()]}
                      'test_accuracy': [scores_dt['test_accuracy'].mean()]}
    #                   'test_recall': [scores_dt['test_recall'].mean()],
    #                   'test_precision': [scores_dt['test_precision'].mean()],
    #                   'test_roc_auc': [scores_dt['test_roc_auc'].mean()]}
    scores_mean_dt.update({'classifier': 'Decision Tree'})
    all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_dt, index=[filename])))

    # Random Forest
    clf_rf: RandomForestClassifier = RandomForestClassifier(n_estimators=500, max_depth=10, verbose=1, n_jobs=-1)
    scores_rf = cross_validate(clf_rf, X, y, cv=cv, n_jobs=-1, return_train_score=False, scoring=['accuracy'])
    print("F1 Scores for RF")
    print(scores_rf)
    # Get mean of 5 test times for measurements
    scores_mean_rf = {'fit_time': [scores_rf['fit_time'].mean()],
                      'score_time': [scores_rf['score_time'].mean()],
                      # 'test_score': [scores_rf['test_score'].mean()]}
                      'test_accuracy': [scores_rf['test_accuracy'].mean()]}
    #                   'test_recall': [scores_rf['test_recall'].mean()],
    #                   'test_precision': [scores_rf['test_precision'].mean()],
    #                   'test_roc_auc': [scores_rf['test_roc_auc'].mean()]}
    scores_mean_rf.update({'classifier': 'Random Forest'})
    all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_rf, index=[filename])))

    # Save to CSV
    all_scores.to_csv(path_or_buf='{}{}'.format(filename, '.csv'))


def classify_all_raw(data: pd.DataFrame, target_feature: str, cv: int, filename: str):
    """
    Classifies the data with the four classifers with default parameters.
    :param cv:
    :param target_feature: target feature name
    :param data: input dataset
    :return:
    """

    # put target column as the last column
    class_column = data[target_feature]
    data.drop(columns=target_feature, inplace=True)
    data.insert(len(data.columns), target_feature, class_column)

    # do not include the target column (which is the last one) on data dataframe
    X = data.iloc[:, :-1]
    y = data.iloc[:, data.columns.get_loc(target_feature)]

    # Splitting the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # KNN
    # clf_knn: KNeighborsClassifier = KNeighborsClassifier()
    # scores_knn = cross_validate(clf_knn, X, y, cv=cv, scoring=['f1', 'accuracy', 'recall', 'precision', 'roc_auc'])
    # print("F1 Scores for KNN")
    # print(scores_knn)
    # # Get mean of 5 test times for measurements
    # scores_mean_knn = {'fit_time': [scores_knn['fit_time'].mean()],
    #                    'score_time': [scores_knn['score_time'].mean()],
    #                    'test_f1': [scores_knn['test_f1'].mean()],
    #                    'test_accuracy': [scores_knn['test_accuracy'].mean()],
    #                    'test_recall': [scores_knn['test_recall'].mean()],
    #                    'test_precision': [scores_knn['test_precision'].mean()],
    #                    'test_roc_auc': [scores_knn['test_roc_auc'].mean()]}
    # scores_mean_knn.update({'classifier': 'KNN'})
    # all_scores = pd.DataFrame(data=scores_mean_knn, index=[filename])

    # Naive-Bayes
    # Binary Features <- Bernoulli Naive Bayes
    # Continuous Features <- Gaussian Naive Bayes
    # clf_gnb = GaussianNB()
    # scores_nb = cross_validate(clf_gnb, X, y, cv=cv, scoring=['f1', 'accuracy', 'recall', 'precision', 'roc_auc'])
    # print("F1 Scores for GNB")
    # print(scores_nb)
    # # Get mean of 5 test times for measurements
    # scores_mean_nb = {'fit_time': [scores_nb['fit_time'].mean()],
    #                   'score_time': [scores_nb['score_time'].mean()],
    #                   'test_f1': [scores_nb['test_f1'].mean()],
    #                   'test_accuracy': [scores_nb['test_accuracy'].mean()],
    #                   'test_recall': [scores_nb['test_recall'].mean()],
    #                   'test_precision': [scores_nb['test_precision'].mean()],
    #                   'test_roc_auc': [scores_nb['test_roc_auc'].mean()]}
    # scores_mean_nb.update({'classifier': 'Naive Bayes'})
    # all_scores = pd.DataFrame(data=scores_mean_nb, index=[filename])
    # all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_nb, index=[filename])))

    # Decision Trees
    clf_dt: DecisionTreeClassifier = DecisionTreeClassifier(max_depth=20)
    scores_dt = cross_validate(clf_dt, X, y, cv=cv, scoring=['f1', 'accuracy', 'recall', 'precision', 'roc_auc'])
    print("F1 Scores for DT")
    print(scores_dt)
    # Get mean of 5 test times for measurements
    scores_mean_dt = {'fit_time': [scores_dt['fit_time'].mean()],
                      'score_time': [scores_dt['score_time'].mean()],
                      'test_f1': [scores_dt['test_f1'].mean()],
                      'test_accuracy': [scores_dt['test_accuracy'].mean()],
                      'test_recall': [scores_dt['test_recall'].mean()],
                      'test_precision': [scores_dt['test_precision'].mean()],
                      'test_roc_auc': [scores_dt['test_roc_auc'].mean()]}
    scores_mean_dt.update({'classifier': 'Decision Tree'})
    all_scores = pd.DataFrame(data=scores_mean_dt, index=[filename])
    # all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_dt, index=[filename])))

    # Random Forest
    clf_rf: RandomForestClassifier = RandomForestClassifier(n_estimators=500, max_depth=10, verbose=1, n_jobs=-1)
    scores_rf = cross_validate(clf_rf, X, y, cv=cv, scoring=['f1', 'accuracy', 'recall', 'precision', 'roc_auc'])
    print("F1 Scores for RF")
    print(scores_rf)
    # Get mean of 5 test times for measurements
    scores_mean_rf = {'fit_time': [scores_rf['fit_time'].mean()],
                      'score_time': [scores_rf['score_time'].mean()],
                      'test_f1': [scores_rf['test_f1'].mean()],
                      'test_accuracy': [scores_rf['test_accuracy'].mean()],
                      'test_recall': [scores_rf['test_recall'].mean()],
                      'test_precision': [scores_rf['test_precision'].mean()],
                      'test_roc_auc': [scores_rf['test_roc_auc'].mean()]}
    scores_mean_rf.update({'classifier': 'Random Forest'})
    all_scores = pd.concat((all_scores, pd.DataFrame(data=scores_mean_rf, index=[filename])))

    # Save to CSV
    all_scores.to_csv(path_or_buf='{}{}'.format(filename, '.csv'))


def predict_kdd_kaggle(clf, test_dat):
    """
    Predicts the kaggle test and compares
    :param test_dat:
    :param clf:
    :return:
    """

    # Loading the kaggle results data
    data_kaggle: pd.DataFrame = pd.read_csv(Path('../datasets/Kaggle-KDD/kaggle-test.csv'))

    # Predicting the Kaggle Values
    y_kaggle_pred = clf.predict(test_dat)

    # Get performance
    print('\nKaggle Performance:')
    perf_kaggle = get_perf(data_kaggle['TARGET_B'], y_kaggle_pred)

    print('Storing the predictions csv file...')
    # store prediction
    prediction_dataframe = pd.DataFrame({'CONTROLN': data_kaggle['CONTROLN'], 'TARGET_B': y_kaggle_pred})
    prediction_dataframe.to_csv(kdd_data_folder / 'prediction-kaggle.csv', index=False)


def load_raw_data(dataset_name: str):
    """
    Loads raw datasets for train and test
    :return: raw train and test dataset
    """
    try:
        raw_dat: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['initial_path_train'],
                                            sep=DATASETS[dataset_name]['separator'], low_memory=False)
        return raw_dat
    except KeyError:
        print('Dataset with key {} doesnt exist.'.format(dataset_name))
        return None


def analyze_raw_dataset(dataset_name: str):
    """
    Analyzes the raw data sets.
    Possible values: { 'kdd', 'breast_cancer', 'image_segmentation', 'bank_marketing'}
    :return: Output in the console with some information about the dataset.
    """

    # Loads raw data
    raw_dat_train = load_raw_data(dataset_name=dataset_name)
    if raw_dat_train is None:
        print('Something went wrong.')
        return

    folder = Path('../examples/' + dataset_name + '/')

    # Exploratory Analysis
    print("\nShape:")
    print(raw_dat_train.shape)
    print("\nCount:")
    print(raw_dat_train.count())  # checks how many missing values are in the dataset
    print("\nHead:")
    print(raw_dat_train.head())
    print("\nColumns:")
    print(raw_dat_train.columns)

    if dataset_name == 'kdd':
        target_feature_str = 'TARGET_B'
        # Find missing values per column and plot
        series = pd.DataFrame(raw_dat_train.isna().sum())
        series = series[~(series == 0).any(axis=1)]
        series.plot.barh(figsize=(24, 18), color='steelblue')
        plt.savefig(folder / '{}{}'.format(dataset_name, '_missing_values.png'), dpi=300)
        plt.clf()
    elif dataset_name == 'bank_marketing':
        target_feature_str = 'y'
    elif dataset_name == 'image_segmentation':
        target_feature_str = 'CLASS'
    elif dataset_name == 'breast_cancer':
        target_feature_str = 'class'
    else:
        print('Error: Couldnt find specified dataset.')
        return

    # Pie chart % of target
    (raw_dat_train[target_feature_str].value_counts(normalize=True) * 100) \
        .plot.pie(figsize=(6, 6), colormap='tab20c', autopct='%.2f%%')
    plt.savefig(folder / '{}{}'.format(dataset_name, '_target_feature_percantage.png'), dpi=300)
    plt.clf()

    # Dataset Histograms
    raw_dat_train.hist(figsize=(24, 18), color='steelblue')
    plt.savefig(folder / '{}{}'.format(dataset_name, '_histograms.png'), dpi=300)
    plt.clf()

    # Plot correlation matrix of all features
    corr = raw_dat_train.corr()

    # Seaborn method
    sns.set(style="white")

    # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(24, 18))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap)

    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(folder / '{}{}'.format(dataset_name, '_sns_corr_matrix.png'), dpi=300)
    plt.clf()

    # Some statistics about raw_dat's variables
    print("\nDescribe:")
    print(raw_dat_train.describe())

    # % of target
    print('\nPercentage of target feature\s:')
    print(raw_dat_train[target_feature_str].value_counts(normalize=True) * 100)


# Klas e marrun hazer prej https://github.com/rebordao/kdd98cup
class Preprocessor:

    @staticmethod
    def fill_nans(dat):
        """
        Fills in NaNs with either the mean or the most common value.
        """

        return DataFrameImputer().fit_transform(dat)


# Klas e marrun hazer prej https://github.com/rebordao/kdd98cup
class DataFrameImputer(TransformerMixin):
    """
    This class came from http://stackoverflow.com/questions/25239958/
    impute-categorical-missing-values-in-scikit-learn
    """

    def __init__(self):
        """
        Impute missing values.
        Columns of dtype object are imputed with the most frequent value in col.
        Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
