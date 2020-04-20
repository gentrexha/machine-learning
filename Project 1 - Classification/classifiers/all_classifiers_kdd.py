import sys

from sklearn.feature_selection import VarianceThreshold

from utils import *
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from sklearn.utils import shuffle, resample
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

"""
Kjo klase osht diqka krejt fantazi, e paparapare. Qetu ndodh gjithqka. Qka t'kish qef qetu e gjen.
"""


def raw_classification():
    """
    Raw classification without any preprocessing at all.
    :return:
    """

    # Loads raw data
    raw_dat_train = load_raw_data('kdd')

    # Loads test data
    # test_dat: pd.DataFrame = pd.read_csv(DATASETS['kdd']['initial_path_test'])

    # # do not include ID on the model
    # columns_without_ID = list(raw_dat_train.columns)
    # columns_without_ID.remove('CONTROLN')
    #
    # # Classification on raw data
    # classify_all_raw(pd.get_dummies(raw_dat_train[columns_without_ID]), 'TARGET_B', 5, 'raw_data_with_dummies')

    # Classification on numerical values only and features w/o missing values
    # https://www.quora.com/How-can-I-use-KNN-for-mixed-data-categorical-and-numerical
    train_dat = raw_dat_train.select_dtypes(include=[np.number])

    # Drop columns with NaN values
    train_dat = train_dat.dropna(axis=1)

    # do not include ID on the model
    columns_without_ID = list(train_dat.columns)
    columns_without_ID.remove('CONTROLN')

    classify_all(train_dat[columns_without_ID], 'TARGET_B', 5, 'numerical_and_drop_missing_data')


def preprocess():
    """
    Preprocess the data
    :return:
    """

    # Loads raw data
    raw_dat_train = load_raw_data('kdd')

    # Loads test data
    dat_test: pd.DataFrame = pd.read_csv(DATASETS['kdd']['initial_path_test'])

    # Gets some redundant variables based on variance, sparsity & common sense
    # Some vars that don't seem of good value
    # More info:
    # https://stats.stackexchange.com/questions/309612/removing-features-with-low-variance-in-classification-models/309615
    # ['RDATE_5', 'RAMNT_5'] Skan vlera hiq motrat n'pidh
    redundant_vars = ['CONTROLN', 'ZIP', 'RDATE_5', 'RAMNT_5']

    # Drop features with a variance smaller than 0.001
    # I drop veq 3-4 ashtu kshtu
    raw_dat_train_var = raw_dat_train.var()
    redundant_vars.extend(raw_dat_train_var.index[raw_dat_train_var < 0.001])
    # Drops redundant cols
    dat_train = raw_dat_train.drop(redundant_vars, axis=1)
    dat_test = dat_test.drop(redundant_vars, axis=1)

    # # Up-sample
    # df_majority = dat_train[dat_train.TARGET_B == 0]
    # df_minority = dat_train[dat_train.TARGET_B == 1]
    #
    # df_minority_upsampled = resample(df_minority, replace=True,  # sample with replacement
    #                                  n_samples=2371,  # to match majority class
    #                                  random_state=0)  # reproducible results
    #
    # df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    #
    # dat_train = df_upsampled

    # Down-sample
    df_majority = dat_train[dat_train.TARGET_B == 0]
    df_minority = dat_train[dat_train.TARGET_B == 1]
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=129,  # to match minority class
                                       random_state=0)  # reproducible results
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    dat_train = df_downsampled

    # Imputes the data and fills in the missing values
    dat_train = Preprocessor.fill_nans(dat_train)
    dat_test = Preprocessor.fill_nans(dat_test)

    # Drop features with no values at all
    dat_train = dat_train.drop(axis=1, labels=dat_train.columns[dat_train.isnull().any()])

    # Changes categorical vars to a numerical form
    dat_train = pd.get_dummies(dat_train)
    dat_test = pd.get_dummies(dat_test)

    # classify_all(dat_train, 'TARGET_B', 5, 'filled_missing_values_and_dummies')

    # Scale data
    # https://stats.stackexchange.com/questions/105478/feature-normalization-standardization-before-or-after-feature-selection
    y = dat_train.TARGET_B
    dat_train = dat_train.drop("TARGET_B", axis=1)
    dat_train = get_dataset_normalized(dat_train)
    dat_train = dat_train.assign(TARGET_B=y.values)
    dat_test = get_dataset_normalized(dat_test)

    # TODO: Do we need this? Research.
    # Shuffles observations
    # dat_train.apply(np.random.permutation)

    # Feature Selection #

    # Correlation-based Feature Selection #
    # Computes correlation between raw_dat_train["TARGET_B"] and the predictors
    TARGET_B_corr = dat_train.corr()["TARGET_B"].copy()
    TARGET_B_corr = TARGET_B_corr.sort_values(ascending=False)

    # Sorts and picks the first x features
    # TODO: get optimal x value automatically
    tmp = abs(TARGET_B_corr).copy()
    tmp = tmp.sort_values(ascending=False)
    important_vars = [tmp.index[0]]
    important_vars.extend(list(tmp.index[2:52]))  # removes other target

    classify_all(dat_train[important_vars], 'TARGET_B', 5, 'down_sampling')

    # clf_knn, clf_gnb, clf_dt, clf_rf = classify_all(dat_train[important_vars], 'TARGET_B', "Correlation-based")
    # predict_kdd_kaggle(clf_knn, dat_test[tmp.index[2:52]])
    # predict_kdd_kaggle(clf_gnb, dat_test[tmp.index[2:52]])
    # predict_kdd_kaggle(clf_dt, dat_test[tmp.index[2:52]])
    # predict_kdd_kaggle(clf_rf, dat_test[tmp.index[2:52]])
    # dat_train[important_vars]
    # dat_test[tmp.index[2:52]])

    # Hyper-parameter tuning
    # random_forests_on_breast_cancer_grid_search(dat_train[important_vars], dat_test[tmp.index[2:52]])

    # Random Forest with said parameters
    # put target column as the last column
    # class_column = dat_train['TARGET_B']
    # dat_train.drop(columns='TARGET_B', inplace=True)
    # dat_train.insert(len(dat_train.columns), 'TARGET_B', class_column)

    # Split train data from target feature
    # x_train = dat_train.iloc[:, :-1]
    # y_train = dat_train.iloc[:, dat_train.columns.get_loc('TARGET_B')]
    #
    # clf_rf = RandomForestClassifier(n_estimators=914, max_depth=38, criterion='entropy')
    # clf_rf = clf_rf.fit(x_train, y_train)
    #
    # predict_kdd_kaggle(clf_rf, dat_test[tmp.index[2:52]])

    # Variance-based Feature Selection #
    # https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
    # TODO: C'kommento babo
    # y = dat_train.TARGET_B
    # X = dat_train.drop("TARGET_B", axis=1)
    # sel = VarianceThreshold(threshold=0.005)
    # X_new = sel.fit_transform(X)
    # print(X_new)
    # print(dat_test)
    # dat_test_new = sel.fit_transform(dat_test)
    # print(dat_test_new)
    # df = pd.DataFrame(data=X_new)
    # df = df.assign(TARGET_B=y.values)
    # clf_knn, clf_gnb, clf_dt, clf_rf = classify_all(df, 'TARGET_B', 'Variance-based')
    # predict_kdd_kaggle(clf_knn, dat_test_new)
    # predict_kdd_kaggle(clf_gnb, dat_test_new)
    # predict_kdd_kaggle(clf_dt, dat_test_new)
    # predict_kdd_kaggle(clf_rf, dat_test_new)

    # Univariate Feature Selection #
    # TODO: C'kommento babo
    # y = dat_train.TARGET_B
    # X = dat_train.drop("TARGET_B", axis=1)
    # X_new = SelectKBest(chi2, k=10).fit_transform(X.values, y.values)
    # df = pd.DataFrame(data=X_new)
    # df = df.assign(TARGET_B=y.values)
    # classify_all(df, "Univariate")

    # Tree-based Feature Selection #
    # TODO: C'kommento babo
    # y = dat_train.TARGET_B
    # X = dat_train.drop("TARGET_B", axis=1)
    #
    # clf = ExtraTreesClassifier(n_estimators=50)
    # clf = clf.fit(X, y)
    #
    # model = SelectFromModel(clf, prefit=True)
    # X_new = model.transform(X)
    # df = pd.DataFrame(data=X_new)
    # df = df.assign(TARGET_B=y.values)
    # classify_all(df, "Tree-based")

    # L1-based feature selection #
    # TODO: C'kommento babo
    # y = dat_train.TARGET_B
    # X = dat_train.drop("TARGET_B", axis=1)
    #
    # lsvc = LinearSVC(C=0.99, penalty="l1", dual=False).fit(X, y)
    # model = SelectFromModel(lsvc, prefit=True)
    # X_new = model.transform(X)
    # print(X_new.shape)
    # df = pd.DataFrame(data=X_new)
    # df = df.assign(TARGET_B=y.values)
    # classify_all(df, "L1-based")

    # https://stackoverflow.com/questions/46062679/right-order-of-doing-feature-selection-pca-and-normalization
    # TODO: C'kommento babo
    # y = dat_train.TARGET_B
    # X = dat_train.drop("TARGET_B", axis=1)
    # pca = PCA(svd_solver='auto', n_components='mle')
    # pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # print(pca.explained_variance_ratio_.shape)
    # print(pca.singular_values_)
    # print(pca.singular_values_.shape)

    # TODO: Sampling


def random_forests_on_breast_cancer_grid_search(dat_train, dat_test):
    """

    :param train:
    :return:
    """

    # Test data for CONTROL_N
    dat_kaggle: pd.DataFrame = pd.read_csv(DATASETS['kdd']['initial_path_test'])

    # put target column as the last column
    class_column = dat_train['TARGET_B']
    dat_train.drop(columns='TARGET_B', inplace=True)
    dat_train.insert(len(dat_train.columns), 'TARGET_B', class_column)

    # Split train data from target feature
    x_train = dat_train.iloc[:, :-1]
    y_train = dat_train.iloc[:, dat_train.columns.get_loc('TARGET_B')]

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
    # columns_without_ID = list(x_train.columns)
    # columns_without_ID.remove('CONTROLN')

    # k_best_columns = get_k_best_features(x_train[columns_without_ID], y_train, K=10)
    # k_best_columns = get_k_best_features(x_train[columns_without_ID], y_train, K=10)
    # x_train = drop_highly_correlated_features(x_train, 0.8)
    # k_best_columns = columns_without_ID

    # Fit the random search model
    print('Fitting/training the random forests model...')
    rf_grid_search.fit(x_train, y_train)

    try:
        prediction_dataframe = pd.DataFrame(rf_grid_search.cv_results_)
        prediction_dataframe.to_csv(kdd_data_folder / 'grid_search_random_forest.csv', index=False)
    except Exception as e:
        print('An erro occurred while storing grind search results')
        pass

    print('Predicting the training data...')
    predictions = rf_grid_search.best_estimator_.predict(dat_test)

    print('Best parameters found by grid_search: {}'.format(rf_grid_search.best_estimator_.get_params()))

    print('Storing random forests results to csv predictions for {}...'.format(dat_test))
    # store prediction
    prediction_dataframe = pd.DataFrame({'CONTROL_N': dat_kaggle['CONTROLN'],
                                         'TARGET_B': [1 if val == 1 else 0 for val in predictions]})

    prediction_dataframe.to_csv(breast_cancer_data_folder / 'prediction-random-forests.csv', index=False)


def main():
    """
    Main function.
    :return:
    """
    # raw_classification()
    preprocess()
    # analyze_raw_dataset('kdd')
    # analyze_raw_dataset('bank_marketing')
    # analyze_raw_dataset('image_segmentation')
    # analyze_raw_dataset('breast_cancer')


if __name__ == '__main__':
    main()
    sys.exit(0)
