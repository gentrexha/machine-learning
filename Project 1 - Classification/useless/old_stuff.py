

def random_forests_on_kdd(n_estimators: int):
    dataset_name = 'kdd'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('TARGET_B')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    classifier: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=4)

    # do not include ID on the model (it makes no sense)
    target_training_columns = ['RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'RecencyOfGiving', 'AmountOfGiving',
                               'FrequencyOfGiving', 'UrbanicityLevel', 'SocioEconomicStatus', 'CLUSTER', 'HOMEOWNR',
                               'NUMCHLD', 'INCOME', 'GENDER', 'WEALTH1']

    print('Fitting/training the RandomForest model...')
    classifier.fit(x_train[target_training_columns], y_train)

    print('Predicting the training data...')
    predictions = classifier.predict(dataset_test[target_training_columns])

    print('Storing random forests csv predictions for {}...'.format(dataset_name))
    # store prediction
    prediction_dataframe = pd.DataFrame({'CONTROLN': dataset_test['CONTROLN'], 'TARGET_B': predictions})
    prediction_dataframe.to_csv(kdd_data_folder / 'prediction-random-forests.csv', index=False)


def random_forests_on_kdd_randomized_search(n_estimators: int):
    dataset_name = 'kdd'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('TARGET_B')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    target_training_columns = ['RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'RecencyOfGiving', 'AmountOfGiving',
                               'FrequencyOfGiving', 'UrbanicityLevel', 'SocioEconomicStatus', 'CLUSTER', 'HOMEOWNR',
                               'NUMCHLD', 'INCOME', 'GENDER', 'WEALTH1']

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
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    # Fit the random search model
    print('Fitting/training the random forests model...')
    rf_random.fit(x_train[target_training_columns], y_train)

    print('Predicting the training data...')
    predictions = rf_random.best_estimator_.predict(dataset_test[target_training_columns])

    print('Storing random forests csv predictions for {}...'.format(dataset_name))
    # store prediction
    prediction_dataframe = pd.DataFrame({'CONTROLN': dataset_test['CONTROLN'], 'TARGET_B': predictions})
    prediction_dataframe.to_csv(kdd_data_folder / 'prediction-random-forests.csv', index=False)


def random_forests_on_kdd_grid_search(n_estimators: int):
    dataset_name = 'kdd'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('TARGET_B')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    target_training_columns = ['RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'RecencyOfGiving', 'AmountOfGiving',
                               'FrequencyOfGiving', 'UrbanicityLevel', 'SocioEconomicStatus', 'CLUSTER', 'HOMEOWNR',
                               'NUMCHLD', 'INCOME', 'GENDER', 'WEALTH1']

    # Number of trees in random forest
    n_estimators = [int(x) for x in pd.np.linspace(start=50, stop=200, num=25)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in pd.np.linspace(2, 12, num=2)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    grid_search = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap
                   }
    pprint(grid_search)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_grid_search = GridSearchCV(estimator=rf, param_grid=grid_search, cv=3, verbose=2, n_jobs=-1)

    # Fit the random search model
    print('Fitting/training the random forests model...')
    rf_grid_search.fit(x_train[target_training_columns], y_train)

    print('Predicting the training data...')
    predictions = rf_grid_search.best_estimator_.predict(dataset_test[target_training_columns])

    print('Storing random forests csv predictions for {}...'.format(dataset_name))
    # store prediction
    prediction_dataframe = pd.DataFrame({'CONTROLN': dataset_test['CONTROLN'], 'TARGET_B': predictions})
    prediction_dataframe.to_csv(kdd_data_folder / 'prediction-random-forests.csv', index=False)


def perform_knn_on_kdd(n_neighbors: int, weight: str):
    dataset_name = 'kdd'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('TARGET_B')]

    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_test'])

    classifier: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)

    target_training_columns = ['RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'RecencyOfGiving', 'AmountOfGiving',
                               'FrequencyOfGiving', 'UrbanicityLevel', 'SocioEconomicStatus', 'CLUSTER2', 'HOMEOWNR',
                               'NUMCHLD', 'INCOME', 'GENDER', 'WEALTH1']

    x_train[target_training_columns] = get_dataset_normalized(x_train[target_training_columns])
    dataset_test[target_training_columns] = get_dataset_normalized(dataset_test[target_training_columns])

    print('Fitting/training the RandomForest model...')
    classifier.fit(x_train[target_training_columns], y_train)

    print('Predicting the training data...')
    predictions = classifier.predict(dataset_test[target_training_columns])

    print('Storing random forests csv predictions for {}...'.format(dataset_name))
    # store prediction
    prediction_dataframe = pd.DataFrame({'CONTROLN': dataset_test['CONTROLN'], 'TARGET_B': predictions})
    prediction_dataframe.to_csv(kdd_data_folder / 'prediction-knn.csv', index=False)


def performn_knn_on_kdd_nominal(n_neighbors: int, weight: str):
    dataset_name = 'kdd'
    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['no_nominal_path_train'])
    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['no_nominal_path_test'])

    # do not include the target column (which is the last one) on data dataframe
    x_train = dataset_train.iloc[:, :-1]
    y_train = dataset_train.iloc[:, dataset_train.columns.get_loc('TARGET_B')]

    classifier: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)


