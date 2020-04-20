import sys

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from utils import load_raw_data, DATASETS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


def remove_unnecessary_columns_crime_dataset():

    df = load_raw_data('crime')

    # Drop non-predictive
    # -- communityname: Community name - not predictive - for information only (string)
    # -- countyCode: numeric code for county - not predictive, and many missing values (numeric)
    # -- communityCode: numeric code for community - not predictive and many missing values (numeric)
    # -- fold: fold number for non-random 10 fold cross validation, potentially useful for debugging, paired tests -
    # not predictive (numeric - integer)

    # Drop other to be predicted columns, because having a model with others would be not fair
    # -- murdPerPop: number of murders per 100K population
    # -- rapes: number of rapes in 1995
    # -- rapesPerPop: number of rapes per 100K population
    # -- robberies: number of robberies in 1995
    # -- robbbPerPop: number of robberies per 100K population
    # -- assaults: number of assaults in 1995
    # -- assaultPerPop: number of assaults per 100K population
    # -- burglaries: number of burglaries in 1995
    # -- burglPerPop: number of burglaries per 100K population
    # -- larcenies: number of larcenies in 1995
    # -- larcPerPop: number of larcenies per 100K population
    # -- autoTheft: number of auto thefts in 1995
    # -- autoTheftPerPop: number of auto thefts per 100K population
    # -- arsons: number of arsons in 1995
    # -- arsonsPerPop: number of arsons per 100K population
    # -- ViolentCrimesPerPop: total number of violent crimes per 100K
    # -- nonViolPerPop: total number of non-violent crimes per 100K population

    # Binarize
    # -- state: US state (by 2 letter postal abbreviation)(nominal)

    df = df.drop(['communityname', 'countyCode', 'communityCode', 'fold', 'murders', 'murdPerPop', 'rapes',
                  'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop',
                  'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
                  'nonViolPerPop'], axis=1)

    # Dummy encoding states
    df = pd.get_dummies(df)

    # Saving to CSV
    df.to_csv(DATASETS['crime']['ml'])


def remove_missing_values_crime_dataset():
    # Remove features where there's more than 80% missing data
    df = pd.read_csv(DATASETS['crime']['ml'])
    df = df.loc[:, df.isnull().sum() < 0.8 * df.shape[0]]
    # Idk how this column got created
    df = df.drop(['Unnamed: 0'], axis=1)
    df.to_csv(DATASETS['crime']['interim'])


def fill_missing_values_crime_dataset():
    df = pd.read_csv(DATASETS['crime']['interim'])
    # Impute Mean to others
    df = df.fillna(df.mean())
    # Idk how this column got created
    df = df.drop(['Unnamed: 0'], axis=1)
    df.to_csv(DATASETS['crime']['filled'])


def feature_selection_crime_dataset():

    # Load data
    df = load_raw_data('crime')

    # Drop other predictors and non predictive values
    df = df.drop(['communityname', 'countyCode', 'communityCode', 'fold', 'murders', 'murdPerPop', 'rapes',
                  'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop',
                  'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
                  'nonViolPerPop'], axis=1)

    # Dummy encoding states
    df = pd.get_dummies(df)

    # Drop columns with 80% missing data
    df = df.loc[:, df.isnull().sum() < 0.8 * df.shape[0]]

    # Impute Mean to others
    df = df.fillna(df.mean())

    # Set target column as last column
    target_column = df['ViolentCrimesPerPop']
    df.drop(columns='ViolentCrimesPerPop', inplace=True)
    df.insert(len(df.columns), 'ViolentCrimesPerPop', target_column)

    # Split data
    X = df.drop(['ViolentCrimesPerPop'], axis=1)
    y = df['ViolentCrimesPerPop']

    # Scale methods to try
    scalers = [StandardScaler(), MinMaxScaler()]

    for scaler in scalers:
        # Scale data
        X = scaler.fit_transform(X)

        # create the RFE model for the svm classifier
        # and select attributes
        estimator = LinearRegression()
        selector = RFE(estimator=estimator, n_features_to_select=15, step=1, verbose=2)
        selector = selector.fit(X, y)
        # print summaries for the selection of attributes
        print(selector.support_)
        print(selector.ranking_)

        # Define list with indexes of best selected features
        feature_list = []

        # Find indexes of best features
        for index, item in enumerate(selector.ranking_):
            if item == 1:
                feature_list.append(index)

        # Use these features in a pd.DataFrame and export it to CSV
        feature_selected_df = pd.DataFrame(data=X[:, feature_list], columns=df.columns[feature_list].values.tolist())
        feature_selected_df['ViolentCrimesPerPop'] = y
        feature_selected_df.to_csv('C:/Projects/University/Machine Learning/ml-regression/data/raw/Crime/crimedata_feature_'
                  + str(scalers.index(scaler))+'.csv')
        print('Saved CSV successfully!')


def feature_selection_bike_dataset():
    # Load and split data
    df = load_raw_data('bike_sharing')

    # Split dataset
    df = df.drop(['dteday', 'id'], axis=1)
    X = df.drop(['cnt'], axis=1)
    y = df['cnt']

    # Scale methods to try
    scaler = MinMaxScaler()

    # Scale data
    X = scaler.fit_transform(X)

    # create the RFE model for the svm classifier
    # and select attributes
    estimator = LinearRegression()
    selector = RFE(estimator=estimator, step=1, verbose=2)
    selector = selector.fit(X, y)
    # print summaries for the selection of attributes
    print(selector.support_)
    print(selector.ranking_)

    # Define list with indexes of best selected features
    feature_list = []

    # Find indexes of best features
    for index, item in enumerate(selector.ranking_):
        if item == 1:
            feature_list.append(index)

    # Use these features in a pd.DataFrame and export it to CSV
    feature_selected_df = pd.DataFrame(data=X[:, feature_list], columns=df.columns[feature_list].values.tolist())
    feature_selected_df['cnt'] = y
    feature_selected_df.to_csv('C:/Projects/University/Machine '
                               'Learning/ml-regression/data/raw/BikeSharing/bikesharing_feature.csv')
    print('Saved CSV successfully!')


def main():
    # remove_unnecessary_columns_crime_dataset()
    # remove_missing_values_crime_dataset()
    # fill_missing_values_crime_dataset()
    # feature_selection_crime_dataset()
    feature_selection_bike_dataset()

if __name__ == '__main__':
    main()
    sys.exit(0)
