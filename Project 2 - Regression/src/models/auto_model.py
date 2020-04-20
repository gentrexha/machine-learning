import sys
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from utils import load_raw_data
import pandas as pd


def auto_bike():
    # Load dataset
    df = load_raw_data('bike_sharing')

    # Split dataset
    X = df.drop(['cnt', 'dteday', 'id'], axis=1)
    y = df['cnt']

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_bike_sharing_pipeline.py')


def auto_crime():
    # Load dataset
    df = load_raw_data('crime')

    # Dropping nonpredictive and other to be predicted columns
    # 'state' is dropped  # before not dropped
    df = df.drop(['state', 'communityname', 'countyCode', 'communityCode', 'fold', 'murders', 'murdPerPop', 'rapes',
                  'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop',
                  'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
                  'nonViolPerPop'], axis=1)

    # Dummy encoding states
    # df = pd.get_dummies(df)

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

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_crime_pipeline.py')


def auto_beijing():
    # Load dataset
    df = pd.read_csv('C:/Projects/University/Machine Learning/ml-regression/data/processed/PRSA_data_2010.1.1-2014.12'
                     '.31_dum.csv')

    # Fill missing values with 0
    df = df.fillna(0)

    # Drop missing values
    df = df.drop(df[df.pm25 == 0].index)

    # Split dataset
    X = df.drop(['No', 'pm25', 'cbwd'], axis=1)
    y = df['pm25']

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_bike_sharing_pipeline.py')


def auto_student_performance():
    # Load dataset
    df = pd.read_csv('C:/Projects/University/Machine Learning/ml-regression/data/processed/StudentPerformance.shuf.train_dum.csv')

    # Split
    X = df.drop(
        ['id', 'Grade', 'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
         'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], axis=1)

    # save the labels in y
    y = df['Grade']

    # Train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_bike_sharing_pipeline.py')


def main():
    # auto_bike()
    # auto_crime()
    auto_beijing()
    # auto_student_performance()


if __name__ == '__main__':
    main()
    sys.exit(0)
