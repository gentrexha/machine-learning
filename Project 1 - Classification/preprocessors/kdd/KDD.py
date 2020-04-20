import sys

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from classifiers.config import DATASETS, kdd_data_folder


def split_mdmaud_and_domain(dataset: pd.DataFrame) -> pd.DataFrame:
    # split the data from MDMAUD column to three meaningful columns
    dataset = dataset.assign(
        RecencyOfGiving=dataset['MDMAUD'].apply(lambda x: get_recency_of_giving(x)))

    dataset = dataset.assign(
        FrequencyOfGiving=dataset['MDMAUD'].apply(lambda x: x[1] if x and x[1] != 'X' else 0))

    dataset = dataset.assign(
        AmountOfGiving=dataset['MDMAUD'].apply(lambda x: get_amount_of_giving_level(x)))

    dataset.drop(columns='MDMAUD', inplace=True)

    # split DOMAIN column into two meaningful columns
    dataset = dataset.assign(
        UrbanicityLevel=dataset['DOMAIN'].apply(lambda x: get_urbanicity_level(x)))

    dataset = dataset.assign(
        SocioEconomicStatus=dataset['DOMAIN'].apply(lambda x: get_socio_economic_status(x)))

    dataset.drop(columns='DOMAIN', inplace=True)
    return dataset


def get_recency_of_giving(value: str):
    """
        MDMAUD column description:

        The Major Donor Matrix code
        The codes describe frequency and amount of
        giving for donors who have given a $100+
        gift at any time in their giving history.
        An RFA (recency/frequency/monetary) field.

        The (current) concatenated version is a nominal
        or symbolic field. The individual bytes could separately be
        used as fields and refer to the following:

        First byte: Recency of Giving
          C=Current Donor
          L=Lapsed Donor
          I=Inactive Donor
          D=Dormant Donor

        2nd byte: Frequency of Giving
          1=One gift in the period of recency
          2=Two-Four gifts in the period of recency
          5=Five+ gifts in the period of recency

        3rd byte: Amount of Giving
          L=Less than $100(Low Dollar)
          C=$100-499(Core)
          M=$500-999(Major)
          T=$1,000+(Top)

        4th byte: Blank/meaningless/filler
        'X' indicates that the donor is not a major donor.

    For the first bit (RecencyOfGiving), we map as follows:
        Current = 4,
        Lapsed = 3,
        Dormant = 2,
        Inactive = 1,
        None = 0
    """
    if not value or value[0] == 'X':
        return 0

    if value[0] == 'C':
        return 4
    elif value[0] == 'L':
        return 3
    elif value[0] == 'D':
        return 2
    elif value[0] == 'O':
        return 1
    else:
        return 0


def get_amount_of_giving_level(value: str):
    """
    For the third bit (AmountOfGiving), we map as follows:
        L=Less than $100(Low Dollar) - 1
        C=$100-499(Core) - 2
        M=$500-999(Major) - 3
        T=$1,000+(Top) - 4
        None - 0
    """
    if not value or value[2] == 'X':
        return 0

    if value[2] == 'L':
        return 1
    elif value[2] == 'C':
        return 2
    elif value[2] == 'M':
        return 3
    elif value[2] == 'T':
        return 4
    else:
        return 0


def get_urbanicity_level(value: str):
    """
    DOMAIN column description:

    DOMAIN/Cluster code. A nominal or symbolic field.
    could be broken down by bytes as explained below.

    1st byte = Urbanicity level of the donor's neighborhood
      U=Urban
      C=City
      S=Suburban
      T=Town
      R=Rural

    2nd byte = Socio-Economic status of the neighborhood
      1 = Highest SES
      2 = Average SES
      3 = Lowest SES (except for Urban communities, where
          1 = Highest SES, 2 = Above average SES,
          3 = Below average SES, 4 = Lowest SES.)

    For the Urbanicity level, we do the following mapping:
        U=Urban - 5
        C=City - 4
        S=Suburban - 3
        T=Town - 2
        R=Rural - 1
        None - 0
    """
    if not value or value[0] == 'X':
        return 0

    if value[0] == 'R':
        return 1
    elif value[0] == 'T':
        return 2
    elif value[0] == 'S':
        return 3
    elif value[0] == 'C':
        return 4
    elif value[0] == 'U':
        return 5
    else:
        return 0


def get_socio_economic_status(value: str):
    """
    Keep the same variables for SES:
    1 = Highest SES
    2 = Average SES
    3 = Lowest SES (except for Urban communities, where
      1 = Highest SES, 2 = Above average SES,
      3 = Below average SES, 4 = Lowest SES.)

    """
    # sometimes value is just a string which contains a white space. Remove it
    value = value.replace(' ', '')

    if not value or value[1] == 'X':
        return 0
    else:
        return value[1]


def get_cluster_as_float(value: str):
    """
    """
    value = value.replace(' ', '')
    try:
        return float(value)
    except ValueError:
        return pd.np.nan


def main():
    """
    Pre-processes kdd dataset and stores it

    Procedure:
        - Load dataset
        - Perform pre processing as described in README
        - Store preprocessed dataset
        - Apply classification algorithms on the preprocessed file

    Here we preprocess only the columns in the `target_variables` list, which were picked from
    https://pdfs.semanticscholar.org/865a/6dba275f21ea42a10616f59d85da6d26eae1.pdf, page 75+
    """
    dataset_name = 'kdd'

    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['initial_path_train'])
    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['initial_path_test'])

    target_variables = ['RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'MDMAUD', 'DOMAIN', 'CLUSTER', 'HOMEOWNR',
                        'NUMCHLD',
                        'INCOME', 'GENDER', 'WEALTH1']

    flag_variables = ['RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'HOMEOWNR', 'WEALTH1']

    # replace flag variables with either 1 or 0
    for column in flag_variables:
        dataset_train[column] = dataset_train[column].apply(lambda x: 1 if x == 'X' else 0)
        dataset_test[column] = dataset_test[column].apply(lambda x: 1 if x == 'X' else 0)

    # update gender column with M=1, F=0
    dataset_train['GENDER'] = dataset_train['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
    dataset_test['GENDER'] = dataset_test['GENDER'].apply(lambda x: 1 if x == 'M' else 0)

    # split MDMAUD and DOMAIN columns
    dataset_train = split_mdmaud_and_domain(dataset_train)
    dataset_test = split_mdmaud_and_domain(dataset_test)

    dataset_train['CLUSTER'] = dataset_train['CLUSTER'].apply(lambda x: get_cluster_as_float(x))
    dataset_test['CLUSTER'] = dataset_test['CLUSTER'].apply(lambda x: get_cluster_as_float(x))

    # replace NaN's foreach column with the mean value
    dataset_train.fillna(dataset_train.mean(), inplace=True)
    dataset_test.fillna(dataset_test.mean(), inplace=True)

    # put target column (which in this case is class) as the last column
    class_column = dataset_train['TARGET_B']
    dataset_train.drop(columns='TARGET_B', inplace=True)
    dataset_train.insert(len(dataset_train.columns), 'TARGET_B', class_column)

    print('Storing preprocessed datasets...')
    dataset_train.to_csv(kdd_data_folder / 'kdd-train-preprocessed.csv', index=False)
    dataset_test.to_csv(kdd_data_folder / 'kdd-test-preprocessed.csv', index=False)

    print('Done preprocessing!')


def no_nominal():
    """
    Preprocess kdd without any nominal values and store it.
    :return:
    """
    dataset_name = 'kdd'

    dataset_train: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['initial_path_train'])
    dataset_test: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['initial_path_test'])

    # remove all nominal values
    dataset_train = dataset_train._get_numeric_data()
    dataset_test = dataset_test._get_numeric_data()

    # replace NaN's foreach column with the mean value
    dataset_train.fillna(dataset_train.mean(), inplace=True)
    dataset_test.fillna(dataset_test.mean(), inplace=True)

    # put target column (which in this case is class) as the last column
    class_column = dataset_train['TARGET_B']
    dataset_train.drop(columns='TARGET_B', inplace=True)
    dataset_train.insert(len(dataset_train.columns), 'TARGET_B', class_column)

    print('Storing preprocessed datasets...')
    dataset_train.to_csv(kdd_data_folder / 'kdd-train-preprocessed-no_nominal.csv', index=False)
    dataset_test.to_csv(kdd_data_folder / 'kdd-test-preprocessed-no_nominal.csv', index=False)

    print('Done preprocessing!')


if __name__ == '__main__':
    # main()
    no_nominal()
    sys.exit(0)
