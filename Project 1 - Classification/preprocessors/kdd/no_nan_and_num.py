import sys
from classifiers.utils import *


def raw_classification():
    """
    Raw classification without any preprocessing at all.
    :return:
    """

    # Loads raw data
    raw_dat_train = load_raw_data('kdd')
    # raw_dat_test
    # Classification on raw data
    # classify_all(raw_dat_train, "Raw data without any classification")

    # Classification on numerical values only and features w/o missing values
    # https://www.quora.com/How-can-I-use-KNN-for-mixed-data-categorical-and-numerical
    train_dat = raw_dat_train.select_dtypes(include=[np.number])
    # Drop columns with NaN values
    train_dat = train_dat.dropna(axis=1)

    # do not include ID on the model
    columns_without_ID = list(train_dat.columns)
    columns_without_ID.remove('CONTROLN')

    classify_all(train_dat[columns_without_ID], 'TARGET_B', 'Numerical values only and features w/o missing values')


def main():
    raw_classification()


if __name__ == '__main__':
    analyze_raw_dataset('breast_cancer')
    # main()
    sys.exit(0)
