import sys

import pandas as pd
from classifiers.config import DATASETS, bank_data_folder


def main():
    """
    Pre-processes bank marketing dataset and stores it

    Procedure:
        - Load dataset
        - Perform pre processing as described in README
        - Store preprocessed dataset
        - Apply classification algorithms on the preprocessed file
    """
    dataset_name = 'bank_marketing'

    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['initial_path'],
                                        sep=DATASETS[dataset_name]['initial_separator'])

    one_hot_encode_columns = ['job', 'marital', 'contact']
    integer_encode_columns = {
        'education': {
            'unknown': 0,
            'primary': 1,
            'secondary': 2,
            'tertiary': 3

        },
        'month': {
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9,
            'oct': 10,
            'nov': 11,
            'dec': 12
        },
        'poutcome': {
            'failure': 0,
            'unknown': 1,
            'other': 1,
            'success': 2
        }
    }

    # perform one hot encoding for decided columns
    for column in one_hot_encode_columns:
        print('One hot encoding of {}'.format(column))

        # Get one hot encoding of column
        one_hot_encoded = pd.get_dummies(dataset[column], prefix=column)

        insert_at_position = dataset.columns.get_loc(column)
        for test_colum in one_hot_encoded.columns:
            dataset.insert(insert_at_position, test_colum, one_hot_encoded[test_colum].values)
            insert_at_position += 1

        # delete the original columns from the data set (since we now have one-hot-encoded them)
        del dataset[column]

    # perform integer encoding for decided columns
    for column in integer_encode_columns.keys():
        print('Integer encoding of {}'.format(column))

        dataset[column] = dataset[column].apply(lambda x: integer_encode_columns[column].get(x, 'ERROR'))

    yes_no_columns = ['default', 'housing', 'loan']
    # remaining columns have yes/no values. Convert yes=1, no=0 (maybe the classifiers do that anyways, but play safe)
    for column in yes_no_columns:
        print('Yes/No encoding of {}'.format(column))

        dataset[column] = dataset[column].apply(lambda x: 1 if x == 'yes' else 0 if x == 'no' else 'ERROR')

    dataset.to_csv(bank_data_folder / 'bank-full-preprocessed.csv', index=False)

    print('Done preprocessing!')


if __name__ == '__main__':
    main()
    sys.exit(0)
