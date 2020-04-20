import sys
import calendar

import pandas as pd
from classifiers.config import *


def value_contains_invalid_dates(value: str) -> bool:
    """Returns a boolean which tells whether value contains a substring with month abbreviations (which is what
    happens quite often in the breast cancer dataset)"""

    month_abbreviations = [i.lower() for i in calendar.month_abbr]\

    # remove first element, which strangely is an empty char
    month_abbreviations.remove('')
    if any(month_abbr in value.lower() for month_abbr in month_abbreviations):
        print('Found one invalid value {}'.format(value))
        return True

    # no month abbr found
    return False


def main():
    """
    Pre-processes breast cancer dataset and stores it

    Procedure:
        - Load dataset
        - Put class column (which is our target) as the very last one
    """
    dataset_name = 'breast_cancer'

    dataset_train: pd.DataFrame = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
        names=['ID', 'class', 'radiusMean', 'textureMean', 'perimeterMean', 'areaMean', 'smoothnessMean', 'compactnessMean',
         'concavityMean', 'concavePointsMean', 'symmetryMean', 'fractalDimensionMean', 'radiusStdErr', 'textureStdErr',
         'perimeterStdErr', 'areaStdErr', 'smoothnessStdErr', 'compactnessStdErr', 'concavityStdErr',
         'concavePointsStdErr', 'symmetryStdErr', 'fractalDimensionStdErr', 'radiusWorst', 'textureWorst',
         'perimeterWorst', 'areaWorst', 'smoothnessWorst', 'compactnessWorst', 'concavityWorst', 'concavePointsWorst',
         'symmetryWorst', 'fractalDimensionWorst']
    )

    # put target column (which in this case is class) as the last column
    class_column = dataset_train['class']
    dataset_train.drop(columns='class', inplace=True)
    dataset_train.insert(len(dataset_train.columns), 'class', class_column)

    print('Storing preprocessed datasets...')
    dataset_train.to_csv(breast_cancer_data_folder / 'breast-cancer-train-preprocessed.csv', index=False)

    print('Done preprocessing!')


if __name__ == '__main__':
    main()
    sys.exit(0)
