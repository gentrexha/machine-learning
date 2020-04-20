import sys

import pandas as pd
from classifiers.config import *


def main():
    """
    Debug only purposes. No real pre-processing needed for image_segmentation
    """
    dataset_name = 'image_segmentation'

    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    print('Done preprocessing!')


if __name__ == '__main__':
    main()
    sys.exit(0)
