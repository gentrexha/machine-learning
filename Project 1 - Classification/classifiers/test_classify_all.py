import sys
import pandas as pd

from config import *
from utils import classify_all, get_dataset_normalized, classify_all_multiple


def main():
    # print('Doing BreastCancer normalized')
    # dataset_name = 'breast_cancer'
    #
    # dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])
    #
    # columns_without_ID = list(dataset.columns)
    # columns_without_ID.remove('ID')
    # columns_without_ID.remove('class')
    #
    # dataset[columns_without_ID] = get_dataset_normalized(dataset[columns_without_ID])
    # dataset['class'] = dataset['class'].map(lambda x: 1 if x == 'B' else 0)
    # classify_all(data=dataset, target_feature='class', cv=5, filename='BreastCancerNormalized')

    print('Doing ImageSegmentation')
    dataset_name = 'image_segmentation'

    # Raw
    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])
    dataset['CLASS'] = dataset['CLASS'].map(lambda x: IMAGE_SEGMENTATION_CLASSES[x])
    classify_all_multiple(data=dataset, target_feature='CLASS', cv=5, filename='ImageSegmentationPlain')

    # Normalized
    columns_without_ID = list(dataset.columns)
    columns_without_ID.remove('CLASS')
    dataset[columns_without_ID] = get_dataset_normalized(dataset[columns_without_ID])
    classify_all_multiple(data=dataset, target_feature='CLASS', cv=5, filename='ImageSegmentationNormalized')

    # Feature selection
    classify_all_multiple(data=dataset, target_feature='CLASS', cv=5, filename='ImageSegmentationFeatureSelected')

    # print('Doing BreastCancer normalized')
    # dataset_name = 'bank_marketing'
    # dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path'])
    #
    # columns_without_y = list(dataset.columns)
    # columns_without_y.remove('ID')
    #
    # dataset[columns_without_y] = get_dataset_normalized(dataset[columns_without_y])
    # dataset['y'] = dataset['y'].map(lambda x: 1 if x == 'yes' else 0)
    # classify_all(data=dataset, target_feature='y', cv=5, filename='BankMarketingNormalized')
    #
    # print('Done!')


def test():
    print('Doing BreastCancer k_best')
    dataset_name = 'breast_cancer'

    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])

    columns_without_ID = list(dataset.columns)
    columns_without_ID.remove('ID')
    columns_without_ID.remove('class')

    dataset[columns_without_ID] = get_dataset_normalized(dataset[columns_without_ID])
    dataset['class'] = dataset['class'].map(lambda x: 1 if x == 'B' else 0)
    classify_all(data=dataset, target_feature='class', cv=5, filename='BreastCancerNormalized')

    # print('Doing ImageSegmentation')
    # dataset_name = 'image_segmentation'
    # dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path_train'])
    # dataset['CLASS'] = dataset['CLASS'].map(lambda x: IMAGE_SEGMENTATION_CLASSES[x])
    # classify_all(data=dataset, target_feature='CLASS', cv=5, filename='ImageSegmentationPlain')

    print('Doing BreastCancer normalized')
    dataset_name = 'bank_marketing'
    dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['preprocessed_path'])

    columns_without_y = list(dataset.columns)
    columns_without_y.remove('ID')

    dataset[columns_without_y] = get_dataset_normalized(dataset[columns_without_y])
    dataset['y'] = dataset['y'].map(lambda x: 1 if x == 'yes' else 0)
    classify_all(data=dataset, target_feature='y', cv=5, filename='BankMarketingNormalized')

    print('Done!')


if __name__ == '__main__':
    main()
    sys.exit(0)
