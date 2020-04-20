from pathlib import Path

bank_data_folder = Path('../datasets/bank/')
image_segmentation_folder = Path('../datasets/image_segmentation/')
breast_cancer_data_folder = Path('../datasets/Kaggle-BreastCancer/')
kdd_data_folder = Path('../datasets/Kaggle-KDD/')

kdd_examples_folder = Path('../examples/kdd/')
bank_marketing_examples_folder = Path('../examples/bank_marketing/')
breast_cancer_examples_folder = Path('../examples/breast_cancer/')
image_segmentation_examples_folder = Path('../examples/image_segmentation/')

IMAGE_SEGMENTATION_CLASSES = {
    'GRASS': 1,
    'BRICKFACE': 2,
    'SKY': 3,
    'FOLIAGE': 4,
    'CEMENT': 5,
    'WINDOW': 6,
    'PATH': 7
}


DATASETS = {
    'bank_marketing': {
        'preprocessed_path': bank_data_folder / 'bank-full-preprocessed.csv',
        'initial_path_train': bank_data_folder / 'bank-full.csv',

        'separator': ';'
    },
    'image_segmentation': {
        # image segmentation has mixed data, therfore train uses test
        'preprocessed_path_train': image_segmentation_folder / 'segmentation.data',
        'initial_path_train': image_segmentation_folder / 'segmentation.data',
        'preprocessed_path_test': image_segmentation_folder / 'segmentation.test',

        'separator': ',',
    },
    'breast_cancer': {
        'preprocessed_path_train': breast_cancer_data_folder / 'breast-cancer-train-preprocessed.csv',
        'preprocessed_path_test': breast_cancer_data_folder / 'breast-cancer-test-preprocessed.csv',

        'initial_path_train': breast_cancer_data_folder / 'breast-cancer-diagnostic.shuf.lrn.csv',
        'initial_path_test': breast_cancer_data_folder / 'breast-cancer-diagnostic.shuf.tes.csv',

        'separator': ',',
    },
    'kdd': {
        'preprocessed_path_train': kdd_data_folder / 'kdd-train-preprocessed.csv',
        'preprocessed_path_test': kdd_data_folder / 'kdd-test-preprocessed.csv',

        'initial_path_train': kdd_data_folder / 'train.csv',
        'initial_path_test': kdd_data_folder / 'test.csv',

        'no_nominal_path_train': kdd_data_folder / 'kdd-train-preprocessed-no_nominal.csv',
        'no_nominal_path_test': kdd_data_folder / 'kdd-test-preprocessed-no_nominal.csv',

        'separator': ',',
    }
}
