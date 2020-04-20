import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import preprocessing
from math import sqrt

# Folders to access /raw/ data more easily
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

beijing_pm25_data_folder = Path('../../data/raw/BeijingPM2.5/')
bike_sharing_data_folder = Path('../../data/raw/BikeSharing/')
crime_data_folder = Path('../../data/raw/Crime/')
student_performance_data_folder = Path('../../data/raw/StudentPerformance/')

# Dictionary with all the datasets
DATASETS = {
    'beijing_pm25': {
        'raw': beijing_pm25_data_folder / 'PRSA_data_2010.1.1-2014.12.31.csv',
        'na_values': '',
    },
    'bike_sharing': {
        'raw': bike_sharing_data_folder / 'bikeSharing.shuf.train.csv',
        'test': bike_sharing_data_folder / 'bikeSharing.shuf.test.csv',
        'na_values': '',
    },
    'crime': {
        'raw': crime_data_folder / 'crimedata.csv',
        'ml': crime_data_folder / 'crimedata_ml.csv',
        'interim': crime_data_folder / 'crimedata_interim.csv',
        'filled': crime_data_folder / 'crimedata_filled.csv',
        'feature': crime_data_folder / 'crimedata_feature.csv',
        'na_values': '?',
    },
    'student_performance': {
        'raw': student_performance_data_folder / 'StudentPerformance.shuf.train.csv',
        'test': bike_sharing_data_folder / 'StudentPerformance.shuf.test.csv',
        'na_values': '',
    },
}

# Folders to access /reports/figures/ more easily
beijing_pm25_figures_folder = Path('../../reports/figures/BeijingPM2.5/')
bike_sharing_figures_folder = Path('../../reports/figures/BikeSharing/')
crime_figures_folder = Path('../../reports/figures/Crime/')
student_performance_figures_folder = Path('../../reports/figures/StudentPerformance/')

# Easy access to Reports folder paths
REPORTS = {
    'beijing_pm25': {
        'figures': beijing_pm25_figures_folder,
    },
    'bike_sharing': {
        'figures': bike_sharing_figures_folder,
    },
    'crime': {
        'figures': crime_figures_folder,
    },
    'student_performance': {
        'figures': student_performance_figures_folder,
    },
}


def analyze_raw_dataset(dataset_name: str):
    """
    Analyzes the raw data sets.
    Possible values: { 'beijing_pm25', 'bike_sharing', 'crime', 'student_performance'}
    :return: Output in the console with some information about the dataset.
    """

    # Loads raw data
    raw_dat = load_raw_data(dataset_name=dataset_name)
    if raw_dat is None:
        print('Something went wrong.')
        return
    print('\nLoaded dataset: {}.'.format(dataset_name))

    folder = REPORTS[dataset_name]['figures']

    # Exploratory Analysis
    print("\nShape:")
    print(raw_dat.shape)
    print("\nCount:")
    print(raw_dat.count())  # checks how many missing values are in the dataset
    missing_values_table(raw_dat)
    print("\nHead:")
    print(raw_dat.head())
    print("\nColumns:")
    print(raw_dat.columns)

    print('\nTotal Missing Values:')
    print(raw_dat.isnull().sum())

    # if dataset_name == 'kdd':
    #     target_feature_str = 'TARGET_B'
    #     # Find missing values per column and plot
    #     series = pd.DataFrame(raw_dat.isna().sum())
    #     series = series[~(series == 0).any(axis=1)]
    #     series.plot.barh(figsize=(24, 18), color='steelblue')
    #     plt.savefig(folder / '{}{}'.format(dataset_name, '_missing_values.png'), dpi=300)
    #     plt.clf()
    # elif dataset_name == 'bank_marketing':
    #     target_feature_str = 'y'
    # elif dataset_name == 'image_segmentation':
    #     target_feature_str = 'CLASS'
    # elif dataset_name == 'breast_cancer':
    #     target_feature_str = 'class'
    # else:
    #     print('Error: Couldnt find specified dataset.')
    #     return

    # Pie chart % of target
    # (raw_dat[target_feature_str].value_counts(normalize=True) * 100) \
    #     .plot.pie(figsize=(6, 6), colormap='tab20c', autopct='%.2f%%')
    # plt.savefig(folder / '{}{}'.format(dataset_name, '_target_feature_percentage.png'), dpi=300)
    # plt.clf()

    # Dataset Histograms
    print('\nPlotting Histograms.')
    raw_dat.hist(figsize=(24, 18), color='steelblue')
    plt.savefig(folder / '{}{}'.format(dataset_name, '_histograms.png'), dpi=300)
    plt.clf()

    # Plot correlation matrix of all features
    print('\nPlotting Correlation Matrix.')
    corr = raw_dat.corr()

    # Seaborn method
    sns.set(style="white")

    # Generate a mask for the upper triangle
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(24, 18))

    # Generate a custom diverging colormap
    corr_map = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=corr_map)

    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(folder / '{}{}'.format(dataset_name, '_sns_corr_matrix.png'), dpi=300)
    plt.clf()

    # Some statistics about raw_dat's variables
    print("\nDescribe:")
    print(raw_dat.describe())


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("\nYour selected dataframe has " + str(df.shape[1]) + " columns.\nThere are " + str(
        mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    return mis_val_table_ren_columns


def load_raw_data(dataset_name: str):
    """
    Loads raw datasets for train and test
    :return: raw train and test dataset
    """
    try:
        raw_dat: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['raw'],
                                            na_values=DATASETS[dataset_name]['na_values'], low_memory=False)
        return raw_dat
    except KeyError:
        print('Dataset with key {} doesnt exist.'.format(dataset_name))
        return None


def get_dataset_normalized(dataset: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe after normalizing it's values"""
    print('Normalizing dataset...')
    return pd.DataFrame(preprocessing.normalize(dataset), columns=dataset.columns.values)


def get_dataset_scaled(dataset: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe after scaling it's values"""
    print('Scaling dataset...')
    return pd.DataFrame(preprocessing.scale(dataset), columns=dataset.columns.values)


def get_dataset_min_max_scaled(dataset: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe after scaling it's values"""
    print('Min-max scaling dataset...')
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(dataset)
    return pd.DataFrame(scaled_df, columns=dataset.columns.values)


def print_measurement_scores(y, y_pred, model: str, desc: str):
    """
    y: actual target values.
    y_pred: predicted target values.
    model: Model used to make prediction.
    desc: more information about data and model used.
    Prints various Regression Measurement Scores.
    More info here: https://stackoverflow.com/questions/17197492/root-mean-square-error-in-python
    """
    # print('\n')
    print('-' * 40)
    print('Model used: {}. Description: {}'.format(model, desc))
    print('-' * 40 + 'v')
    print('MAE: {}'.format(mean_absolute_error(y, y_pred)))
    print('MSE: {}'.format(mean_squared_error(y, y_pred)))
    print('R2: {}'.format(r2_score(y, y_pred)))
    print('MAE: {}'.format(median_absolute_error(y, y_pred)))
    print('RMSE: {}'.format(sqrt(mean_squared_error(y, y_pred))))
    print('-' * 40 + '^')

