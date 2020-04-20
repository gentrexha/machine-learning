import sys
import pandas as pd
import matplotlib.pyplot as plt
from classifiers.config import *


def plot_corr(df, size=40):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('foo.png')


def main():
    dataset_name = 'KDD'
    train_dataset: pd.DataFrame = pd.read_csv(DATASETS[dataset_name]['path'], low_memory=False)
    plot_corr(train_dataset)
    print("Done!")


if __name__ == '__main__':
    main()
    sys.exit(0)
