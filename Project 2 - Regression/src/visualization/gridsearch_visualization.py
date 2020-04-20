import matplotlib.pyplot as plt
import numpy as np
import sys


def param_svr():
    x1 = np.logspace(-15, 5, base=2)  # C
    x2 = np.logspace(-5, 15, base=2)  # Gamma
    N = 50
    y = np.zeros(N)
    plt.plot(x1, y, 'o', label='C')

    plt.plot(x2, y + 0.5, 'o', label='Gamma')

    plt.ylim([-0.5, 1])

    plt.title('SVR Parameter Grid Search')

    plt.legend(loc='upper left')

    plt.savefig('gridsearch_figure.png')
    plt.clf()


def param_linearregression():
    x1 = [0, 1]
    x2 = [0, 1]
    x3 = [0, 1]

    N = 2
    y = np.zeros(N)
    plt.plot(x1, y, 'o', label='fit_intercept')

    plt.plot(x2, y + 0.5, 'o', label='normalize')

    plt.plot(x3, y - 0.5, 'o', label='copy_X')

    plt.ylim([-2, 2])

    plt.title('LinearRegression Parameter Grid Search')

    plt.legend(loc='upper left')

    plt.savefig('gridsearch_linearregression.png')
    plt.clf()


def param_mlpr():
    param_grid = {'solver': ['lbfgs'],
                  'max_iter': [1000, 1500, 2000],
                  'alpha': 10.0 ** -np.arange(1, 10),
                  'hidden_layer_sizes': np.arange(10, 15)}

    x1 = [1000, 1500, 2000]
    x2 = 10.0 ** -np.arange(1, 10)
    x3 = np.arange(10, 15)

    y1 = np.zeros(int(len(x1)))
    y2 = np.zeros(int(len(x2)))
    y3 = np.zeros(int(len(x3)))

    plt.plot(x1, y1, 'o', label='max_iter')

    plt.plot(x2, y2 + 0.5, 'o', label='alpha')

    plt.plot(x3, y3 - 0.5, 'o', label='hidden_layer_sizes')

    plt.ylim([-2, 2])

    plt.title('MLPRegressor Parameter Grid Search')

    plt.legend(loc='upper left')

    plt.savefig('gridsearch_linearregression.png')
    plt.clf()


def param_rfr():
    param_grid = {
        "n_estimators": [10, 20, 30, 50, 100],
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_split": [2, 4, 8, 10, 12, 14, 16],
        "bootstrap": [True, False],
    }

    x1 = [10, 20, 30, 50, 100, 200, 300, 400, 500, 1000]
    x2 = [2, 4, 8, 10, 12, 14, 16]
    x3 = [0, 1]

    y1 = np.zeros(int(len(x1)))
    y2 = np.zeros(int(len(x2)))
    y3 = np.zeros(int(len(x3)))

    plt.plot(x1, y1, 'o', label='n_estimators')

    plt.plot(x2, y2 + 0.5, 'o', label='min_samples_split')

    plt.plot(x3, y3 - 0.5, 'o', label='bootstrap')

    plt.plot([], [], 'o', label='max_features')

    plt.ylim([-2, 2])

    plt.title('RandomForestRegressor Parameter Grid Search')

    plt.legend(loc='upper left')

    plt.savefig('gridsearch_rfr.png')
    plt.clf()


def main():
    param_rfr()


if __name__ == '__main__':
    main()
    sys.exit(0)


