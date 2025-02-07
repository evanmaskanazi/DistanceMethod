import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from scipy.spatial import KDTree
import scipy.stats
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def gs(A):
    """Gram-Schmidt orthogonalization exactly as in original."""
    Q, R = np.linalg.qr(A)
    return Q


def calculate_distances(gstest):
    """Calculate inverse distances between points."""
    n = gstest.shape[0]
    gsarr0 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j and not np.array_equal(gstest[i], gstest[j]):
                gsarr0[i][j] = pow(np.linalg.norm(gstest[i] - gstest[j]), -1.0)
            elif i == j:
                gsarr0[i][j] = 0

    gsarrinv = np.zeros(n)
    for i in range(n):
        gsarrinv[i] = 1.0 / pow(np.sum(gsarr0, axis=1)[i], 1)

    return gsarrinv


def calculate_bin_errors(gsarrinv, trainout, y_test):
    """Calculate errors for each bin."""
    splita = np.array_split(np.sort(gsarrinv), 10)
    bin_errors = []

    for split in splita:
        err = np.empty((0, 3), float)
        for i in range(len(gsarrinv)):
            if gsarrinv[i] in split:
                err = np.append(err, np.array([[trainout[i], y_test[i], gsarrinv[i]]]), axis=0)
        bin_errors.append(np.mean(err.T[0]) if len(err) > 0 else np.mean(trainout))

    return np.array(bin_errors)


def main():
    # Load data
    dftest0 = pd.read_csv('testEgEf.txt')
    dftrain0 = pd.read_csv('trainEgEf.txt')

    dftest = dftest0.drop("Ef", axis=1)
    dftrain = dftrain0.drop("Ef", axis=1)
    X_train = dftrain.drop("Eg", axis=1)
    X_test = dftest.drop("Eg", axis=1)
    y_train = dftrain['Eg'].values.astype('float')
    y_test = dftest['Eg'].values.astype('float')

    # Train SVR model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    param_grid = {
        'svr__C': [100],
        'svr__gamma': ['auto'],
        'svr__kernel': ['rbf'],
        'svr__epsilon': [0.001]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    # Calculate prediction errors
    prederror = np.abs(y_pred - y_test)

    # Stack data exactly as in original
    stack1 = np.vstack((np.array(X_test).T, prederror, y_test, y_pred))
    trainpdf1 = stack1.T
    trainint = np.array(trainpdf1.T[0:int(np.array(X_train).shape[1])].T)
    trainout = prederror

    # Get feature importance using same grid parameters
    grid0 = GridSearchCV(pipeline, param_grid, cv=5)
    grid0.fit(trainint, trainout)
    r = permutation_importance(grid0, trainint, trainout, n_repeats=30, random_state=0)

    # 1. GS and Feature Importance
    gstest = gs(trainint)
    for i in range(gstest.shape[1]):
        gstest.T[i] = gstest.T[i] * (1.0 * pow(r.importances_mean[i], 0.5) + 0.0)
    VecStd = calculate_bin_errors(calculate_distances(gstest), trainout, y_test)

    # 2. No GS or Feature Importance
    gstest = trainint.copy()
    VecStdN = calculate_bin_errors(calculate_distances(gstest), trainout, y_test)

    # 3. Ten Points with GS and Feature Importance
    A = gs(np.array(X_train))
    X_test_mod = X_test.copy()
    for i in range(A.shape[1]):
        A.T[i] = A.T[i] * (1.0 * pow(r.importances_mean[i], 0.5) + 0.0)
        X_test_mod.iloc[:, i] = X_test_mod.iloc[:, i] * (1.0 * pow(r.importances_mean[i], 0.5) + 0.0)

    disttest2 = np.zeros(len(X_test))
    gs_test_mod = gs(np.array(X_test_mod))
    for i in range(len(X_test)):
        disttest2[i] = np.sum(KDTree(A).query(gs_test_mod[i], 10)[0]) / 10
    VecStdGSv = calculate_bin_errors(disttest2, trainout, y_test)

    # 4. Ten Points without GS or Feature Importance
    disttest2 = np.zeros(len(X_test))
    for i in range(len(X_test)):
        disttest2[i] = np.sum(KDTree(np.array(X_train)).query(np.array(X_test)[i], 10)[0]) / 10
    VecStdv = calculate_bin_errors(disttest2, trainout, y_test)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    Vec = np.arange(1, 11)

    plt.plot(Vec, VecStd, 'b', label='GS and Feature Importance')
    plt.plot(Vec, VecStdN, 'g', label='No GS or Feature Importance')
    plt.plot(Vec, VecStdGSv, 'r', label='Ten Points, GS and Feature Importance')
    plt.plot(Vec, VecStdv, 'k', label='Ten Points, No GS or Feature Importance')

    legend = plt.legend(loc='upper left', shadow=True, fontsize='medium')
    legend.get_frame().set_facecolor('w')

    # Configure axis
    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, which='both', direction='in', width=2, length=4,
                       bottom=True, top=False, left=True, right=False,
                       labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        getattr(ax, f'{axis}axis').set_major_locator(MultipleLocator(1.0 if axis == 'x' else 0.02))
        getattr(ax, f'{axis}axis').set_minor_locator(MultipleLocator(1.0 if axis == 'x' else 0.02))

    ax.grid(True)
    ax.set_facecolor('w')
    for spine in ax.spines.values():
        spine.set_color('black')

    plt.xlabel('Group Number', fontsize=15)
    plt.ylabel('Error (Counts)', fontsize=15)
    plt.savefig('test_simplified.svg')

    # Print statistics
    mean_trainout = np.mean(trainout)
    print(np.sum(abs(np.array(VecStd) - mean_trainout)) / mean_trainout)
    print(np.sum(abs(np.array(VecStdv) - mean_trainout)) / mean_trainout)
    print("GS/FT", scipy.stats.spearmanr(Vec, VecStd)[0])
    print("No GS/FT", scipy.stats.spearmanr(Vec, VecStdv)[0])


if __name__ == "__main__":
    main()
