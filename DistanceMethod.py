import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale
from sklearn import datasets
import matplotlib.cm as cm
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import SVR
import seaborn as sns; sns.set()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import re
import gzip
import gensim
import logging
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from csv import reader
from math import sqrt
from sklearn.preprocessing import StandardScaler
#setup word2vec routines for extracting numerical data from text
#for classification of medical descriptions
path = get_tmpfile("word2vec.model")
#import nltk
#nltk.download('punkt')
#from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action='ignore')
import gensim
from gensim.models import Word2Vec
from sklearn.metrics import mean_absolute_error
from numpy.random import seed
from numpy.random import randn
from numpy import percentile
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from scipy.stats import uniform
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
import scipy
import matplotlib.pyplot as pyplot
from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression
modellinear = LinearRegression()
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np
import pandas as pd
from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
import random
import math


from hpsklearn import HyperoptEstimator, any_classifier
from hyperopt import tpe
import numpy as np
from hpsklearn import  random_forest, svc, knn
from hyperopt import hp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from scipy import spatial
from sklearn.preprocessing import StandardScaler
stsc = StandardScaler()
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost
from xgboost import XGBRegressor
xboostr = XGBRegressor()
from sklearn.model_selection import RandomizedSearchCV
param_distxg = {'n_estimators': stats.randint(50, 150),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4],
                'seed':[2]
             }



# Model 3 - Support Vector Regression (SVR)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
tsne= TSNE(n_components=4, learning_rate='auto',init='random')
ldac=LinearDiscriminantAnalysis(n_components=5)
pca = PCA(n_components=5)
from sklearn.feature_selection import SelectKBest, chi2
import xgboost
from xgboost import XGBRegressor

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs = SFS(SVR(),
          k_features=10,
          forward=False,
          floating=False,
          scoring = 'r2',
          cv = 0)
from sklearn.feature_selection import RFE
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)

dftest0 = pd.read_csv(r'testEgEf.txt')
dftrain0 = pd.read_csv(r'trainEgEf.txt')
dftest=dftest0.drop("Ef",1)
dftrain=dftrain0.drop("Ef",1)
X_train1=dftrain.drop("Eg",1)
X_test1=dftest.drop("Eg",1)
y_train1=np.asarray(dftrain['Eg']).astype('float')
y_test1=np.asarray(dftest['Eg']).astype('float')


steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
'SVM__epsilon':[0.001]}, cv=5)






xboostr = XGBRegressor()
param_distxg = {'n_estimators': stats.randint(50, 150),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4],
                'seed':[2]
             }

param_distxgo = {'n_estimators': [100],
              'learning_rate': [0.3],
              'subsample': [0.6],
              'max_depth': [6],
              'colsample_bytree': [0.7],
              'min_child_weight': [3],
                'seed':[2]
             }


grid1=linear_model.LinearRegression()

grid.fit(X_train1, y_train1)
grid1.fit(X_train1,y_train1)

svr_score = grid.score(X_train1,y_train1)
svr_score1 = grid.score(X_test1,y_test1)
y_predicted1 = grid.predict(X_test1)
y_predicted00 = grid.predict(X_test1)



prederror=np.zeros(len(y_predicted1))
preddir=np.zeros(len(y_predicted1))
prederrordist=np.zeros(len(y_predicted1))
for i in range(len(y_predicted1)):
    prederror[i]=abs(y_predicted1[i] - y_test1[i])
    
stack1=np.array(np.vstack((np.array(X_test1).T,prederror,y_test1,y_predicted1)))


print(y_test1,'y test 1')

print(y_predicted1,'y pred 1')

print(mean_absolute_error(y_predicted1,y_test1))

trainpdf1=stack1.T
print(5)

#EfEg

trainint=np.array(trainpdf1.T[0:int(np.array(X_train1).shape[1])].T)


print(np.array(trainint).shape,'trainint')

#EfEg
trainout=np.array(trainpdf1).T[int(np.array(X_train1).shape[1])]




from sklearn.inspection import permutation_importance
grid0= GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
'SVM__epsilon':[0.001]}, cv=5)
grid0.fit(trainint,trainout)

r = permutation_importance(grid0,trainint, trainout,n_repeats=30,
                          random_state=0)




#trainpdf1=np.vstack((stack1.T,stack2.T,stack3.T,stack4.T,stack5.T,stack6.T,stack7.T,stack8.T,stack9.T,stack10.T))
def gs(X):
    Q, R = np.linalg.qr(X)
    return Q
def gram_schmidt(A):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    # Get the number of vectors.
    n = A.shape[1]
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A


gstest=gs(np.array(trainint))

gsarr=np.zeros((np.array(gstest).shape[0],np.array(gstest).shape[0]))

print(np.array(gsarr).shape[0],'int array test')
print(np.array(gstest).shape,np.array(gstest)[0],'int array gs')

print(np.array(gstest[0]))
for i in range(np.array(gstest).shape[1]):
    gstest.T[i] = gstest.T[i] * (1.0*pow(r.importances_mean[i],0.5) + 0.0)
print(np.array(gstest[0]))



gsarr0 = np.zeros((np.array(gstest).shape[0], np.array(gstest).shape[0]))
for i in range(np.array(gsarr).shape[0]):
    for j in range(np.array(gsarr).shape[0]):
        if (i != j and (np.array_equal(gstest[i], gstest[j]) == False)):
          
            gsarr0[i][j] = pow(np.linalg.norm(np.array(gstest)[i] - np.array(gstest)[j]),
                               -1.0)
        elif (i == j):
            gsarr0[i][j] = 0
      
gsarrinv = np.zeros(np.array(gsarr).shape[0])

for i in range(np.array(gsarr0).shape[0]):
    gsarrinv[i] = 1 / (pow(np.sum(gsarr0, axis=1)[i], 1))
   
splita = np.array_split(np.sort(gsarrinv), 10)

err1 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[0]):
        err1 = np.append(err1, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err2 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[1]):
        err2 = np.append(err2, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)

err3 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[2]):
        err3 = np.append(err3,np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err4 =  np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[3]):
        err4 = np.append(err4, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err5 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[4]):
        err5 = np.append(err5, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err6 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[5]):
        err6 = np.append(err6,np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err7 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[6]):
        err7 = np.append(err7, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err8 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[7]):
        err8 = np.append(err8, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err9 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[8]):
        err9 = np.append(err9, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err10 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[9]):
        err10 = np.append(err10, np.array([[trainout[i],y_test1[i],gsarrinv[i]]]),axis=0)
Vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VecStd = [np.mean(err1.T[0]), np.mean(err2.T[0]), np.mean(err3.T[0]), np.mean(err4.T[0]), np.mean(err5.T[0]),
          np.mean(err6.T[0]), np.mean(err7.T[0]), np.mean(err8.T[0]), np.mean(err9.T[0]), np.mean(err10.T[0])]

A = np.array(X_train1)
disttest2 = np.zeros(np.array(trainint).shape[0])
for i in range(np.array(trainint).shape[0]):
    disttest2[i] = np.sum(spatial.KDTree(A).query(np.array(trainint)[i], 10)[0]) / 10

gsarrinv2 = gsarrinv
gsarrinv = disttest2
splita = np.array_split(np.sort(gsarrinv), 10)

err1 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[0]):
        err1 = np.append(err1, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err2 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[1]):
        err2 = np.append(err2, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)

err3 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[2]):
        err3 = np.append(err3,np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err4 =  np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[3]):
        err4 = np.append(err4, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err5 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[4]):
        err5 = np.append(err5, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err6 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[5]):
        err6 = np.append(err6,np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err7 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[6]):
        err7 = np.append(err7, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err8 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[7]):
        err8 = np.append(err8, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err9 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[8]):
        err9 = np.append(err9, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err10 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[9]):
        err10 = np.append(err10, np.array([[trainout[i],y_test1[i],gsarrinv[i]]]),axis=0)
Vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VecStdv = [np.mean(err1.T[0]), np.mean(err2.T[0]), np.mean(err3.T[0]), np.mean(err4.T[0]), np.mean(err5.T[0]),
           np.mean(err6.T[0]), np.mean(err7.T[0]), np.mean(err8.T[0]), np.mean(err9.T[0]), np.mean(err10.T[0])]



gstest=np.array(trainint)
gsarr=np.zeros((np.array(gstest).shape[0],np.array(gstest).shape[0]))
print(np.array(gsarr).shape[0],'int array test')
print(np.array(gstest).shape,np.array(gstest)[0],'int array gs')

print(np.array(gstest[0]))

print(np.array(gstest[0]))

gsarr0 = np.zeros((np.array(gstest).shape[0], np.array(gstest).shape[0]))
for i in range(np.array(gsarr).shape[0]):
    for j in range(np.array(gsarr).shape[0])
        if (i != j and (np.array_equal(gstest[i], gstest[j]) == False)):
           
            gsarr0[i][j] = pow(np.linalg.norm(np.array(gstest)[i] - np.array(gstest)[j]),
                               -1.0)
        elif (i == j):
            gsarr0[i][j] = 0
        
gsarrinv = np.zeros(np.array(gsarr).shape[0])

for i in range(np.array(gsarr0).shape[0]):
    gsarrinv[i] = 1 / (pow(np.sum(gsarr0, axis=1)[i], 1))
 
splita = np.array_split(np.sort(gsarrinv), 10)

err1 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[0]):
        err1 = np.append(err1, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err2 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[1]):
        err2 = np.append(err2, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)

err3 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[2]):
        err3 = np.append(err3,np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err4 =  np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[3]):
        err4 = np.append(err4, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err5 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[4]):
        err5 = np.append(err5, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err6 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[5]):
        err6 = np.append(err6,np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err7 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[6]):
        err7 = np.append(err7, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err8 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[7]):
        err8 = np.append(err8, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err9 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[8]):
        err9 = np.append(err9, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err10 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[9]):
        err10 = np.append(err10, np.array([[trainout[i],y_test1[i],gsarrinv[i]]]),axis=0)
Vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VecStdN = [np.mean(err1.T[0]), np.mean(err2.T[0]), np.mean(err3.T[0]), np.mean(err4.T[0]), np.mean(err5.T[0]),
          np.mean(err6.T[0]), np.mean(err7.T[0]), np.mean(err8.T[0]), np.mean(err9.T[0]), np.mean(err10.T[0])]



A = gs(np.array(X_train1))
X_test1F=X_test1
for i in range(np.array(A).shape[1]):
    A.T[i] = A.T[i] * (1.0 * pow(r.importances_mean[i], 0.5) + 0.0)
    X_test1F.T[i] = X_test1F.T[i] * (1.0 * pow(r.importances_mean[i], 0.5) + 0.0)
disttest2 = np.zeros(np.array(trainint).shape[0])
for i in range(np.array(trainint).shape[0]):
    disttest2[i] = np.sum(spatial.KDTree(A).query(gs(np.array(X_test1F))[i], 10)[0]) / 10

gsarrinv2 = gsarrinv
gsarrinv = disttest2
splita = np.array_split(np.sort(gsarrinv), 10)

err1 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[0]):
        err1 = np.append(err1, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err2 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[1]):
        err2 = np.append(err2, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)

err3 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[2]):
        err3 = np.append(err3,np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err4 =  np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[3]):
        err4 = np.append(err4, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err5 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[4]):
        err5 = np.append(err5, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err6 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[5]):
        err6 = np.append(err6,np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err7 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[6]):
        err7 = np.append(err7, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err8 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[7]):
        err8 = np.append(err8, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err9 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[8]):
        err9 = np.append(err9, np.array([[trainout[i], y_test1[i],gsarrinv[i]]]),axis=0)
err10 = np.empty((0, 3), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[9]):
        err10 = np.append(err10, np.array([[trainout[i],y_test1[i],gsarrinv[i]]]),axis=0)
Vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VecStdGSv = [np.mean(err1.T[0]), np.mean(err2.T[0]), np.mean(err3.T[0]), np.mean(err4.T[0]), np.mean(err5.T[0]),
           np.mean(err6.T[0]), np.mean(err7.T[0]), np.mean(err8.T[0]), np.mean(err9.T[0]), np.mean(err10.T[0])]


fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)

plt.plot(Vec, VecStd, 'b', label=r'GS and Feature Importance')
plt.plot(Vec, VecStdN, 'g', label=r'No GS or Feature Importance')
plt.plot(Vec, VecStdGSv, 'r', label=r'Ten Points, GS and Feature Importance')
plt.plot(Vec, VecStdv, 'k', label=r'Ten Points, No GS or Feature Importance')

legend18 = plt.legend(loc='upper left', shadow=True, fontsize='medium')
legend18.get_frame().set_facecolor('w')
ax2.grid()
ax2.tick_params(which='major', bottom=True, top=False, left=True,
                right=False, width=2, direction='in')
ax2.tick_params(which='major', labelbottom=True, labeltop=False,
                labelleft=True, labelright=False, width=2, direction='in')
ax2.tick_params(axis=u'both', which='major', length=4)
ax2.xaxis.set_major_locator(MultipleLocator(1.0))
ax2.yaxis.set_major_locator(MultipleLocator(0.02))
ax2.tick_params(which='minor', bottom=True, top=False, left=True,
                right=False, width=2, direction='in')
ax2.tick_params(which='minor', labelbottom=True, labeltop=False,
                labelleft=True, labelright=False, width=2, direction='in')
ax2.tick_params(axis=u'both', which='minor', length=4)
ax2.xaxis.set_minor_locator(MultipleLocator(1.0))
ax2.yaxis.set_minor_locator(MultipleLocator(0.02))
ax2.set_facecolor('w')
ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black')
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')
plt.xlabel('Group Number', fontsize=15)
plt.ylabel('Error (Counts)', fontsize=15)
plt.savefig('test.svg')
print(np.sum(abs(np.array(VecStd)-np.mean(trainout)))/np.mean(trainout))
print(np.sum(abs(np.array(VecStdv)-np.mean(trainout)))/np.mean(trainout))
print("GS/FT", scipy.stats.spearmanr(Vec,VecStd)[0])
print("No GS/FT", scipy.stats.spearmanr(Vec,VecStdv)[0])
