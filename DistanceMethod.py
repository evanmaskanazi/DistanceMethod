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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor
def multiscale(x, scales):
    return np.hstack([x.reshape(-1, 1) / pow(3., i) for i in scales])


def encode_scalar_column(x, scales=[-1, 0, 1, 2, 3, 4, 5, 6]):
    return np.hstack([np.sin(multiscale(x, scales)), np.cos(multiscale(x, scales))])
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
st_x= StandardScaler()


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

import csv
file = open("MetallicGlass.csv")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)

df = pd.read_csv(r'C:/Users/Downloads/combine.csv') #Combined ABX3 and A2BB'X6 Perovskite Structures.
dfp0=pd.read_csv(r'PerovskiteStability.csv')
dfp=dfp0.drop("energy_above_hull (meV/atom)",1)
dfskill = pd.read_csv(r'Skillcraft.txt')
dftet0 = pd.read_csv(r'Tetuanpower.csv')
dftet=dftet0.drop("DateTime",1)
dfpark = pd.read_csv(r'parkinson.data')
dfday0 = pd.read_csv(r'day.csv')
dfday=dfday0.drop("dteday",1)
dfhour0 = pd.read_csv(r'hour.csv')
dfhour=dfhour0.drop("dteday",1)
datahydro =  pd.read_csv('yachthydro.data', header=None, delimiter=r"\s+")
dfsuperconduct = pd.read_csv(r'superconductivity.csv')
dfforest00 = pd.read_csv(r'forestfire.csv')
dfforest0=dfforest00.drop("month",1)
dfforest=dfforest0.drop("day",1)
dfenergy00 = pd.read_csv(r'energydatacomplete.txt')
dfenergy0=dfenergy00.drop("date",1)
dfenergy11=dfenergy0.drop("rv2",1)
dfenergy=dfenergy11.drop("rv1",1)
dfskillcraft = pd.read_csv(r'skillcraft.txt')
dfslice = pd.read_csv(r'slice_localization_data.csv')
dfUCICBM = pd.read_csv(r'UCICBM.txt', header=None, delimiter=r"\s+")
dftest0 = pd.read_csv(r'testEgEf.txt')
dftrain0 = pd.read_csv(r'trainEgEf.txt')
dftest=dftest0.drop("Eg",1)
dftrain=dftrain0.drop("Eg",1)
#df=dff.drop("E",1)
Xtrainshuffle = np.loadtxt("Xtrainshuffle.txt")
Xlatshuffle = np.loadtxt("Xlatshuffle.txt")

#X=df.drop("Eg",1)
X=np.array(rows).T[3:23].T.astype('float')
#X=np.array(dfp).T[7:76].T.astype('float')
#X=dfskill.drop("ActionsInPAC",1)
#X=np.array(dftet.T[0:5]).T
#X=dfpark.drop("PPE",1)
#X=np.array(dfday).T[0:14].T.astype('float')
#X=np.array(dfhour).T[0:14].T.astype('float')
#X=np.array(datahydro).T[0:6].T.astype('float')
#X=dfsuperconduct.drop("critical_temp",1)
#X=dfforest.drop("area",1)
#X=dfenergy.drop("Appliances",1)
#X=dfskillcraft.drop("ComplexAbilitiesUsed",1)
#X=dfslice.drop("reference",1)
#X=np.array(dfUCICBM).T[0:16].T.astype('float')
#X=np.array(Xtrainshuffle[0:5250])[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 18, 19]]

df['Ef'] = df['Ef'].astype('float64')
dfskill['ActionsinPAC'] = dfskill['ActionsInPAC'].astype('float64')
dfpark['PPE'] = dfpark['PPE'].astype('float64')




#y = np.asarray(df['Eg'])
y=np.array(rows).T[2].astype('float')
#y = np.asarray(dfp0['energy_above_hull (meV/atom)']).astype('float')
#y=np.asarray(dfskill['ActionsInPAC'])
#y=np.array(dftet).T[5]
#y=np.asarray(dfpark['PPE'])
#y=np.asarray(dfday['cnt']).astype('float')
#y=np.asarray(dfhour['cnt']).astype('float')
#y=np.array(datahydro).T[6].astype('float')
#y=np.asarray(dfsuperconduct['critical_temp']).astype('float')
#y=np.asarray(dfforest['area']).astype('float')
#y=np.asarray(dfenergy['Appliances']).astype('float')
#y=np.asarray(dfskillcraft['ComplexAbilitiesUsed']).astype('float')
#y=np.asarray(dfslice['reference']).astype('float')
#y=np.array(dfUCICBM).T[17]
#y=np.array(Xlatshuffle[0:5250]).T[5]
X= poly.fit_transform(X)




#for i in range(len(y)):
  #  if (y[i] <=0):
        #y[i]=0
   #     print(y[i])
print(y[0:30],'ytest')
print(y[30:60],'ytest 2')
print(y[60:90],'ytest 3')

Xf=df.drop("Ef",1)
df['Ef'] = df['Ef'].astype('float64')
yf = np.asarray(df['Ef'])
#dffinal = pd.read_excel(r'C:/Users/Downloads/final_candidates.xlsx')
#dfform = pd.read_excel(r'C:/Users/Downloads/formability_database.xlsx')
#dfstab = pd.read_excel(r'C:/Users/Downloads/stability_database.xlsx')

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
'''
from sklearn.utils import shuffle
Xinputy=np.vstack((np.array(X.T),y))
Xinputvecy=np.array(Xinputy).T
Xyvec=shuffle(np.array(Xinputvecy), random_state=1)
X_input=np.array(Xyvec.T[0:np.array(Xyvec).shape[1]-1]).T
y=np.array(Xyvec.T[np.array(Xyvec).shape[1]-1])
'''
pcaifit=pca.fit(np.array(X.T[1:60]))
X_inputpca=np.vstack((np.array(X.T[1:60]),pcaifit.components_)).T
#X_input=np.array(X.T[1:60]).T
#pcaifit=pca.fit(np.array(X).T)
#X_inputpca=np.vstack((np.array(X).T,pcaifit.components_)).T
X_input=np.array(X)
X_train1=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
X_test1=np.array(X_input[0:int(0.1*np.array(X_input).shape[0])])
y_train1=y[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
y_test1=y[0:int(0.1*np.array(X_input).shape[0])]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_train1=sc.fit_transform(X_train1)
#X_test1=sc.fit_transform(X_test1)
#X_train1=dftrain.drop("Ef",1)
#X_test1=dftest.drop("Ef",1)
#y_train1=np.asarray(dftrain['Ef']).astype('float')
#y_test1=np.asarray(dftest['Ef']).astype('float')

X_inputf=np.array(Xf.T[1:60]).T
X_train1f, X_test1f, y_train1f, y_test1f = train_test_split(X_inputf , yf, test_size=0.15, random_state=1)
from sklearn.feature_selection import SelectKBest, chi2

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs = SFS(SVR(),
          k_features=10,
          forward=False,
          floating=False,
          scoring = 'r2',
          cv = 0)
from sklearn.feature_selection import RFE
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
#sfs.fit(X_train1[:, 1:10], y_train1)
#sfs.fit(X_train1, y_train1)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
#print(sfs.k_feature_names_)
#print(np.array(sfs.k_feature_names_))
#featname=list(sfs.k_feature_names_)
#print(int(featname[0]),int(featname[1]),int(featname[2]),int(featname[3]),int(featname[4]))
'best feat 6 15 18 20 24'


steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
'SVM__epsilon':[0.001]}, cv=5)
#grid=RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
#                                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
#                                   verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)



import xgboost
from xgboost import XGBRegressor
xboostr = XGBRegressor()
param_distxg = {'n_estimators': stats.randint(50, 150),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4],
                'seed':[2]
             }

#grid=RandomizedSearchCV(estimator = xboostr,
#   param_distributions = param_distxg, n_iter = 100, cv = 10,
#   verbose=2, random_state=42, n_jobs = -1)

#grid=GridSearchCV (GradientBoostingRegressor (),{
#    'n_estimators': [2000], 'max_depth': [2], 'min_samples_split': [2], 'learning_rate': [0.1],
#    'loss': ['ls'], 'random_state':[72]}, cv=5)
#grid=GridSearchCV (KernelRidge (),{ 'alpha':[0.001],'kernel':['linear']}, cv=5)
#grid=GridSearchCV (DecisionTreeRegressor(),{
 #   'criterion':['friedman_mse'], 'random_state':[72], 'splitter':['best'], 'max_depth':[None]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
#                                             }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train1, y_train1)
svr_score = grid.score(X_train1,y_train1)
svr_score1 = grid.score(X_test1,y_test1)
y_predicted1 = grid.predict(X_test1)
#grid.fit(X_train1f, y_train1f)
#svr_scoref = grid.score(X_train1f,y_train1f)
#svr_score1f = grid.score(X_test1f,y_test1f)
#y_predicted1f = grid.predict(X_test1f)


prederror=np.zeros(len(y_predicted1))
for i in range(len(y_predicted1)):
    prederror[i]=abs(y_predicted1[i] - y_test1[i])
#stack1=np.array(np.vstack((np.array(X_test1).T,prederror,y_test1,y_predicted1)))
#prederrorf=np.zeros(len(y_predicted1f))
#for i in range(len(y_predicted1f)):
#    prederrorf[i]=abs(y_predicted1f[i] - y_test1f[i])
stack1=np.array(np.vstack((np.array(X_test1).T,prederror,y_test1,y_predicted1)))



#X_input=np.array(X.T[1:60]).T
X#_input=np.array(X)
X_train2=np.vstack((np.array(X_input[0:int(0.1*np.array(X_input).shape[0])]),np.array(X_input[int(0.2*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test2=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(0.2*np.array(X_input).shape[0])])
y_train2=np.hstack((np.array(y[0:int(0.1*np.array(X_input).shape[0])]),np.array(y[int(0.2*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test2=y[int(0.1*np.array(X_input).shape[0]):int(0.2*np.array(X_input).shape[0])]
#X_train2=sc.fit_transform(X_train2)
#X_test2=sc.fit_transform(X_test2)
X_inputf=np.array(Xf.T[1:60]).T
X_train2f, X_test2f, y_train2f, y_test2f = train_test_split(X_inputf , yf, test_size=0.15, random_state=2)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train2, y_train2)
svr_score = grid.score(X_train2,y_train2)
svr_score1 = grid.score(X_test2,y_test2)
y_predicted2 = grid.predict(X_test2)
#grid.fit(X_train2f, y_train2f)
#svr_scoref = grid.score(X_train2f,y_train2f)
#svr_score2f = grid.score(X_test2f,y_test2f)
#y_predicted2f = grid.predict(X_test2f)

prederror=np.zeros(len(y_predicted2))
for i in range(len(y_predicted2)):
    prederror[i]=abs(y_predicted2[i] - y_test2[i])
#stack2=np.array(np.vstack((np.array(X_test2).T,prederror,y_test2,y_predicted2)))
#prederrorf=np.zeros(len(y_predicted2f))
#for i in range(len(y_predicted2f)):
#    prederrorf[i]=abs(y_predicted2f[i] - y_test2f[i])
stack2=np.array(np.vstack((np.array(X_test2).T,prederror,y_test2,y_predicted2)))

#X_input=np.array(X.T[1:60]).T
#X_input=np.array(X)
X_train3=np.vstack((np.array(X_input[0:int(0.2*np.array(X_input).shape[0])]),np.array(X_input[int(0.3*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test3=np.array(X_input[int(0.2*np.array(X_input).shape[0]):int(0.3*np.array(X_input).shape[0])])
y_train3=np.hstack((np.array(y[0:int(0.2*np.array(X_input).shape[0])]),np.array(y[int(0.3*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test3=y[int(0.2*np.array(X_input).shape[0]):int(0.3*np.array(X_input).shape[0])]
#X_train3=sc.fit_transform(X_train3)
#X_test3=sc.fit_transform(X_test3)
X_inputf=np.array(Xf.T[1:60]).T
X_train3f, X_test3f, y_train3f, y_test3f = train_test_split(X_inputf , yf, test_size=0.15, random_state=3)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train3, y_train3)
svr_score = grid.score(X_train3,y_train3)
svr_score1 = grid.score(X_test3,y_test3)
y_predicted3 = grid.predict(X_test3)
#grid.fit(X_train3f, y_train3f)
#svr_score3f = grid.score(X_train3f,y_train3f)
#svr_score3f = grid.score(X_test3f,y_test3f)
#y_predicted3f = grid.predict(X_test3f)


prederror=np.zeros(len(y_predicted3))
for i in range(len(y_predicted3)):
    prederror[i]=abs(y_predicted3[i] - y_test3[i])
#stack3=np.array(np.vstack((np.array(X_test3).T,prederror,y_test3,y_predicted3)))
#prederrorf=np.zeros(len(y_predicted3f))
#for i in range(len(y_predicted3f)):
#    prederrorf[i]=abs(y_predicted3f[i] - y_test3f[i])
stack3=np.array(np.vstack((np.array(X_test3).T,prederror,y_test3,y_predicted3)))

#X_input=np.array(X.T[1:60]).T
#X_input=np.array(X)
X_train1=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
X_test1=np.array(X_input[0:int(0.1*np.array(X_input).shape[0])])
y_train1=y[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
y_test1=y[0:int(0.1*np.array(X_input).shape[0])]
X_train4=np.vstack((np.array(X_input[0:int(0.3*np.array(X_input).shape[0])]),np.array(X_input[int(0.4*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test4=np.array(X_input[int(0.3*np.array(X_input).shape[0]):int(0.4*np.array(X_input).shape[0])])
y_train4=np.hstack((np.array(y[0:int(0.3*np.array(X_input).shape[0])]),np.array(y[int(0.4*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test4=y[int(0.3*np.array(X_input).shape[0]):int(0.4*np.array(X_input).shape[0])]
#X_train4=sc.fit_transform(X_train4)
#X_test4=sc.fit_transform(X_test4)
X_inputf=np.array(Xf.T[1:60]).T
X_train4f, X_test4f, y_train4f, y_test4f = train_test_split(X_inputf , yf, test_size=0.15, random_state=4)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train4, y_train4)
svr_score = grid.score(X_train4,y_train4)
svr_score1 = grid.score(X_test4,y_test4)
y_predicted4 = grid.predict(X_test4)
#grid.fit(X_train4f, y_train4f)
#svr_scoref = grid.score(X_train4f,y_train4f)
#svr_score4f = grid.score(X_test4f,y_test4f)
#y_predicted4f = grid.predict(X_test4f)

prederror=np.zeros(len(y_predicted4))
for i in range(len(y_predicted4)):
    prederror[i]=abs(y_predicted4[i] - y_test4[i])
#stack4=np.array(np.vstack((np.array(X_test4).T,prederror,y_test4,y_predicted4)))
#prederrorf=np.zeros(len(y_predicted4f))
#for i in range(len(y_predicted4f)):
#    prederrorf[i]=abs(y_predicted4f[i] - y_test4f[i])
stack4=np.array(np.vstack((np.array(X_test4).T,prederror,y_test4,y_predicted4)))



#X_input=np.array(X.T[1:60]).T
#X_input=np.array(X)
X_train1=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
X_test1=np.array(X_input[0:int(0.1*np.array(X_input).shape[0])])
y_train1=y[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
y_test1=y[0:int(0.1*np.array(X_input).shape[0])]
X_train5=np.vstack((np.array(X_input[0:int(0.4*np.array(X_input).shape[0])]),np.array(X_input[int(0.5*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test5=np.array(X_input[int(0.4*np.array(X_input).shape[0]):int(0.5*np.array(X_input).shape[0])])
y_train5=np.hstack((np.array(y[0:int(0.4*np.array(X_input).shape[0])]),np.array(y[int(0.5*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test5=y[int(0.4*np.array(X_input).shape[0]):int(0.5*np.array(X_input).shape[0])]
#X_train5=sc.fit_transform(X_train5)
#X_test5=sc.fit_transform(X_test5)
X_inputf=np.array(Xf.T[1:60]).T
X_train5f, X_test5f, y_train5f, y_test5f = train_test_split(X_inputf , yf, test_size=0.15, random_state=5)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train5, y_train5)
svr_score = grid.score(X_train5,y_train5)
svr_score5 = grid.score(X_test5,y_test5)
y_predicted5 = grid.predict(X_test5)
#grid.fit(X_train5f, y_train5f)
#svr_scoref = grid.score(X_train5f,y_train5f)
#svr_score5f = grid.score(X_test5f,y_test5f)
#y_predicted5f = grid.predict(X_test5f)

prederror=np.zeros(len(y_predicted5))
for i in range(len(y_predicted5)):
    prederror[i]=abs(y_predicted5[i] - y_test5[i])
#stack4=np.array(np.vstack((np.array(X_test4).T,prederror,y_test4,y_predicted4)))
#prederrorf=np.zeros(len(y_predicted5f))
#for i in range(len(y_predicted5f)):
#    prederrorf[i]=abs(y_predicted5f[i] - y_test5f[i])
stack5=np.array(np.vstack((np.array(X_test5).T,prederror,y_test5,y_predicted5)))



#X_input=np.array(X.T[1:60]).T
#X_input=np.array(X)
X_train1=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
X_test1=np.array(X_input[0:int(0.1*np.array(X_input).shape[0])])
y_train1=y[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
y_test1=y[0:int(0.1*np.array(X_input).shape[0])]
X_train6=np.vstack((np.array(X_input[0:int(0.5*np.array(X_input).shape[0])]),np.array(X_input[int(0.6*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test6=np.array(X_input[int(0.5*np.array(X_input).shape[0]):int(0.6*np.array(X_input).shape[0])])
y_train6=np.hstack((np.array(y[0:int(0.5*np.array(X_input).shape[0])]),np.array(y[int(0.6*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test6=y[int(0.5*np.array(X_input).shape[0]):int(0.6*np.array(X_input).shape[0])]
#X_train6=sc.fit_transform(X_train6)
#X_test6=sc.fit_transform(X_test6)
X_inputf=np.array(Xf.T[1:60]).T
X_train6f, X_test6f, y_train6f, y_test6f = train_test_split(X_inputf , yf, test_size=0.15, random_state=6)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train6, y_train6)
svr_score = grid.score(X_train6,y_train6)
svr_score6 = grid.score(X_test6,y_test6)
y_predicted6 = grid.predict(X_test6)
#grid.fit(X_train6f, y_train6f)
#svr_scoref = grid.score(X_train6f,y_train6f)
#svr_score6f = grid.score(X_test6f,y_test6f)
#y_predicted6f = grid.predict(X_test6f)

prederror=np.zeros(len(y_predicted6))
for i in range(len(y_predicted6)):
    prederror[i]=abs(y_predicted6[i] - y_test6[i])
#stack4=np.array(np.vstack((np.array(X_test4).T,prederror,y_test4,y_predicted4)))
#prederrorf=np.zeros(len(y_predicted6f))
#for i in range(len(y_predicted6f)):
#    prederrorf[i]=abs(y_predicted6f[i] - y_test6f[i])
stack6=np.array(np.vstack((np.array(X_test6).T,prederror,y_test6,y_predicted6)))


#X_input=np.array(X.T[1:60]).T
#X_input=np.array(X)
X_train1=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
X_test1=np.array(X_input[0:int(0.1*np.array(X_input).shape[0])])
y_train1=y[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
y_test1=y[0:int(0.1*np.array(X_input).shape[0])]
X_train7=np.vstack((np.array(X_input[0:int(0.6*np.array(X_input).shape[0])]),np.array(X_input[int(0.7*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test7=np.array(X_input[int(0.6*np.array(X_input).shape[0]):int(0.7*np.array(X_input).shape[0])])
y_train7=np.hstack((np.array(y[0:int(0.6*np.array(X_input).shape[0])]),np.array(y[int(0.7*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test7=y[int(0.6*np.array(X_input).shape[0]):int(0.7*np.array(X_input).shape[0])]
#X_train7=sc.fit_transform(X_train7)
#X_test7=sc.fit_transform(X_test7)
X_inputf=np.array(Xf.T[1:60]).T
X_train7f, X_test7f, y_train7f, y_test7f = train_test_split(X_inputf , yf, test_size=0.15, random_state=7)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train7, y_train7)
svr_score = grid.score(X_train7,y_train7)
svr_score1 = grid.score(X_test7,y_test7)
y_predicted7 = grid.predict(X_test7)
#grid.fit(X_train7f, y_train7f)
#svr_scoref = grid.score(X_train7f,y_train7f)
#svr_score7f = grid.score(X_test7f,y_test7f)
#y_predicted7f = grid.predict(X_test7f)

prederror=np.zeros(len(y_predicted7))
for i in range(len(y_predicted7)):
    prederror[i]=abs(y_predicted7[i] - y_test7[i])
#stack4=np.array(np.vstack((np.array(X_test4).T,prederror,y_test4,y_predicted4)))
#prederrorf=np.zeros(len(y_predicted7f))
#for i in range(len(y_predicted7f)):
#    prederrorf[i]=abs(y_predicted7f[i] - y_test7f[i])
stack7=np.array(np.vstack((np.array(X_test7).T,prederror,y_test7,y_predicted7)))



#X_input=np.array(X.T[1:60]).T
#X_input=np.array(X)
X_train1=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
X_test1=np.array(X_input[0:int(0.1*np.array(X_input).shape[0])])
y_train1=y[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
y_test1=y[0:int(0.1*np.array(X_input).shape[0])]
X_train8=np.vstack((np.array(X_input[0:int(0.7*np.array(X_input).shape[0])]),np.array(X_input[int(0.8*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test8=np.array(X_input[int(0.7*np.array(X_input).shape[0]):int(0.8*np.array(X_input).shape[0])])
y_train8=np.hstack((np.array(y[0:int(0.7*np.array(X_input).shape[0])]),np.array(y[int(0.8*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test8=y[int(0.7*np.array(X_input).shape[0]):int(0.8*np.array(X_input).shape[0])]
#X_train8=sc.fit_transform(X_train8)
#X_test8=sc.fit_transform(X_test8)
X_inputf=np.array(Xf.T[1:60]).T
X_train8f, X_test8f, y_train8f, y_test8f = train_test_split(X_inputf , yf, test_size=0.15, random_state=8)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train8, y_train8)
svr_score = grid.score(X_train8,y_train8)
svr_score1 = grid.score(X_test8,y_test8)
y_predicted8 = grid.predict(X_test8)
#grid.fit(X_train8f, y_train8f)
#svr_scoref = grid.score(X_train8f,y_train8f)
#svr_score8f = grid.score(X_test8f,y_test8f)
#y_predicted8f = grid.predict(X_test8f)

prederror=np.zeros(len(y_predicted8))
for i in range(len(y_predicted8)):
    prederror[i]=abs(y_predicted8[i] - y_test8[i])
#stack4=np.array(np.vstack((np.array(X_test4).T,prederror,y_test4,y_predicted4)))
#prederrorf=np.zeros(len(y_predicted8f))
#for i in range(len(y_predicted8f)):
#    prederrorf[i]=abs(y_predicted8f[i] - y_test8f[i])
stack8=np.array(np.vstack((np.array(X_test8).T,prederror,y_test8,y_predicted8)))




#X_input=np.array(X.T[1:60]).T
#X_input=np.array(X)
X_train1=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
X_test1=np.array(X_input[0:int(0.1*np.array(X_input).shape[0])])
y_train1=y[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
y_test1=y[0:int(0.1*np.array(X_input).shape[0])]
X_train9=np.vstack((np.array(X_input[0:int(0.8*np.array(X_input).shape[0])]),np.array(X_input[int(0.9*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test9=np.array(X_input[int(0.8*np.array(X_input).shape[0]):int(0.9*np.array(X_input).shape[0])])
y_train9=np.hstack((np.array(y[0:int(0.8*np.array(X_input).shape[0])]),np.array(y[int(0.9*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test9=y[int(0.8*np.array(X_input).shape[0]):int(0.9*np.array(X_input).shape[0])]
#X_train9=sc.fit_transform(X_train9)
#X_test9=sc.fit_transform(X_test9)
X_inputf=np.array(Xf.T[1:60]).T
X_train9f, X_test9f, y_train9f, y_test9f = train_test_split(X_inputf , yf, test_size=0.15, random_state=9)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train9, y_train9)
svr_score = grid.score(X_train9,y_train9)
svr_score1 = grid.score(X_test9,y_test9)
y_predicted9 = grid.predict(X_test9)
#grid.fit(X_train9f, y_train9f)
#svr_scoref = grid.score(X_train9f,y_train9f)
#svr_score9f = grid.score(X_test9f,y_test9f)
#y_predicted9f = grid.predict(X_test9f)

prederror=np.zeros(len(y_predicted9))
for i in range(len(y_predicted9)):
    prederror[i]=abs(y_predicted9[i] - y_test9[i])
#stack4=np.array(np.vstack((np.array(X_test4).T,prederror,y_test4,y_predicted4)))
#prederrorf=np.zeros(len(y_predicted9f))
#for i in range(len(y_predicted9f)):
#    prederrorf[i]=abs(y_predicted9f[i] - y_test9f[i])
stack9=np.array(np.vstack((np.array(X_test9).T,prederror,y_test9,y_predicted9)))



#X_input=np.array(X.T[1:60]).T
#X_input=np.array(X)
X_train1=np.array(X_input[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
X_test1=np.array(X_input[0:int(0.1*np.array(X_input).shape[0])])
y_train1=y[int(0.1*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
y_test1=y[0:int(0.1*np.array(X_input).shape[0])]
X_train10=np.vstack((np.array(X_input[0:int(0.9*np.array(X_input).shape[0])]),np.array(X_input[int(np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
X_test10=np.array(X_input[int(0.9*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])
y_train10=np.hstack((np.array(y[0:int(0.9*np.array(X_input).shape[0])]),np.array(y[int(np.array(X_input).shape[0]):int(np.array(X_input).shape[0])])))
y_test10=y[int(0.9*np.array(X_input).shape[0]):int(np.array(X_input).shape[0])]
#X_train10=sc.fit_transform(X_train10)
#X_test10=sc.fit_transform(X_test10)
X_inputf=np.array(Xf.T[1:60]).T
X_train10f, X_test10f, y_train10f, y_test10f = train_test_split(X_inputf , yf, test_size=0.15, random_state=10)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)
#rfr = GridSearchCV (RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
 #                                            }, cv=5)
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_train10, y_train10)
svr_score = grid.score(X_train10,y_train10)
svr_score1 = grid.score(X_test10,y_test10)
y_predicted10 = grid.predict(X_test10)
#grid.fit(X_train10f, y_train10f)
#svr_scoref = grid.score(X_train10f,y_train10f)
#svr_score10f = grid.score(X_test10f,y_test10f)
#y_predicted10f = grid.predict(X_test10f)

prederror=np.zeros(len(y_predicted10))
for i in range(len(y_predicted10)):
    prederror[i]=abs(y_predicted10[i] - y_test10[i])
#stack4=np.array(np.vstack((np.array(X_test4).T,prederror,y_test4,y_predicted4)))
#prederrorf=np.zeros(len(y_predicted10f))
#for i in range(len(y_predicted10f)):
#    prederrorf[i]=abs(y_predicted10f[i] - y_test10f[i])
stack10=np.array(np.vstack((np.array(X_test10).T,prederror,y_test10,y_predicted10)))


print(y_test1,'y test 1')
#print(y_test2,'y test 2')
#print(y_test3,'y test 3')
#print(y_test4,'y test 4')
print(y_predicted1,'y pred 1')
#print(y_predicted2,'y pred 2')
#print(y_predicted3,'y pred 3')
#print(y_predicted4,'y pred 4')
#print(mean_absolute_error(y_predicted1,y_test1),mean_absolute_error(y_predicted2,y_test2),mean_absolute_error(y_predicted3,y_test3))
print(mean_absolute_error(y_predicted1,y_test1))

#trainpdf1=np.vstack((stack1.T,stack2.T,stack3.T,stack4.T))
#trainpdf1=np.vstack((stack1.T,stack2.T,stack3.T))
#trainpdf1=np.vstack((stack1.T,stack2.T))
#trainpdf1=np.vstack((stack1.T))
trainpdf1=np.vstack((stack1.T,stack2.T,stack3.T,stack4.T,stack5.T,stack6.T,stack7.T,stack8.T,stack9.T,stack10.T))
#trainpdf1=np.vstack((stack1.T,stack2.T,stack3.T,stack4.T,stack5.T,stack6.T,stack7.T,stack8.T))
#trainpdf1=stack1.T

#trainint=np.array(trainpdf1)
print(5)

#trainint=np.array(trainpdf1.T[0:59].T)
#trainint=np.array(trainpdf1.T[0:70].T)
#trainint=np.array(trainpdf1.T[0:19].T)
#trainint=np.array(trainpdf1.T[0:5].T)
#trainint=np.array(trainpdf1.T[0:21].T)
trainint=np.array(trainpdf1.T[0:int(np.array(X).shape[1])].T)
#EfEg
#trainint=np.array(trainpdf1.T[0:int(np.array(X).shape[1]-1)].T)
#trainint=np.array(trainpdf1.T[0:int(np.array(X_train1).shape[1])].T)

#trainint=np.array(trainpdf1)[:, [0,1,12,13,21,22,23,27,38,55]]

print(np.array(trainint).shape,'trainint')

#trainout=np.array(trainpdf1).T[59]
#trainout=np.array(trainpdf1).T[70]
#trainout=np.array(trainpdf1).T[19]
#trainout=np.array(trainpdf1).T[5]
#trainout=np.array(trainpdf1).T[21]
trainout=np.array(trainpdf1).T[int(np.array(X).shape[1])]
#EfEg
#trainout=np.array(trainpdf1).T[int(np.array(X).shape[1]-1)]
#trainout=np.array(trainpdf1).T[int(np.array(X_train1).shape[1])]

'''
import csv
file = open("CASP.txt")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)

X_inputN=np.array(rows).T[1:10].T

yN=np.array(rows).T[0]

X_trainNT, X_testNT, y_trainNT, y_testNT = train_test_split(X_inputN , yN, test_size=0.02, random_state=10)

steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
#grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['linear'],
#                                           'SVM__epsilon':[0.001]})
from sklearn.svm import LinearSVR
#grid=make_pipeline(StandardScaler(),LinearSVR(random_state=0, tol=1e-5))

regr = make_pipeline(StandardScaler(),LinearSVR(random_state=0, tol=1e-5))
grid = GridSearchCV(RandomForestRegressor(),{'max_depth':[None], 'random_state':[0], 'criterion':['mse']
                                             })
rfr=RandomizedSearchCV(estimator = xboostr,
 param_distributions = param_distxg, n_iter = 100, cv = 10,
verbose=2, random_state=42, n_jobs = -1)
grid.fit(X_trainNT, y_trainNT)
svr_score = grid.score(X_trainNT,y_trainNT)
svr_score1 = grid.score(X_testNT,y_testNT)
y_predictedNT = grid.predict(X_testNT)


prederror=np.zeros(len(y_predictedNT))
for i in range(len(y_predictedNT)):
    prederror[i]=abs(y_predictedNT[i].astype('float') - y_testNT[i].astype('float'))


trainint=X_testNT

trainout=prederror
'''



from sklearn.inspection import permutation_importance
#grid0= GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
#'SVM__epsilon':[0.001]}, cv=5)
grid0=RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                                   verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
#grid0=RandomizedSearchCV(estimator = xboostr,
#   param_distributions = param_distxg, n_iter = 100, cv = 10,
#   verbose=2, random_state=42, n_jobs = -1)
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



gsarr0 = np.zeros((np.array(gstest).shape[0], np.array(gstest).shape[0]))
for i in range(np.array(gsarr).shape[0]):
    for j in range(np.array(gsarr).shape[0]):
        # for i in range(500):
        #   for j in range(500):
        # gsarr[i][j] = math.exp(mean_absolute_error(np.array(gstest)[i], np.array(gstest)[j]))
        if (i != j and (np.array_equal(gstest[i], gstest[j]) == False)):
            # gsarr0[i][j] = pow(math.exp(pow(mean_absolute_error(pow(np.array(gstest)[i],1.0), pow(np.array(gstest)[j],1.0)),0.5) ),-1.0)
            gsarr0[i][j] = pow(np.linalg.norm(np.array(gstest)[i] - np.array(gstest)[j]),
                               -1.0)
            # gsarr[i][j] = pow(math.exp(pow(mean_absolute_error(pow(np.array(gstest)[i],1.0), pow(np.array(gstest)[j],1.0))
            #                            *pow((abs(scipy.stats.pearsonr(np.array(gstest)[i],np.array(gstest)[j])[0]
            #                                    +scipy.stats.spearmanr(np.array(gstest)[i], np.array(gstest)[j])[0]
            #                                 )),-1.0),0.5) ),-1.0)
        elif (i == j):
            gsarr0[i][j] = 0
        # gsarr[i][j] = (0.0)*(abs(scipy.stats.spearmanr(np.array(gstest)[i], np.array(gstest)[j])[0]))\
        #             + pow(mean_squared_error(pow(np.array(gstest)[i],1.0), pow(np.array(gstest)[j],1.0)),0.5)+\
        #              +   (1.0-r2_score(np.array(gstest)[i], np.array(gstest)[j]) )+ \
        #           pow(mean_absolute_error(np.array(gstest)[i], np.array(gstest)[j]),0.5) \
        #          +0.0*(abs(spatial.distance.cosine(np.array(gstest)[i], np.array(gstest)[j])))
    #   print(i,j,'ij')
#       gsarr[i][j] = r2_score(np.array(gstest)[i], np.array(gstest)[j])
gsarrinv = np.zeros(np.array(gsarr).shape[0])

for i in range(np.array(gsarr0).shape[0]):
    # gsarrinv[i]=1/(np.sum(gsarr,axis=1)[i])
    gsarrinv[i] = 1 / (pow(np.sum(gsarr0, axis=1)[i], 1))
    # gsarrinv[i] = 1 / (pow(np.sum(gsarr, axis=1)[i], 1))
    # gsarr[i][::-1].sort()
    # gsarrinv[i] = 1 / (np.sum(gsarr[i][0:20]))
splita = np.array_split(np.sort(gsarrinv), 10)

err1 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[0]):
        err1 = np.append(err1, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
        print(i)
err2 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[1]):
        err2 = np.append(err2, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)

err3 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[2]):
        err3 = np.append(err3, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err4 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[3]):
        err4 = np.append(err4, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err5 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[4]):
        err5 = np.append(err5, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err6 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[5]):
        err6 = np.append(err6, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err7 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[6]):
        err7 = np.append(err7, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err8 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[7]):
        err8 = np.append(err8, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err9 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[8]):
        err9 = np.append(err9, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err10 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[9]):
        err10 = np.append(err10, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                            np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                          axis=0)
Vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VecStd = [np.mean(err1.T[0]), np.mean(err2.T[0]), np.mean(err3.T[0]), np.mean(err4.T[0]), np.mean(err5.T[0]),
          np.mean(err6.T[0]), np.mean(err7.T[0]), np.mean(err8.T[0]), np.mean(err9.T[0]), np.mean(err10.T[0])]

A = trainint
#A = np.array(np.array([x for x in X_input.tolist() if x not in trainint.tolist()]))
disttest2 = np.zeros(np.array(trainint).shape[0])
for i in range(np.array(trainint).shape[0]):
    disttest2[i] = np.sum(spatial.KDTree(A).query(np.array(trainint)[i], 10)[0]) / 10

gsarrinv2 = gsarrinv
gsarrinv = disttest2
splita = np.array_split(np.sort(gsarrinv), 10)

err1 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[0]):
        err1 = np.append(err1, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
        print(i)
err2 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[1]):
        err2 = np.append(err2, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)

err3 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[2]):
        err3 = np.append(err3, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err4 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[3]):
        err4 = np.append(err4, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err5 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[4]):
        err5 = np.append(err5, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err6 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[5]):
        err6 = np.append(err6, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err7 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[6]):
        err7 = np.append(err7, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err8 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[7]):
        err8 = np.append(err8, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err9 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[8]):
        err9 = np.append(err9, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err10 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[9]):
        err10 = np.append(err10, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                            np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                          axis=0)
Vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VecStdv = [np.mean(err1.T[0]), np.mean(err2.T[0]), np.mean(err3.T[0]), np.mean(err4.T[0]), np.mean(err5.T[0]),
           np.mean(err6.T[0]), np.mean(err7.T[0]), np.mean(err8.T[0]), np.mean(err9.T[0]), np.mean(err10.T[0])]



gstest=np.array(trainint)
#gstest=gram_schmidt(np.array(trainint))
#gstest=np.array(trainint)
gsarr=np.zeros((np.array(gstest).shape[0],np.array(gstest).shape[0]))
#gsarr=np.zeros((500,500))
print(np.array(gsarr).shape[0],'int array test')
print(np.array(gstest).shape,np.array(gstest)[0],'int array gs')

print(np.array(gstest[0]))
for i in range(np.array(gstest).shape[1]):
    #gstest.T[i]=gstest.T[i]*math.exp(pow(r.importances_mean[i],0.5))
    #gstest.T[i] = gstest.T[i] * (1.0*pow(r.importances_mean[i],0.5) + 0.0)
    gstest.T[i] = gstest.T[i] * (1.0 * pow(abs(r.importances_mean[i]), 0.0) + 0.0)
print(np.array(gstest[0]))

gsarr0 = np.zeros((np.array(gstest).shape[0], np.array(gstest).shape[0]))
for i in range(np.array(gsarr).shape[0]):
    for j in range(np.array(gsarr).shape[0]):
        # for i in range(500):
        #   for j in range(500):
        # gsarr[i][j] = math.exp(mean_absolute_error(np.array(gstest)[i], np.array(gstest)[j]))
        if (i != j and (np.array_equal(gstest[i], gstest[j]) == False)):
            # gsarr0[i][j] = pow(math.exp(pow(mean_absolute_error(pow(np.array(gstest)[i],1.0), pow(np.array(gstest)[j],1.0)),0.5) ),-1.0)
            gsarr0[i][j] = pow(np.linalg.norm(np.array(gstest)[i] - np.array(gstest)[j]),
                               -1.0)
            # gsarr[i][j] = pow(math.exp(pow(mean_absolute_error(pow(np.array(gstest)[i],1.0), pow(np.array(gstest)[j],1.0))
            #                            *pow((abs(scipy.stats.pearsonr(np.array(gstest)[i],np.array(gstest)[j])[0]
            #                                    +scipy.stats.spearmanr(np.array(gstest)[i], np.array(gstest)[j])[0]
            #                                 )),-1.0),0.5) ),-1.0)
        elif (i == j):
            gsarr0[i][j] = 0
        # gsarr[i][j] = (0.0)*(abs(scipy.stats.spearmanr(np.array(gstest)[i], np.array(gstest)[j])[0]))\
        #             + pow(mean_squared_error(pow(np.array(gstest)[i],1.0), pow(np.array(gstest)[j],1.0)),0.5)+\
        #              +   (1.0-r2_score(np.array(gstest)[i], np.array(gstest)[j]) )+ \
        #           pow(mean_absolute_error(np.array(gstest)[i], np.array(gstest)[j]),0.5) \
        #          +0.0*(abs(spatial.distance.cosine(np.array(gstest)[i], np.array(gstest)[j])))
    #   print(i,j,'ij')
#       gsarr[i][j] = r2_score(np.array(gstest)[i], np.array(gstest)[j])
gsarrinv = np.zeros(np.array(gsarr).shape[0])

for i in range(np.array(gsarr0).shape[0]):
    # gsarrinv[i]=1/(np.sum(gsarr,axis=1)[i])
    gsarrinv[i] = 1 / (pow(np.sum(gsarr0, axis=1)[i], 1))
    # gsarrinv[i] = 1 / (pow(np.sum(gsarr, axis=1)[i], 1))
    # gsarr[i][::-1].sort()
    # gsarrinv[i] = 1 / (np.sum(gsarr[i][0:20]))
splita = np.array_split(np.sort(gsarrinv), 10)

err1 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[0]):
        err1 = np.append(err1, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
        print(i)
err2 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[1]):
        err2 = np.append(err2, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)

err3 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[2]):
        err3 = np.append(err3, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err4 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[3]):
        err4 = np.append(err4, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err5 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[4]):
        err5 = np.append(err5, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err6 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[5]):
        err6 = np.append(err6, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err7 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[6]):
        err7 = np.append(err7, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err8 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[7]):
        err8 = np.append(err8, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err9 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[8]):
        err9 = np.append(err9, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err10 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[9]):
        err10 = np.append(err10, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                            np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                          axis=0)
Vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VecStdN = [np.mean(err1.T[0]), np.mean(err2.T[0]), np.mean(err3.T[0]), np.mean(err4.T[0]), np.mean(err5.T[0]),
          np.mean(err6.T[0]), np.mean(err7.T[0]), np.mean(err8.T[0]), np.mean(err9.T[0]), np.mean(err10.T[0])]


A = gs(trainint)
#A = gs(np.array(np.array([x for x in X_input.tolist() if x not in trainint.tolist()])))
trainintF=trainint
for i in range(np.array(A).shape[1]):
    # gstest.T[i]=gstest.T[i]*math.exp(pow(r.importances_mean[i],0.5))
    A.T[i] = A.T[i] * (1.0 * pow(r.importances_mean[i], 0.5) + 0.0)
    trainintF.T[i] = trainintF.T[i] * (1.0 * pow(r.importances_mean[i], 0.5) + 0.0)
disttest2 = np.zeros(np.array(trainint).shape[0])
for i in range(np.array(trainint).shape[0]):
    disttest2[i] = np.sum(spatial.KDTree(A).query(gs(np.array(trainintF))[i], 10)[0]) / 10

gsarrinv2 = gsarrinv
gsarrinv = disttest2
splita = np.array_split(np.sort(gsarrinv), 10)

err1 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[0]):
        err1 = np.append(err1, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
        print(i)
err2 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[1]):
        err2 = np.append(err2, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)

err3 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[2]):
        err3 = np.append(err3, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err4 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[3]):
        err4 = np.append(err4, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err5 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[4]):
        err5 = np.append(err5, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err6 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[5]):
        err6 = np.append(err6, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err7 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[6]):
        err7 = np.append(err7, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err8 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[7]):
        err8 = np.append(err8, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err9 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[8]):
        err9 = np.append(err9, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                          np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                         axis=0)
err10 = np.empty((0, 4), float)
for i in range(len(gsarrinv)):
    if (gsarrinv[i] in splita[9]):
        err10 = np.append(err10, np.array([[trainout[i], np.array(trainpdf1).T[int(np.array(X).shape[1] + 1)][i],
                                            np.array(trainpdf1).T[int(np.array(X).shape[1] + 2)][i], gsarrinv[i]]]),
                          axis=0)
Vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VecStdGSv = [np.mean(err1.T[0]), np.mean(err2.T[0]), np.mean(err3.T[0]), np.mean(err4.T[0]), np.mean(err5.T[0]),
           np.mean(err6.T[0]), np.mean(err7.T[0]), np.mean(err8.T[0]), np.mean(err9.T[0]), np.mean(err10.T[0])]


fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)

plt.plot(Vec, VecStd, 'b', label=r'GS and Feature Importance')
plt.plot(Vec, VecStdN, 'g', label=r'No GS or Feature Importance')
plt.plot(Vec, VecStdGSv, 'r', label=r'Ten Points, GS and Feature Importance')
plt.plot(Vec, VecStdv, 'k', label=r'Ten Points, No GS or Feature Importance')
plt.ylim(0,0.18)
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
