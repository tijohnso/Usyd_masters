# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:44:27 2016

@author: tijohnso
"""

import pandas as pd
import numpy as np
import operator
import xgboost as xgb
import matplotlib.pyplot as pl
from sklearn.preprocessing import StandardScaler

#=============Import, Merge, Sort, Name ==============================================================#
test_data_df = pd.read_csv('/Users/tijohnso/Masters/ML_Code/code/input/test_data.csv', header=None)
training_data_df = pd.read_csv('/Users/tijohnso/Masters/ML_Code/code/input/training_data.csv', header=None)
training_desc_df = pd.read_csv('/Users/tijohnso/Masters/ML_Code/code/input/training_desc.csv', header=None)
training_labels_df = pd.read_csv('/Users/tijohnso/Masters/ML_Code/code/input/training_labels.csv', header=None)

training_join_df = pd.merge(left=training_desc_df, right=training_labels_df, how='left', left_on=0, right_on=0, sort=True)
training_join_df = pd.merge(left=training_join_df, right=training_data_df, how='left', left_on=0, right_on=0, sort=True)

training_join_df = training_join_df.rename(columns = {0:'App', '1_x':'Description', '1_y':'Label'})
test_data_df = test_data_df.rename(columns = {0:'App'})
training_data_df = training_data_df.rename(columns = {0:'App'})
training_labels_df = training_labels_df.rename(columns = {0:'App', 1:'Label'})
training_desc_df = training_desc_df.rename(columns = {0:'App', '1_x':'Description'})

#=============PCA====================================================#
#Scale Train and Test data
X = StandardScaler().fit_transform(training_join_df.iloc[:,3:])
X_test = StandardScaler().fit_transform(test_data_df.iloc[:,1:])

#Covariance matrix
COV = np.cov(X, rowvar=0)
#COV = (X - np.mean(X, axis=0)).T.dot((X - np.mean(X, axis=0))) / (X.shape[0]-1)

#Eigen decomposition
E_vals, E_vecs = np.linalg.eig(COV)
Sigma = np.diag(E_vals)

#SVD
#E_vecs, E_vals, V  = np.linalg.svd(X)
#assert allclose(abs((U*E_vecs).sum(0)),1.)
#assert allclose(abs(((S**2)*E_vecs).sum(0)),1.)

# Verify if V'SV == VSV' == COV (symmetric matrix)
E_vecs.dot(Sigma.dot(E_vecs.T))

#Analyse feature reduction / variance trade-off:
sum_evals = sum(E_vals)
retained_variance = [(i / sum_evals)*100 for i in sorted(E_vals, reverse=True)]
cum_retained_variance = np.cumsum(retained_variance)
print(int(cum_retained_variance[1000]),'%', int(cum_retained_variance[5250]),'%', int(cum_retained_variance[7000]),'%', int(cum_retained_variance[10000]),'%')
#print(np.argmax(E_vals), np.argmin(E_vals))

#=============Prepare data for XGBoost==============================================================#
#Choose 5250 features giving 80% retained variance
i = 5250
sorted_reduced_evecs = E_vecs[np.argsort(E_vals)[-i:]]

#Determine reduced projection matrix for both (normalised) test and train
Xp = X.dot(sorted_reduced_evecs.T)
X_testp = X_test.dot(sorted_reduced_evecs.T)
Xp_df = pd.DataFrame(Xp)
X_testp_df = pd.DataFrame(X_testp)

#Assemble Train, Test, y as dataframes
X_train_cols = (training_join_df['App'], Xp_df)
X_test_cols = (test_data_df['App'], X_testp_df)
y_train_cols = (training_join_df['Label'])
#training_join_df.loc['Desc'] -- is Desc worth adding in?

X_train_df = pd.concat(X_train_cols, axis=1)
X_test_df = pd.concat(X_test_cols, axis=1)

#Convert to Array
train_X = X_train_df.values
test_X = X_test_df.values
train_y = training_join_df['Label'].values

#=============XGBoost==============================================================
#Build Dmatrix and specify booster parameters . 
xg_train = xgb.DMatrix(train_X, label=train_y)
xg_test = xgb.DMatrix(test_X)

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = .1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 30

#Cross validation:
xgb.cv(param, xg_train, num_boost_round=200, nfold=5, metrics=('mlogloss'), obj=None,
       feval=None, maximize=False, early_stopping_rounds=3, fpreproc=None, as_pandas=True,
       show_progress=True, show_stdv=False, seed=0)
#
##Training:
##xgboost.train(params, dtrain, num_boost_round=10, evals=(), obj=None, 
##              feval=None, maximize=False, early_stopping_rounds=None, 
##              evals_result=None, verbose_eval=True, learning_rates=None, 
##              xgb_model=None)
#
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
evals_result = {}
num_round = 10
bst = xgb.train(param,xg_train, num_round, evals_result=evals_result)
predictions = bst.predict(xg_test)

