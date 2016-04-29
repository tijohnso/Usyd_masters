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
#from sklearn.preprocessing import StandardScaler

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
#X = StandardScaler().fit_transform(training_join_df.iloc[:,3:])
#X_test = StandardScaler().fit_transform(test_data_df.iloc[:,1:])

X = training_join_df.iloc[:,3:]
X_test = test_data_df.iloc[:,1:]

#Covariance matrix
#COV3 = (X - np.mean(X, axis=0)).T.dot((X - np.mean(X, axis=0))) / (X.shape[0]-1)
#COV2 = (X).T.dot(X) / (X.shape[0]-1)
#COV = np.cov(X, rowvar=0)
#Mean_X = np.mean(X, axis=0)

#Eigen decomposition
#E_vals, E_vecs = np.linalg.eig(COV)
#Sigma = np.diag(E_vals)

#SVD
U, S, V  = np.linalg.svd(X)
#assert allclose(abs((U*E_vecs).sum(0)),1.)
#assert allclose(abs(((S**2)*E_vecs).sum(0)),1.)

# Verify if V'SV == VSV' == COV (symmetric matrix)
#E_vecs.dot(Sigma.dot(E_vecs.T))

#Analyse feature reduction / variance trade-off:
sum_evals = sum(S)
retained_variance = [(i / sum_evals)*100 for i in sorted(S, reverse=True)]
cum_retained_variance = np.cumsum(retained_variance)

#Just checking to make sure sorting doesn't change anything (since U, V should already be sorted):
sum_evals2 = sum(S)
retained_variance2 = [(i / sum_evals2)*100 for i in S]
cum_retained_variance2 = np.cumsum(retained_variance2)

#Converting to evals
evals = S**2/(len(S)-1)

sum_evals3 = sum(evals)
retained_variance3 = [(i / sum_evals3)*100 for i in evals]
cum_retained_variance3 = np.cumsum(retained_variance3)

print(int(cum_retained_variance[1000]),'%', int(cum_retained_variance[5250]),'%', int(cum_retained_variance[7000]),'%', int(cum_retained_variance[10000]),'%')
print(int(cum_retained_variance2[1000]),'%', int(cum_retained_variance2[5250]),'%', int(cum_retained_variance2[7000]),'%', int(cum_retained_variance2[10000]),'%')
print(int(cum_retained_variance3[1000]),'%', int(cum_retained_variance3[5250]),'%', int(cum_retained_variance3[7000]),'%', int(cum_retained_variance3[10000]),'%')
print(int(cum_retained_variance3[1000]),'%', int(cum_retained_variance3[2000]),'%', int(cum_retained_variance3[3000]),'%', int(cum_retained_variance3[4000]),'%')


#=============Prepare data for XGBoost==============================================================#
#Choose 5250 features giving 80% retained variance
k = 2000
V_k = U[:,:k]

#Determine reduced projection matrix for both test and train
#Xp = X.dot(V_k.T)
#X_testp = X_test.dot(V_k.T)

Xp = X.dot(V_k)
X_testp = X_test.dot(V_k)
Xp_df = pd.DataFrame(Xp)
X_testp_df = pd.DataFrame(X_testp)

#Convert labels to numeric values for input into XGBoost
train_y_f = training_join_df['Label'].astype('category').to_frame()
train_y_f = train_y_f.apply(lambda z: z.cat.codes)

#Check the categorical conversion lines up
map_cols = (training_join_df['Label'], train_y_f)
train_y_map_df = pd.concat(map_cols, axis=1)

#Convert to Array for input into XGBoost
train_X = Xp_df.values
test_X = X_testp_df.values
train_y = train_y_f.values

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

