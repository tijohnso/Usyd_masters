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
import time
#from sklearn.preprocessing import StandardScaler
from decimal import Decimal as de
from scipy.optimize import fmin_l_bfgs_b, minimize, fmin_bfgs
import sklearn as skl
from numpy import zeros, e, array, asarray, log, ones, sum, shape, c_, r_, append, mean, cov, diag, linalg, cumsum, argsort, hstack, vstack, argmax, tile

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

X = training_join_df.iloc[:,3:].values
X_test = test_data_df.iloc[:,1:].values

#Covariance matrix
COV = (X - mean(X, axis=0)).T.dot((X - mean(X, axis=0))) / (X.shape[0]-1)
#COV2 = cov(X, rowvar=0)

#Eigen decomposition
#E_vals, E_vecs = linalg.eig(COV)
#
#Sigma = diag(E_vals)
#Verify if V'SV == VSV' == COV (symmetric matrix)
#E_vecs.dot(Sigma.dot(E_vecs.T))

#SVD decomposition
#U2, s2, V2  = linalg.svd(X)
U, s, V  = linalg.svd(COV)
#assert allclose(abs((U*E_vecs).sum(0)),1.)
#assert allclose(abs(((S**2)*E_vecs).sum(0)),1.)

#Analyse feature reduction / variance trade-off for Eig function
#sum_evals_eig = sum(E_vals)
#retained_variance_eig = [(i / sum_evals_eig)*100 for i in sorted(E_vals, reverse=True)]
#cum_retained_variance_eig = cumsum(retained_variance_eig)
#
##Analyse feature reduction / variance trade-off for SVD function
##Converting to evals
evals_svd = s**2/(len(s)-1)
sum_evals_svd = sum(evals_svd)
retained_variance_svd = [(i / sum_evals_svd)*100 for i in sorted(evals_svd, reverse=True)]
cum_retained_variance_svd = cumsum(retained_variance_svd)

#evals_svd2 = s2**2/(len(s2)-1)
#sum_evals_svd2 = sum(evals_svd2)
#retained_variance_svd2 = [(i / sum_evals_svd2)*100 for i in sorted(evals_svd2, reverse=True)]
#cum_retained_variance_svd2 = cumsum(retained_variance_svd2)

k_values = [250, 500, 1000, 2000, 3000, 4000]

#print('Eig retained variance:')
#[print(int(cum_retained_variance_eig[k]),'%', end="") for k in k_values]
#print('\n')
print('SVD retained variance using X:')
[print(int(cum_retained_variance_svd[k]),'%', end="") for k in k_values]
print('\n')
#print('SVD retained variance using COV:')
#[print(int(cum_retained_variance_svd2[k]),'%', end="") for k in k_values]
#print('\n')

#Eig retained variance:
#33 %61 %78 %87 %92 %
#
#SVD retained variance using X:
#33 %62 %78 %87 %92 %
#
#SVD retained variance using COV:
#81 %95 %98 %99 %99 %

#=============Prepare Reduced Dimension Matrices==============================================================#
#Choose 5250 features giving 80% retained variance

k = 250

#Wrt SVD function
U_k = U[:,:k]

#Wrt Eig function
#E_vecs_k = E_vecs[:,[argsort(Sigma)[-k:-1]]]

#Determine reduced projection matrix for both test and train
#Xp = X.dot(V_k.T)
#X_testp = X_test.dot(V_k.T)

Xp = X.dot(U_k)
X_testp = X_test.dot(U_k)
Xp_df = pd.DataFrame(Xp)
X_testp_df = pd.DataFrame(X_testp)

#Convert labels to numeric values for input into XGBoost
train_y_f = training_join_df['Label'].astype('category').to_frame()
train_y_f = train_y_f.apply(lambda z: z.cat.codes)

#Check the categorical conversion lines up
map_cols = (training_join_df['Label'], train_y_f)
train_y_map_df = pd.concat(map_cols, axis=1)
print(list(train_y_f.groupby('Label'))[0])

#Convert to Array for input into Classifier
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
param['eval_metric'] = ['merror', 'mlogloss']
#param['eval_metric'] = ['auc','mlogloss', 'merror', 'rmse']
#Multiclass classification error rate. Calculated as (wrong cases) / (all cases).
#(For the predictions, the evaluation will regard the instances with prediction value larger than
#0.5 as positive instances, and the others as negative instances.)

#Cross validation:
xgb.cv(param, xg_train, num_boost_round=40, nfold=5, obj=None,
       feval=None, maximize=False, fpreproc=None, as_pandas=True,
       show_progress=True, show_stdv=False, seed=0)
#
##Training:
##xgboost.train(params, dtrain, num_boost_round=10, evals=(), obj=None, 
##              feval=None, maximize=False, early_stopping_rounds=None, 
##              evals_result=None, verbose_eval=True, learning_rates=None, 
##              xgb_model=None)
#
evallist  = [(dtest,'eval'), (dtrain,'train')]

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
evals_result = {}
num_round = 10
bst = xgb.train(param,xg_train, num_round, evals_result=evals_result)
pred = bst.predict(xg_test)

print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))

xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=2)

#=============Logistic Regression==============================================================

#Regularisation parameter lambda
l = .2 #Lambda
#alpha = .1 #BFGS automatically determines best learning rate
num_cl = 30 #Number of Classes
m = train_X.shape[0] #Number of rows in the data
n = train_X.shape[1] #Number of features
theta_initial = zeros((n+1, 1)) #Initial guess for weights


#Define sigmoid function
def sigmoid(z):
    return 1 / (1 + e**(-z))

#Calcualte the cost to be minimized -- using the sigmoid function
def cost(theta, X, y, l):
    z = X.dot(theta)
    O = (-1 / m) * (log(sigmoid(z)).T.dot(y)  +  log(1-sigmoid(z)).T.dot((1-y)))
    R = (l / (2*m)) * theta[1:,0].dot(theta[1:].T)
    cost = O + R
    return cost
    
#Calculate the gradient of the cost function
def grad(theta, X, y, l):
    z = X.dot(theta).reshape(20104,1)

#    print('ga', shape((1 / m) * (sigmoid(z) - y).T.dot(X)))
#    print('gb', shape(r_[0, (l / m) * theta[1:].T]))
#    print('g1', shape(theta), shape(X))
#    print('g21', shape(sigmoid(z)))
#    print('g3', shape(l / m * theta[1:]))
#    print('g4', shape(r_[0, l / m * theta[1:]]))
#    print('g5', shape(sigmoid(z) - y))
#    print('g50', shape(y))
#    print('g6', shape(1 / m * (sigmoid(z) - y).T.dot(X)))
#    print('g7', shape(grad))
    
    grad = (1 / m) * (sigmoid(z) - y).T.dot(X) + r_[0, (l / m) * theta[1:].T]

    return grad.flatten()

#Calculate the parameters of each feature for each of the 30 classes
def class_weights(X, y, num_cl, l):
    X = c_[ones(m), X]
    result_list = []
    for c in range(num_cl):
        y2 = asarray(y == c)
#        result = fmin_l_bfgs_b(cost, x0=theta_initial, args=(X, y2, l), fprime=grad)
        xopt = fmin_bfgs(cost, x0=theta_initial, args=(X, y2, l), fprime=grad)
#        result = minimize(cost, x0=theta_initial, args=(X, y2, l), method='L-BFGS-B', options={'disp': True})
#        result = minimize(cost, x0=theta_initial, args=(X, y2, l), fprime=grad, method='L-BFGS-B', options={'disp': True})        
        result_list.append(xopt)
        print('Class:',c)
        print('Results',xopt)
#        print('Successful return:',result.success)
#        print('Number of iterations:',result.nit)
#        print('Number of evaluations of objective function:',result.nfev)
#        print('Termination cause:',result.message)
    W = vstack(result_list)
    return W     

def predictions(X, W):
    X = c_[ones(m), X]
    prob = sigmoid(X.dot(W.T))
    pred = prob.max(axis=1)
    return (pred, prob)


#Mainline....................................................
W = array([])
startTime = time.time()

#Call class_weights and predictions functions
W = class_weights(train_X, train_y, num_cl, l)
pred, prob = predictions(train_X, W)

elapsedTime = time.time() - startTime
#Mainline....................................................

#Predictions
for i in shape(prob[:,0]):
    predictions = [argmax(prob[j]) for j in range(len(prob))]
predictions = array(predictions).reshape(20104,1)


accuracy = predictions / train_y

#%matplotlib inline
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#
#print('Confusion matrix ({}):\n'.format(key))
#_ = plt.matshow(confusion_matrix(Y_test, logreg.predict(X_test)), cmap=plt.cm.binary, interpolation='nearest')
#_ = plt.colorbar()
#_ = plt.ylabel('true label')
#_ = plt.xlabel('predicted label')
#print('Classification report ({}):\n'.format(key))
#print(classification_report(Y_test, tree.predict(X_test)))
#skl.confusion_matrix(y_ttrain, pred, labels=None)
