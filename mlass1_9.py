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
from sklearn import cross_validation
from numpy import identity, zeros, e, array, asarray, log, ones, sum, shape, c_, r_, append, mean, cov, diag, linalg, cumsum, argsort, hstack, vstack, argmax, tile

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

#=============Prepare Reduced Dimension Matrices and classifier input====================================================#

#Training and Test sets as arrays
X = training_join_df.iloc[:,3:].values
X_test = test_data_df.iloc[:,1:].values

#Training labels
#Convert labels to numeric values for input into XGBoost
Y = training_join_df['Label'].astype('category').to_frame()
Y = Y.apply(lambda z: z.cat.codes).values

#Check the categorical conversion lines up
#map_cols = (training_join_df['Label'], train_y_f)
#train_y_map_df = pd.concat(map_cols, axis=1)
#print(list(train_y_f.groupby('Label'))[0])

#Cross Validation
X_cvtrain, X_cvtest, y_cvtrain, y_cvtest = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

def fold(X, y, folds, ratio): # X np array of data, y np vector of classes, no folds, ratio of test to train data
    assert X.shape[0] == len(y)
    perm = np.random.permutation(X.shape[0])
    Xp = X[perm,:]
    yp = y[perm]
    bsize = int(X.shape[0]/folds)
    train_size = int(bsize * (1 - ratio)) + 1
    result = []
    for i in range(folds):
        train_X = Xp[(i * bsize):(i * bsize) + train_size, :]
        block_end = X.shape[0] if ((i+1) * bsize - 1) > X.shape[0] else ((i+1) * bsize - 1)
        test_X = Xp[(i * bsize) + train_size:block_end, :]
        train_y = yp[(i * bsize):(i * bsize) + train_size]
        test_y = yp[(i * bsize) + train_size:block_end]
        result.append((train_X, test_X, train_y, test_y))
    return result

print('Building folds at:',time.strftime("%H:%M:%S"))
xv = fold(X, Y, 5, .2)
print('Folds built at:',time.strftime("%H:%M:%S"))
for X_train, X_test, y_train, y_test in xv:
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(y_train[:10])
#=============PCA====================================================#
#Scale Train and Test data
#X = StandardScaler().fit_transform(training_join_df.iloc[:,3:])
#X_test = StandardScaler().fit_transform(test_data_df.iloc[:,1:])

#Covariance matrix
COV = (X_cvtrain - mean(X_cvtrain, axis=0)).T.dot((X_cvtrain - mean(X_cvtrain, axis=0))) / (X_cvtrain.shape[0]-1)
#COV2 = cov(X, rowvar=0)

#Eigen decomposition
#E_vals, E_vecs = linalg.eig(COV)
#
#Sigma = diag(E_vals)
#Verify if V'SV == VSV' == COV (symmetric matrix)
#E_vecs.dot(Sigma.dot(E_vecs.T))

#SVD decomposition

U, s, V  = linalg.svd(COV)


#Analyse feature reduction / variance trade-off for Eig function

#sum_evals_eig = sum(E_vals)
#retained_variance_eig = [(i / sum_evals_eig)*100 for i in sorted(E_vals, reverse=True)]
#cum_retained_variance_eig = cumsum(retained_variance_eig)
#
##Analyse feature reduction / variance trade-off for SVD function
##Converting to evals
#evals_svd = s**2/(len(s)-1)
sum_evals_svd = sum(s)
retained_variance_svd = [(i / sum_evals_svd)*100 for i in sorted(s, reverse=True)]
cum_retained_variance_svd = cumsum(retained_variance_svd)

k_values = [250, 500, 1000, 2000, 3000, 4000, 5000]

#print('Eig retained variance:')
#[print(int(cum_retained_variance_eig[k]),'%', end="") for k in k_values]
#print('\n')
print('SVD retained variance using X:')
[print(int(cum_retained_variance_svd[k]),'%', end="") for k in k_values]
print('\n')

#Eig retained variance:
#33 %61 %78 %87 %92 %
#
#SVD retained variance using X:
#33 %62 %78 %87 %92 %
#

#=============Prepare Reduced Dimension Matrices==============================================================#
#Choose k for at least 80% retained variance

k = 2000

#Wrt SVD function
V_k = V[:,:k]

#Wrt Eig function
#E_vecs_k = E_vecs[:,[argsort(Sigma)[-k:-1]]]

#Determine reduced projection matrix for both test and train
#Xp = X.dot(V_k.T)
#X_testp = X_test.dot(V_k.T)

Xk = X_cvtrain.dot(V_k)
Xk_test = X_cvtest.dot(V_k)
#Xk_df = pd.DataFrame(Xk)
#Xk_test_df = pd.DataFrame(Xk_test)

#Rename for input into Classifier
train_X = Xk
test_X = Xk_test
train_y = y_cvtrain
test_y = y_cvtest

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

#Define sigmoid function
def sigmoid(z):
    return 1 / (1 + e**(-z))

#Calcualte the cost to be minimized -- using the sigmoid function
def cost(theta, X, y, l):
    m = X.shape[0] #Number of rows in the data
    z = X.dot(theta)
    O = (-1 / m) * (log(sigmoid(z)).T.dot(y)  +  log(1-sigmoid(z)).T.dot((1-y)))
#    print(m)
#    print(theta)
#    print(theta[1:])
#    print((theta[1:]))
    R = (l / (2*m)) * theta[1:].dot(theta[1:].T)
    cost = O + R
    return cost
    
#Calculate the gradient of the cost function
def grad(theta, X, y, l):
    m = X.shape[0] #Number of rows in the data
    z = X.dot(theta).reshape(m,1)

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

#=============Classifier Training==============================================================

#Train a classifier for each class
#Calculate the weights
def class_weights(X, y, num_cl, l):
    m = X.shape[0] #Number of rows in the data
    n = train_X.shape[1] #Number of features
    theta_initial = zeros((n+1, 1)) #Initial guess for weights
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

#Start Training....................................................
startTime = time.time()

#Call class_weights and predictions functions
W = array([])
l = .9 #Regularisation parameter lambda
#alpha = .1 #BFGS automatically determines best learning rate
num_cl = 30 #Number of Classes

W = class_weights(train_X, train_y, num_cl, l)

elapsedTime = time.time() - startTime
#End Training....................................................

#=============Testing==============================================================

#Predictions. Determine probabilities for each class across the classifiers and put them in columns 
#Generate prediction by picking the column index of highest probability for each row
def predictions(X, W):
    m = X.shape[0] #Number of rows in the data
    X = c_[ones(m), X]
    prob = sigmoid(X.dot(W.T))
    best_prob = prob.max(axis=1)
    predictions = [argmax(prob[i]) for i in range(len(prob))]
    predictions = array(predictions).reshape(len(predictions),1)
    return (prob, best_prob, predictions)

#Call prediction function
prob_test, best_prob_test, predictions_test = predictions(test_X, W)
prob_train, best_prob_train, predictions_train = predictions(train_X, W)

#=============Evaluation==============================================================

#Determine accuracy on both the test and train sets.
compare_predicted_test = array([test_y[i] == predictions_test[i] for i in range(len(test_y))])
num_correctly_predicted_test = compare_predicted_test.T.dot(ones(shape(test_y)))
test_accuracy = num_correctly_predicted_test / len(test_y)
test_accuracy 

compare_predicted_train = array([train_y[i] == predictions_train[i] for i in range(len(train_y))])
num_correctly_predicted_train = compare_predicted_train.T.dot(ones(shape(train_y)))
train_accuracy = num_correctly_predicted_train / len(train_y)
train_accuracy 

#Test lambda = [.01, .1, .2, .25, .3, .5, .9]
#lambda = .9:[ 0.58468043]: [ 0.64602375]
#lambda = .5:[0.59860731]: [0.67294659]
#lambda = .3:[ 0.60308381]: [ 0.6935895]
#lambda = .25:[ 0.60432728]: [ 0.70322701]
#lambda = .2:[0.60580378]: [0.70880378]
##lambda = .1:[0.58580448] [0.72330079]
##lambda = .01:[0.56180055] [0.77025431]


#Next steps..
#1. recall = tp / (tp + fn)
#2. precision = tp / (tp + fp)
#3. f1 = 2*tp / (2*tp + fn + fp)
#4. Plot confusion matrix

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
