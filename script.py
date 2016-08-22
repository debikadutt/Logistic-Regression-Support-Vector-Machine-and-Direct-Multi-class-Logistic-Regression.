import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import pickle
from pylab import *

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    w = initialWeights

    bias = np.ones((train_data.shape[0]))
    train_data = np.column_stack((train_data, bias))
    wtx = np.dot(train_data,w)
    # print wtx.shape
    wtx = np.array([wtx]).reshape(wtx.shape[0],1)
    #print "new shape",wtx.shape
    theta = sigmoid(wtx)
    #print theta.shape
    temp = theta-labeli
    #print temp.shape,"temp shape"
    #thetax = np.dot(np.transpose(train_data),temp)
    thetax = np.multiply(temp,train_data)
    #print thetax.shape,"thetax shape"
    tsum = np.sum(thetax,0)
    tsum = tsum/n_data
    #print tsum.shape,"tsum shape"
    error_grad = tsum.flatten()

    ytheta = labeli * np.log(theta)
    #print ytheta.shape,"ytheta shape"

    iytheta = (1- labeli)*np.log(1-theta)
    #print iytheta.shape,"inverse ytheta shape"

    sumtheta = (ytheta+iytheta)
    #print sumtheta.shape,"sumtheta shape"
    sumtheta = np.sum(sumtheta)
    sumtheta = (sumtheta/n_data)*-1
    error = sumtheta
    #print sumtheta,"sumtheta"

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """

    bias = np.ones((data.shape[0]))
    data = np.column_stack((data, bias))

    output = np.dot(data,W)
    output = sigmoid(output)
    label = np.argmax(output,1)
    label = label.reshape(label.shape[0],1)

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    t_data, t_label = args
    num_data = t_data.shape[0]
    bias = np.ones((num_data))
    t_data = np.column_stack((t_data, bias))
    num_feature = t_data.shape[1]

    t_label = t_label.reshape(num_data, n_class)

    W = np.array(params).astype('float').reshape(num_feature, n_class)
    t_data = np.array(t_data).astype('float')

    errGr = np.zeros(num_feature * n_class).astype('float').reshape(num_feature, n_class)
    error = 0
    for i in range(0, num_data):
        WX = np.dot(t_data[i], W)
        WX = np.exp(WX).reshape(1, WX.shape[0])
        sumWX = np.sum(WX, axis=1)
        deno = sumWX.reshape(sumWX.shape[0],1)
        thetaNK = WX / deno

        thetay = thetaNK - t_label[i].reshape(1, t_label[i].shape[0])
        data = t_data[i].reshape(t_data[i].shape[0], 1)
        a = np.dot(data,thetay)
        errGr = np.add(errGr, a)

        ytheta = t_label[i].reshape(1, t_label[i].shape[0]) * np.log(thetaNK)
        ysumtheta = np.sum(ytheta)
        error += ysumtheta

    errGr = errGr/num_data
    error = -1 * error/num_data

    error_grad = errGr.flatten()

    return error, error_grad

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    bias = np.ones((data.shape[0]))
    data = np.column_stack((data, bias))

    WX = np.array([]).astype('float')
    sumWX = np.array([]).astype('float')
    WX = np.dot(data, W)
    WX = np.exp(WX)

    label = np.argmax(WX, 1)
    label = label.reshape(label.shape[0], 1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))


with open('params.pickle', 'wb') as f1: 
    pickle.dump(W, f1)
 
# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
# Linear kernel
print('SVM with linear kernel')
clf = SVC(kernel='linear')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')


# Radial basis function with gamma = 1
print('\n\n SVM with radial basis function, gamma = 1')
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

# Radial basis function with gamma = 0
print('\n\n SVM with radial basis function, gamma = 0')
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

arr = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for c in arr:
	print('C value ' + str(c))
	clf = SVC(kernel='rbf', C=c)
	clf.fit(train_data, train_label.flatten())
	print('\n Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
	print('\n Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
	print('\n Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

with open('params_bonus.pickle', 'wb') as f2:
    pickle.dump(W_b, f2)

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
