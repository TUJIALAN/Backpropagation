# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 01:20:30 2019

@author: tujialan
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_linear(n=100):
    """generate data"""
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1) 
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    """generate data"""
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_results(x, y, pred_y,filename):
    """visualize true target and predit target"""
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.savefig(filename)
    
#sigmoid
def sigmoid(x):
    return 1.0/(1.0 +np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x,1.0-x)

def forword(x,w,b):
    a = [x.reshape(x.shape[0],1)]
    z = [np.add(np.dot(w[0],a[0]),b[0])]
    for l in range(1,3):
        a.append(sigmoid(z[l-1]))
        #print(a[1].shape)
        z.append(np.add(np.dot(w[l],a[l]),b[l]))
    y_pred = sigmoid(z[2])
    return a, z, y_pred

def back(a,z,y,y_pred,w):
    d = [0,0,0]
    
    d[2] = np.multiply(derivative_sigmoid(y_pred),(y_pred.T[0]-y).reshape(y_pred.shape))

    for l in reversed(range(2)):
        d[l] = np.multiply(derivative_sigmoid(a[l+1]), np.dot(np.transpose(w[l+1]), d[l+1]))
    
    return d

def optimize(d,lr,w,a,b): 
    dw = [0,0,0]
    db = [0,0,0]
    for l in range(3):
        #print(d[l].shape)
        #print(a[l].shape)
        dw[l] = np.dot(d[l],a[l].transpose())
        db[l] = d[l]
        w[l] -= lr * dw[l]
        b[l] -= lr * db[l]
    return w,b


def initial_w_b():
    w = [0,0,0]
    b = [0,0,0]
    #np.random.seed(5)
    w[0] = np.random.randn(4,2)
    w[1] = np.random.randn(4,4)
    w[2] = np.random.randn(1,4)
    
    b[0] = np.random.randn(4,1)
    b[1] = np.random.randn(4,1)
    b[2] = np.random.randn(1,1)
    #print("w",w,"b",b)
    return w,b


##################################3



def lab1(X,Y,iteration,filename,lr = 0.05):
    w,b = initial_w_b()
    for i in range(iteration):
        Y_pred = np.zeros((X.shape[0],1))
        for j in range(X.shape[0]):
            x = X[j]
            y = Y[j]
            a, z, y_pred = forword(x,w,b)
            Y_pred[j] = y_pred
            d = back(a,z,y,y_pred,w)
            w,b = optimize(d,lr,w,a,b) 
        Y_pred1 = np.where(Y_pred<0.5,0,1)
        accuracy = (len(Y)-np.sum(abs(Y - Y_pred1)))/len(Y)
        #print(accuracy)
        if accuracy == 1:
            print("accuracy = 1 in this iteration",i)
            break
        if i % 1000 == 0:
            Y_pred = Y_pred.reshape(Y.shape)
            print('epoch', i, 'loss :', 1/2*np.sum(np.square(Y-Y_pred)))

    print(Y_pred)
    Y_pred1 = np.where(Y_pred<0.5,0,1)
    #print(Y_pred1)
    show_results(X, Y, np.where(Y_pred<0.5,0,1),filename)   


X,Y = generate_XOR_easy()   
lab1(X,Y,2000000,'XOR',lr = 0.05)
print('XOR finish')
X,Y = generate_linear(n=100)
lab1(X,Y,2000000,'Linear',lr = 0.05)
print('linear finish')
        



