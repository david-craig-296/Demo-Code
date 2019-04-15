#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:31:34 2018

@author: David
"""

# Main file of a demo code for mixed and pure classification.
# Produces a histogram plot of classification.
# Option to use Bloch vecotr and density matrix inputs
# Choice of dimension of qudit

# Generic functions file
import NN_functions_file as fn


import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

import time


''' Network '''

def Network(vector,train_num,test_num):
    ''' START OF MAIN '''
        
    # Dimension of Input Vector Plus Read Data
    train=1
    dim,num,data_train,group_train = fn.read_data(train,vector)
    train=0
    dim,num,data_test,group_test = fn.read_data(train,vector)
    
    purity = np.loadtxt('Purity_Test.txt')
    mag = purity
    

    'Test Set Size'
    size_test = test_num
    
    '''...............................TRAINING...............................'''
    
    print('Training')
    start_time = time.time()
    ' Define network '
    clf = MLPClassifier(solver='sgd', alpha=1e-2, activation = 'relu',
                        hidden_layer_sizes=(32), early_stopping=False,
                        max_iter = 400,learning_rate_init=0.5,
                        tol = 1e-3)
    ' Train network '
    clf.fit(data_train,group_train)
    print('Training Complete')
    train_time = time.time() - start_time
    print('Training Time:\t%s sec' % train_time)
    
    '''..............................Testing.................................'''
    ' Test network: x is network output '
    x = clf.predict(data_test)
    ' Count correct, incorrect classification '
    correct = 0
    pure = []
    mixed = []
    cor = []
    inc = []
    for i in range(size_test):
        if x[i][0] == group_test[i][0]:
            correct+=1
            cor.append(mag[i])
        else:
            inc.append(mag[i])
            
        if x[i][0] == 0:
            pure.append(mag[i])
        if x[i][0] == 1:
            mixed.append(mag[i])
    
    ' Plot a histogram of classifiaction '
    h, bins = np.histogram(mag,64)
    
    plt.figure(figsize=(8,8))
    plt.title('Classification: $d=4$',fontsize=24)
    plt.hist(cor,bins,color='b',label='Correct')
    plt.hist(inc,bins,color='r',label='Incorrect')
    plt.xlabel('Purity',fontsize=22)
    plt.ylabel('Number',fontsize=22)
    plt.ylim(0, test_num/20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig('Purtiy_Classification.pdf', format = 'pdf',
            bbox_inches = 'tight', dpi=300)
    plt.show()
         
    ' Determine calssificaiton accuracy '
    
    accuracy = 100*correct/test_num
    print('\nAccurracy:\t%f %%' %accuracy)
    
' ########################################################## ' 
' ########################################################## ' 
' ########################################################## ' 
   

''' Run Network '''

# Select input type as Bloch Vector or Density Matrix
# Choose to create new data or not

' Create Data ? '
create = 1
'Bloch Vector ?'
vector = 0       # 1 => Bloch vector, 0 => density matrix


# Select dimension of qudit to be generated
' Dimension of Qudit '
dim = 2

# Select training and test set size 
' Training Set Size ' 
train_num = 10000  # default 10000
' Test Set Size '
test_num = 2000    # default 2000

# Create new data
if create == True:
   fn.create_data(vector,dim,train_num,test_num)
   
# Run network function
Network(vector,train_num,test_num)


