#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:58:00 2018

@author: David
"""

# Generic Functions File

# Density Matrix Creation File
import Q_DM_Generate as gg
# Bloch Vector Creation File
import Q_Bloch_generate as bg


import numpy as np

' Read Data From File '
def read_data(train,vector):
    # Read input from file
    if train == True:
        if vector == True:
            f_data = 'Bloch_train.txt'
            f_group = 'Bloch_train_group.txt'
        else:
            f_data = 'density_matrix_train.txt'
            f_group = 'density_matrix_train_group.txt'
    else:
        if vector == True:
            f_data = 'Bloch_test.txt'
            f_group = 'Bloch_test_group.txt'
        else:
            f_data = 'density_matrix_test.txt'
            f_group = 'density_matrix_test_group.txt'
    
    
    data = np.loadtxt(f_data)
    group = np.loadtxt(f_group)
   
    dim = len(data[0])
    num = len(group[0])
    
    return dim,num,data,group

def create_data(vector,dim,train_num,test_num):
    print('\nCreating Data Sets ... ')
    if vector == 0:
        gg.train_matrices(dim,train_num)
        gg.test_matrices(dim,test_num)
    
    ' Bloch Vectors '
    if vector == 1:
        bg.train_vectors(dim,train_num)
        bg.test_vectors(dim,test_num)
