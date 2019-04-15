#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:50:47 2018

@author: David
"""

# File to create Bloch vector input data

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

    
def generation(dim):
    
    # Basis States
    basis = []
    for i in range(dim):
        vector = qt.basis(dim,n=i)
        state = qt.ket2dm(vector)
        basis.append(state)
    # Mixing Parameters    
    p = np.zeros(dim)
    rand = np.random.uniform(0,1)
    if rand>0.5:
        pp = 0
        for i in range(dim):
            p[i] = np.random.uniform(0,1)
        p = p/sum(p)
    else:
        pp = 1
        p[0] = 1
    # Full State
    state = 0
    for i in range(dim):
        state += p[i]*basis[i]
    
    # Random Rotation
    QQ = qt.random_objects.rand_unitary_haar(N=dim)
    # Density Matrix
    rho = QQ*state*QQ.dag()
    
    return rho,pp

def purity(rho):
    
    rr = rho*rho
    pure = rr.tr()
    
    return pure
     
def Qobj_to_array(rho):
    ' Convert density matrix from qutip object to numpy array '
    fname = 'temp.txt'
    qt.file_data_store(fname, rho, sep=' ')
    elements = qt.file_data_read(fname, sep=' ')
    
    return elements

def Bloch_vector(A,dim):
    ' Return Bloch Vector '
    ' specific to single qubit '
    w = np.real(A[0][0]-A[1][1])
    u = 2*np.real(A[0][1])
    v = 2*np.imag(A[1][0])
    
    vector = np.array([u,v,w])
    
    return vector
    
def out_str_vector(v,dim):
    'create output string'
    'works for Bloch vecotr and density matrix '
    rho_string = ''
    for i in range(len(v)):
        row =  str(v[i])
        rho_string+=' '
        rho_string+=row
    return rho_string

def train_vectors(dim,num):
    ' Create training vectors'
    'Files'
    fdata = open("Bloch_train.txt","w")
    fgroup = open("Bloch_train_group.txt","w")
    
    trace = np.zeros(num)
    ' Loop over training set size '
    for count in range(num):
        
        # Density matrix
        rho,pp = generation(dim)
        #Purity
        pure = purity(rho)
        trace[count] = pure
        # Convert to Numpy Array
        rho = Qobj_to_array(rho)
        # Bloch Vector
        vector = Bloch_vector(rho,dim)
        # Formating for output
        out_string = out_str_vector(vector,dim)
       # Assign labels
        if pp==1:
            group_string = ' 0 1'
        else:
            group_string = ' 1 0'
        
        # Write to file
        fdata.write('%s\n' %(out_string))
        fgroup.write('%s\n' %(group_string))
    
    fdata.close()
    fgroup.close()
    # Save purity values
    file = open('Purity_Train.txt','w')
    for i in range(len(trace)):
        file.write('%s\n' %str(trace[i]))
    file.close()
    print('Training Data Complete')

def test_vectors(dim,num):
    ' Create test set '
    fdata = open("Bloch_test.txt","w")
    fgroup = open("Bloch_test_group.txt","w")
    trace = np.zeros(num)
    bloch = np.zeros(num)
    
    ' Loop over test set size '
    for count in range(num):
       
        # Rho
        rho,pp = generation(dim)
        # Purity
        pure = purity(rho)
        trace[count] = pure
        # Convert to numpy array
        rho = Qobj_to_array(rho)
        # Bloch vector
        vector = Bloch_vector(rho,dim)
        # Length of Bloch Vector
        mag = np.sqrt(np.sum(vector*vector))
        bloch[count] = mag
        # Formating for output
        out_string = out_str_vector(vector,dim)
       # Labels
        if pp==1:
            group_string = ' 0 1'
        else:
            group_string = ' 1 0'
        
        # Write to file
        fdata.write('%s\n' %(out_string))
        fgroup.write('%s\n' %(group_string))
    
    fdata.close()
    fgroup.close()
    
   # Save values to files
    'Purity'
    file = open('Purity_Test.txt','w')
    for i in range(len(trace)):
        file.write('%s\n' %str(trace[i]))
    file.close()
    'Magnitude of Bloch Vector '
    file = open('Bloch_Mag_Test.txt','w')
    for i in range(len(bloch)):
        file.write('%s\n' %str(bloch[i]))
    file.close()
    
    print('Test Data Complete\n')
   


