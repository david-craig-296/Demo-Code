#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:33:21 2018

@author: David
"""

# File to create density matrix input data

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
        p[0] = 1
        pp = 1
        
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
    ' return purity of state '
    rr = rho*rho
    pure = rr.tr()
    
    return pure

def Qobj_to_array(rho):
    ' Convert density matrix from qutip object to numpy array '
    fname = 'temp.txt'
    qt.file_data_store(fname, rho, sep=' ')
    elements = qt.file_data_read(fname, sep=' ')
    
    return elements

def real_imaginary_split(A):
    ''' 
    Outputs array in the form:
        
        C = [[real(A[0][0]),imaginary(A[0][0])]
             [real(A[0][1]),imaginary(A[0][1])]
             ...
             [real(A[dim][dim]),imaginary(A[dim][dim])]]'''
    
    dim = len(A[0])
    
    c_dim = dim*dim
    C = np.zeros((c_dim,2))
    
    for i in range(dim):
        for j in range(dim):
            k = i*dim + j
            #print(i,j,k)
            C[k][0] = np.real(A[i][j])
            C[k][1] = np.imag(A[i][j])
    
    return C

def output_string(rho,dim):
    rho_string = ''
    for i in range(dim*dim):
        row =  ' '.join(str(x) for x in rho[i])
        rho_string+=' '
        rho_string+=row
    return rho_string

def train_matrices(dim,num):
    
    fdata = open("density_matrix_train.txt","w")
    fgroup = open("density_matrix_train_group.txt","w")
    
    ppp = np.zeros(num)
    
    for count in range(num):
        # Density Matrix
        rho,pp = generation(dim)
        # Purity
        pure = purity(rho)
        ppp[count] = pure
        # Array of real and imaginary parts
        elements = Qobj_to_array(rho)
        output = real_imaginary_split(elements)
      
        # Formating for output
        rho_string = output_string(output,dim)
        
        if pp==1:
            group_string = ' 0 1'
        else:
            group_string = ' 1 0'
        
        # Write to file
        fgroup.write('%s\n' %(group_string))
        fdata.write('%s\n' %(rho_string))
    
    fgroup.close()
    fdata.close()
    
    'Purity'
    file = open('Purity_Train.txt','w')
    for i in range(len(ppp)):
        file.write('%s\n' %str(ppp[i]))
    file.close()
    
    print('Training Data Complete')

def test_matrices(dim,num):
        
    fdata = open("density_matrix_test.txt","w")
    fgroup = open("density_matrix_test_group.txt","w")
    trace = np.zeros(num)
    ' Loop over test set '
    for count in range(num):
        # Density MAtrix
        rho,pp = generation(dim)
        # Purity
        pure = purity(rho)
        trace[count] = pure
               
        # Array of real and imaginary parts
        elements = Qobj_to_array(rho)
        output = real_imaginary_split(elements)
        
        # Formating for output
        rho_string = output_string(output,dim)
        
        if pp==1:
            group_string = ' 0 1'
        else:
            group_string = ' 1 0'

        # Write to file
        fgroup.write('%s\n' %(group_string))
        fdata.write('%s\n' %(rho_string))
    
    fgroup.close()
    fdata.close()
    # Save values to file
    'Purity'
    file = open('Purity_Test.txt','w')
    for i in range(len(trace)):
        file.write('%s\n' %str(trace[i]))
    file.close()
    
    '''
    plt.figure(figsize=(7,6))
    plt.title('Single Qubit: Tr[rho^2]', fontsize=22)
    #plt.xlabel('Data Number', fontsize = 20)
    plt.ylabel('Tr[rho^2]',fontsize=20)
    plt.tick_params(labelsize=16)
    plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False) 
    plt.plot(range(num),trace,'b.')
    plt.savefig('random_dm_generation.png', format='png', dpi=200)
    plt.show()
    '''
    
    print('Test Data Complete')


    
