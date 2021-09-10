#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:06:59 2021

@author: niharika-shimona
"""

from SPD_net_model import SPDNet
from Training_Module import train_SPDNet,train_ANN

import sys
import numpy as np
import scipy
import os
from sklearn import preprocessing
import pickle

# torch
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from scipy import optimize
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


import time


if __name__ == '__main__':    
    
    #%reset
    
    lr = 0.005 #learning rate
    hidden_size = 60 #ANN size
    num_classes = 1 #HCP, 3 for KKI

    gamma  = 3  #pred 
    lambda_0 = 1 #fMRI
    lambda_1 = 1000 #DTI
    lambda_2 = 0.0005 #weight decay 
   
    num_epochs = 400 #max epochs
    
    #compression
    network_size_fMRI = 15
    network_size_DTI =  15

    #read data
    path_name = '/home/niharika-shimona/Documents/Projects/Autism_Network/Sparse-Connectivity-Patterns-fMRI/AE_sp_conn/Matrix_AE/Data/HCP/'
    dir_name = path_name + '/Outputs/SPDNet_Model/'

    if not os.path.exists(dir_name):
            os.makedirs(dir_name)
                        
    data = sio.loadmat(path_name + 'Data_ASD.mat')
    corr_train = torch.from_numpy(data['corr_train'][:][:]).float() 
    A_train = torch.from_numpy(data['A_train'][:][:]).float()
        
    Y_train = torch.from_numpy(data['Y_train'][:][:]).float()
    Y_test = torch.from_numpy(data['Y_test'][:][:]).float()
        
    idd_train = torch.from_numpy(data['idd_train'])
    idd_test = torch.from_numpy(data['idd_test'])
        
    #initialise model
    input_size = corr_train.size()[1]
    
    corr_mean = torch.mean(corr_train,0)
    [D,V] = torch.symeig(corr_mean,eigenvectors=True)     
    B_init_fMRI = V[:,input_size-network_size_fMRI:] 
    
    A_mean = torch.mean(A_train,0)
    [D,V] = torch.symeig(A_mean,eigenvectors=True)     
    B_init_DTI = V[:,input_size-network_size_DTI:] 
    
     
    model_init = SPDNet(num_classes, input_size, network_size_DTI, network_size_fMRI, hidden_size)
    
    model_init.enc1.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    model_init.enc2.weight = torch.nn.Parameter(B_init_fMRI.transpose(0,1))
    
    model_init.dec3.weight = torch.nn.Parameter(B_init_DTI)
    model_init.dec4.weight =  torch.nn.Parameter(model_init.dec3.weight)

    #train model
    
    [model_gd, B_gd_fMRI, B_gd_DTI, err_out, ] = train_SPDNet(model_init,corr_train,A_train,idd_train,Y_train,gamma,lambda_0,lambda_1,lambda_2,num_epochs,lr)
             
    #uncomment for ANN run
    # print('Training ANN')
    # gamma  = 1  #pred 
    # lambda_0 = 0 #fMRI
    # lambda_1 = 0 #DTI
    # [model_gd, B_gd_fMRI, B_gd_DTI, err_out, ] = train_ANN(model_gd,corr_train,A_train,idd_train,Y_train,gamma,lambda_0,lambda_1,lambda_2,num_epochs,lr)    

    #visualise
    B_gd_norm_fMRI = preprocessing.normalize(B_gd_fMRI.detach().numpy(),'l2',axis=0)
    B_gd_norm_DTI = preprocessing.normalize(B_gd_DTI.detach().numpy(),'l2',axis=0)
       

    font = {'family' : 'normal',
                'size'   : 12}
    matplotlib.rc('font', **font)
    
    corr_test = torch.from_numpy(data['corr_test'][:][:]).float()
    A_test = torch.from_numpy(data['A_test'][:][:]).float() 
    
    #allocate and save
    corr_rec_test = np.zeros(corr_test.size())
    A_rec_test = np.zeros(A_test.size())
    corr_rec_train = np.zeros(corr_train.size())
    A_rec_train = np.zeros(A_train.size())
    Y_pred_train = []
    Y_pred_test = []
    
    for pat_no in range(corr_train.size()[0]):

            corr_n = corr_train[pat_no,:,:]
            [A_n_pred, corr_n_pred, B_fMRI, B_DTI, y_n_pred] = model_gd.forward(corr_n)
            
            corr_rec_train[pat_no,:,:] = corr_n_pred.detach().numpy()
            A_rec_train[pat_no,:,:] = A_n_pred.detach().numpy()
            Y_pred_train.append(y_n_pred)
           
           
    for pat_no in range(corr_test.size()[0]):

            corr_n = corr_test[pat_no,:,:]
            [A_n_pred, corr_n_pred, B_fMRI, B_DTI, y_n_pred] = model_gd.forward(corr_n)
            
            corr_rec_test[pat_no,:,:] = corr_n_pred.detach().numpy()
            A_rec_test[pat_no,:,:] = A_n_pred.detach().numpy()
            Y_pred_test.append(y_n_pred)
            
    Y_pred_train = np.asarray(torch.stack(Y_pred_train).detach().numpy()).reshape(np.shape(Y_train))
    Y_pred_test = np.asarray(torch.stack(Y_pred_test).detach().numpy()).reshape(np.shape(Y_test))  
       
    #mask unknowns
    Y_pred_train[Y_train==0] = 0
    Y_pred_test[Y_test==0] = 0
        
    fig,ax = plt.subplots()
    iter = len(err_out)
    ax.plot(list(range(iter)),err_out[0:iter:1],'r')
    plt.title('Loss',fontsize=16)
    plt.ylabel('Error' ,fontsize=12)
    plt.xlabel('num of iterations',fontsize=12)
    plt.show()
    figname = dir_name +'Loss.png'
    fig.savefig(figname)   # save the figure to file
    plt.close(fig)
    
    dict_save ={'B_gd_fMRI':B_gd_fMRI, 'B_gd_DTI':B_gd_DTI,
                    'model_gd':model_gd,
                    'Y_train':Y_train.detach().numpy(), 'Y_train_obt':Y_pred_train,
                    'Y_test':Y_test.detach().numpy(),'Y_test_obt':Y_pred_test,
                    }
                    
    filename = dir_name + 'output.p'
       
    dict_save_mat ={'B_gd_fMRI':B_gd_fMRI.detach().numpy(),'B_gd_DTI':B_gd_DTI.detach().numpy(),
                        'corr_rec_test': corr_rec_test, 'A_rec_test': A_rec_test,
                        'corr_rec_train': corr_rec_train, 'A_rec_train': A_rec_train,
                        'Y_train':Y_train.detach().numpy(), 'Y_train_obt':Y_pred_train,
                        'Y_test':Y_test.detach().numpy(),'Y_test_obt':Y_pred_test}
                    
                                        
    pickle.dump(dict_save, open(filename, "wb"),protocol=2)
    sio.savemat(dir_name + 'output.mat',dict_save_mat)
