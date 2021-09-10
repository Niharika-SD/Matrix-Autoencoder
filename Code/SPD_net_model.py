#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:44:34 2021

@author: niharika-shimona
"""



# torch modules
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn



class SPDNet(torch.nn.Module):
    def __init__(self, num_classes, input_size, network_size_DTI, network_size_fMRI, hidden_size):
        super(SPDNet, self).__init__()
 
        self.network_size_fMRI = network_size_fMRI #fMRI network size
        self.network_size_DTI = network_size_DTI #DTI network size
        self.input_size = input_size #no of ROIs
        self.hidden_size = hidden_size #ANN hidden size
        self.num_classes = num_classes #number of outputs
             
        # 2DFC forward
        self.enc1 = nn.Linear(in_features=self.input_size, out_features=self.network_size_fMRI,bias=False)
        self.enc2 = nn.Linear(in_features=self.input_size, out_features=self.network_size_fMRI,bias=False)
        self.enc2.weight = torch.nn.Parameter(self.enc1.weight)
        
        # FC decode branch
        self.dec1 = nn.Linear(in_features=self.network_size_fMRI, out_features=self.input_size,bias=False)
        self.dec2 = nn.Linear(in_features=self.network_size_fMRI, out_features=self.input_size,bias=False)
        self.dec1.weight = torch.nn.Parameter(self.enc1.weight.transpose(0,1))
        self.dec2.weight = torch.nn.Parameter(self.dec1.weight)
           
        #prediction branch
        self.conv_agg = nn.Conv1d(in_channels = 1 ,out_channels = 1, kernel_size = self.input_size)
        self.pred1 = nn.Linear(in_features=self.network_size_fMRI, out_features=self.hidden_size)
        self.pred2 = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)

        #SC decode branch
        self.enc3 = nn.Linear(in_features=self.network_size_fMRI, out_features=self.network_size_DTI,bias=False)
        self.enc4 = nn.Linear(in_features=self.network_size_fMRI, out_features=self.network_size_DTI,bias=False)
        self.enc4.weight = torch.nn.Parameter(self.enc3.weight)
        
        self.dec3 = nn.Linear(in_features=self.network_size_DTI, out_features=self.input_size,bias=False)
        self.dec4 = nn.Linear(in_features=self.network_size_DTI, out_features=self.input_size,bias=False)
        self.dec4.weight = torch.nn.Parameter(self.dec3.weight)

 
    def forward(self, x):
        
        #encode fMRI
        z_n = (self.enc1(x))
        c_hidd_fMRI = self.enc2(z_n.transpose(0,1))
        disc_inp = z_n    #discriminative input
        

        #decode fMRI
        z_n = (self.dec1(c_hidd_fMRI)).transpose(0,1)
        corr_n = (self.dec2(z_n)) #predicted
           
        #decode DTI branch     
        z_n_dash = (self.enc3(c_hidd_fMRI)).transpose(0,1)
        diag_c_n_DTI = (self.enc4(z_n_dash))
       
        z_n = (self.dec3(diag_c_n_DTI)).transpose(0,1)
        A_n = self.dec4(z_n)*(torch.ones(self.input_size) - torch.eye(self.input_size))
        A_n = nn.Softmax(2)(A_n.view(1, 1, -1)).view_as(A_n) 
        
        #save bases
        B_fMRI = torch.nn.Parameter(self.dec2.weight)
        B_DTI = torch.nn.Parameter(self.dec4.weight)
        
        #Phenotypic Prediction      
        disc_inp = disc_inp.view(self.network_size_fMRI,1,-1)
        conv_out = self.conv_agg(disc_inp)
        
        out = conv_out.view(1,self.network_size_fMRI)
        out = F.leaky_relu(self.pred1(out),negative_slope=0.05)
        out = self.pred2(out)
      
        y_n = out
        
        return  A_n, corr_n, B_fMRI, B_DTI, y_n
    
        