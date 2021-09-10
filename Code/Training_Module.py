#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:12:06 2021

@author: niharika-shimona
"""

from SPD_net_model import SPDNet
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim, nn
import numpy as np
from copy import copy

def train_SPDNet(SPDnet,corr_train,A_train,idd_train,Y_data,gamma,lambda_0,lambda_1,lambda_2,num_epochs,learning_rate):
     
        """ SPDNet updates for training
        
        SPDnet: model
        
        corr_train: correlation matrices
        A_train: DTI adjacency matrices
        idd_train: DTI with scans present flag
        Y_data: scores
        
        gamma: prediction loss param
        lambda_0: 1 if fmri reconstruction is included, 0 o wise
        lambda_1: DTI weighting
        lambda_2: weight decay param
        
        num_epochs: number of epochs
        learning rate: initial lr
        
        """
        
        
        optimizer = torch.optim.AdamW(SPDnet.parameters(), lr=learning_rate,weight_decay=lambda_2)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.5, last_epoch=-1)
        
        SPDnet.train()
        
        #handling missing output data 
        mask = (Y_data>0).type(torch.FloatTensor) #mask unknown data
        

        #pre-allocate
        loss_prev = 0
        loss_total = torch.zeros(num_epochs,1)
        pred_loss_gen_prev = 0
        
        for epoch in range(num_epochs):
            
            optimizer.zero_grad()
            
            fMRI_recon_loss = 0
            DTI_recon_loss = 0
            pred_loss = 0
            loss_running = 0
                 
            indices  = np.random.permutation(corr_train.size()[0])          
            N =  len(indices)
            n_score= Y_data.size()[1]
            
            for idx in range(len(indices)):            
            
                n = idx
                
                #inputs
                corr_n = corr_train[n,:,:]
                A_n = A_train[n,:,:]
                
                #forward pass
                [A_n_pred, corr_n_pred, B_fMRI, B_DTI, y_n_pred] = SPDnet.forward(corr_n)
                
                #mask examples with no DTI scans
                mask_mat = idd_train[n]
                
                #compute losses for modalities
                fMRI_recon_loss =  fMRI_recon_loss + lambda_0*torch.norm((corr_n-corr_n_pred))**2
                DTI_recon_loss =  DTI_recon_loss + mask_mat*(lambda_1*torch.norm(A_n-A_n_pred)**2)
               
                mask_n = mask[n,:]
                
                #compute prediction losses
                y_n = (Y_data[n,:]).reshape(1,n_score)
                diff = (mask_n.mul(y_n - y_n_pred))
                pred_loss = pred_loss + (torch.norm(diff)**2)/(n_score)
                  
            #backpropagation
            loss = (fMRI_recon_loss + (gamma**2)*pred_loss +  DTI_recon_loss) /N
            loss.backward()
            
            #update weights
            optimizer.step()          
            scheduler.step()
            
            loss_total[epoch] = loss
           
            #print every 20 epochs
            if(epoch%20 == 0):
                print("Epoch: %d,  reconloss fMRI: %1.3f, recon loss DTI: %1.3f, pred loss %1.3f " % (epoch, fMRI_recon_loss, DTI_recon_loss, pred_loss))
                
            #exit criterion
            if(epoch>100 and (pred_loss < 15)):
                
                break
            
            loss_prev = loss_total[epoch-5]
            
            
        SPDnet_upd = copy(SPDnet)
        del SPDnet
        
        return SPDnet_upd, B_fMRI, B_DTI, loss_total.detach().numpy()
    
    
def train_ANN(SPDnet,corr_train,A_train,idd_train,Y_data,gamma,lambda_0,lambda_1,lambda_2,num_epochs,learning_rate):
     
        """ SPDNet updates for training
        
        SPDnet: model
        
        corr_train: correlation matrices
        A_train: DTI adjacency matrices
        idd_train: DTI with scans present flag
        Y_data: scores
        
        gamma: prediction loss param
        lambda_0: 1 if fmri reconstruction is included, 0 o wise
        lambda_1: DTI weighting
        lambda_2: weight decay param
        
        num_epochs: number of epochs
        learning rate: initial lr
        
        """
        
        #freeze weights except ANN
        SPDnet.enc1.requires_grad = False
        SPDnet.enc2.requires_grad = False
        SPDnet.dec1.requires_grad = False
        SPDnet.dec2.requires_grad = False
        SPDnet.enc3.requires_grad = False
        SPDnet.enc4.requires_grad = False
        SPDnet.dec3.requires_grad = False
        SPDnet.enc4.requires_grad = False
        
        optimizer = torch.optim.AdamW(SPDnet.parameters(), lr=learning_rate,weight_decay=lambda_2)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.5, last_epoch=-1)
        
        SPDnet.train()
        

        #handling missing output data 
        mask = (Y_data>0).type(torch.FloatTensor) #mask unknown data
        

        #pre-allocate
        loss_prev = 0
        loss_total = torch.zeros(num_epochs,1)
        pred_loss_gen_prev = 0
        
        for epoch in range(num_epochs):
            
            optimizer.zero_grad()
            
            fMRI_recon_loss = 0
            DTI_recon_loss = 0
            pred_loss = 0
            loss_running = 0
                 
            indices  = np.random.permutation(corr_train.size()[0])          
            N =  len(indices)
            n_score= Y_data.size()[1]
            
            for idx in range(len(indices)):            
            
                n = idx
                
                #inputs
                corr_n = corr_train[n,:,:]
                A_n = A_train[n,:,:]
                
                #forward pass
                [A_n_pred, corr_n_pred, B_fMRI, B_DTI, y_n_pred] = SPDnet.forward(corr_n)
                
                #mask examples with no DTI scans
                mask_mat = idd_train[n]
                
                #compute losses for modalities
                fMRI_recon_loss =  fMRI_recon_loss + lambda_0*torch.norm((corr_n-corr_n_pred))**2
                DTI_recon_loss =  DTI_recon_loss + mask_mat*(lambda_1*torch.norm(A_n-A_n_pred)**2)
               
                mask_n = mask[n,:]
                
                #compute prediction losses
                y_n = (Y_data[n,:]).reshape(1,n_score)
                diff = (mask_n.mul(y_n - y_n_pred))
                pred_loss = pred_loss + (torch.norm(diff)**2)/(n_score)
                  
            #backpropagation
            loss = (fMRI_recon_loss + (gamma**2)* pred_loss +  DTI_recon_loss) /N
            loss.backward()
            
            #update weights
            optimizer.step()          
            scheduler.step()
            
            loss_total[epoch] = loss
           
            #print every 20 epochs
            if(epoch%20 == 0):
                print("Epoch: %d,  reconloss fMRI: %1.3f, recon loss DTI: %1.3f, pred loss %1.3f " % (epoch, fMRI_recon_loss, DTI_recon_loss, pred_loss))
                
            #exit criterion
            if(epoch>100 and (pred_loss < 15)):
                
                break
            
            loss_prev = loss_total[epoch-5]
            
            
        SPDnet_upd = copy(SPDnet)
        del SPDnet
        
        return SPDnet_upd, B_fMRI, B_DTI, loss_total.detach().numpy()