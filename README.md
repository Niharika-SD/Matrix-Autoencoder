# Matrix-Autoencoder
Code for Matrix Autoencoder paper in https://arxiv.org/pdf/2105.14409.pdf
![MatAE](https://github.com/Niharika-SD/Matrix-Autoencoder/blob/master/Images/Matrix_AE.PNG)

Code in Branch Master

INSTRUCTIONS:

1. Read Appropriate Dataset from Data: HCP/ASD -- modify line 55 in Main.py (ASD - num_classes =3, HCP - num_classes = 1 )
2. Change Parameters according to model (default setting) or baselines (i.e. set lambda_0/1/gamma = 0 for no fMRI/no DTI/Decoupled ANN*) + uncomment lines 95 -100 in Main.py for ANN baseline 
3. Open Terminal Run Main.py

ORGANISATION:

~/Code
  1. Main.py - runner script
  2. Train_Module.py - scripts for training models
  3. SPD_net_model.py - script for main model definition

~/Data:

   1. /Data/ASD: Data_ASD.mat - ASD dataset
     
         /Outputs
           /SPDNet
           /*Baselines*
        /Train
        /Test
   2. /Data/HCP
       Data_HCP.mat - dataset for training and testing
       Data_HCP_val.mat - dataset for validation
       
       /Outputs
          /SPDNet
         /*Baselines*
       /Train
       /Test
       /Validation


