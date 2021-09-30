# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 22:08:30 2021

@author: arash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Net, NetNoAttention, NetUserAttention
import torch.optim as optim
from utils import CleanUser, CleanData, WordEmbedding, AddGaussianNoise
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision import transforms
import random
import numpy as np






def trainRNNAct(data):
    
    N=len(data[0][0][0]) #length of exercise embedding
    #print(N)
    seq_len=len(data[0][0])
    exercise_dim=len(data[0][2][0])
    
    #print(exercise_dim)
    model = Net(input_dim=N,exercise_dim=exercise_dim,windowsize=seq_len, user_profile_dim=4)
    #model = NetNoAttention(input_dim=N,exercise_dim=exercise_dim,windowsize=seq_len)
    #model = NetUserAttention(input_dim=N,exercise_dim=exercise_dim,windowsize=seq_len)
    
    #loss_function = nn.MSELoss()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_tot=[]
    transform=transforms.Compose([AddGaussianNoise(0,0.1)])
    temp_loss=0    
    for epoch in range(30):
        # if epoch % 15 ==0:
        #     print('Epoch ',epoch, '---> Loss ', temp_loss/len(data))
       
        X=DataLoader(data,batch_size=16)
        temp_loss=0
        
        for local_batch, local_labels, user, exerc in X:    
           
            
            optimizer.zero_grad()
            #hidden= torch.zeros(model.num_layers,2,N)
            model.hn=model.initHidden()
            tar, hidden = model(local_batch, user, exerc)
            #tar, hidden = model(transform(local_batch), user, exerc)
            
            loss = loss_function(tar[:,-1,:], local_labels[:,0].long())
            temp_loss+=loss.detach().item()
            loss.backward()
            optimizer.step()
            
            
        loss_tot.append(temp_loss/len(data))   
        

        
    return model, loss_tot      


def FineTune(model, data):
    #loss_function = nn.MSELoss()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_tot=[]
    transform=transforms.Compose([AddGaussianNoise(0,0.01)])
    X=DataLoader(data,batch_size=16)
    for epoch in range(1):
        for local_batch, local_labels, user, exerc in X:
            optimizer.zero_grad()
            model.hn=model.initHidden()
            #tar, hidden = model(transform(local_batch), user, exerc)
            tar, hidden = model(local_batch, user, exerc)
            loss = loss_function(tar[:,-1,:], local_labels[:,0].long())
            loss.backward()
            optimizer.step()
    return model


def UpdateNet(model, onesampleSeq, feedback, user, exerc):
           
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_tot=[]
    model.output[3].weight.requires_grad=True
    #model.output[0].weight.requires_grad=True
    transform=transforms.Compose([AddGaussianNoise(0,0.01)])
    for epoch in range(1):
        
        optimizer.zero_grad()
        model.hn=model.initHidden()
        tar, hidden = model(onesampleSeq, user, exerc)
        loss = loss_function(tar[:,-1,:], feedback.long())
        loss.backward()
        optimizer.step()
    return model    

def UpdateNet2(model, Sample, training):
    
       
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_tot=[]
    model.output[3].weight.requires_grad=True
    #model.output[0].weight.requires_grad=True
    transform=transforms.Compose([AddGaussianNoise(0,0.01)])
    for epoch in range(2):
        training2=torch.utils.data.Subset(training, np.random.randint(0,len(training),30))
        
        X=DataLoader(torch.utils.data.ConcatDataset(([Sample],training2,[Sample])),batch_size=32)
        for local_batch, local_labels, user, exerc in X:    
            optimizer.zero_grad()
            model.hn=model.initHidden()
            tar, hidden = model(local_batch, user, exerc)
            loss = loss_function(tar[:,-1,:], local_labels[:,0].long())
            loss.backward()
            optimizer.step()
    return model    
















