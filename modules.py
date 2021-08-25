# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:36:00 2021

@author: arash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils import CleanUser, CleanData, WordEmbedding, AddGaussianNoise

class Net(nn.Module):
    def __init__(self,input_dim=9,exercise_dim=5, user_profile_dim=10, windowsize=4):
        super().__init__()
        self.windowsize=windowsize
        
        self.output= nn.Sequential(nn.Linear(10, 20), nn.Dropout(0.2), nn.ReLU(), nn.Linear(20, input_dim))
        
        self.rnn = nn.RNN(10, 10, bidirectional=False, num_layers=1,nonlinearity='relu', dropout=0.2)
        self.hn=torch.zeros((1,self.windowsize,10))
        self.transform=transforms.Compose([AddGaussianNoise()])
        
        
        self.Wx = nn.Sequential(nn.Linear(input_dim, 20), nn.ReLU())
        self.Wu= nn.Sequential(nn.Linear(user_profile_dim, 15), nn.ReLU(), nn.Dropout(0.2))
        self.W1=nn.Linear(20, 10)
        self.W2=nn.Linear(15, 10)
        self.We = nn.Sequential(nn.Linear(3*windowsize, windowsize), nn.Dropout(0.2))
        
 
    def forward(self, input_activity, user_profile, excer_profile):
        
        Hx=self.Wx(input_activity)
        Hu=self.Wu(user_profile)
        
               
        pu=F.softmax(self.W1(Hx)*self.W2(Hu),dim=2)
        psi=pu*self.W1(Hx)
        
        
               
        pexe=F.softmax(self.We(excer_profile.view(input_activity.size()[0],self.windowsize*3)),dim=1).view(input_activity.size()[0],self.windowsize,1).repeat(1,1,10)
        
                                
        output2, self.hn = self.rnn(psi*pexe, self.hn)
        output=self.output(output2)
        return output, self.hn
    
    def initHidden(self):
        return torch.zeros((1, self.windowsize, 10))
    
    

    
    
    
    
    
    