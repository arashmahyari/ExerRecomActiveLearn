# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:59:14 2021

@author: arash
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import torch, pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils import CleanData, CleanUser, WordEmbedding, DataPreprationWindow, Recall, CleanDataSorted
from modules import Net
from training_net import trainRNNAct, UpdateNet, FineTune, UpdateNet2
from association_rule import RuleMining
from utils import DistanceExcer, IdentifySimilarUsers, IdentifyUsersHistory, ExerLookup
from my_own_embedding import ActEmbed
import torch.nn.functional as F
from scipy.stats import entropy, beta 
from augmentation import AugmentDic, Augment, AugmentDicRule


#user=CleanUser('user_profile.csv')
user=CleanUser('user_demography.csv')

exercise=CleanUser('exercise_profile.csv')

by_user=CleanData('data.csv')

#Augmentation using Association Rule Mining
#rules=RuleMining('data.csv')
#exerDict=AugmentDicRule(rules[0])

#Augmentation using Expert input
exerDict=AugmentDic()


wordmodel=ExerLookup(list(exercise['name']))


windowsize=3
top1=[]
top5=[]
top10=[]
top20=[]


top1_before=[]
top5_before=[]
top10_before=[]
top20_before=[]



sim_values=[]
cos_values=[]


distribution=[]
out_dist=[]
out_dist_before=[]


for _ in range(10):
    
    R=Recall(wordmodel) #calculate accuracy after active learning
    
    R_before=Recall(wordmodel) #calculate accuracy before active learning

    for k_fold in list(by_user.keys())[1:]:
        by_user2=dict(by_user)
        distribution.append([])
        sim_valuestemp=[]
        
        test=by_user2[k_fold]
        del by_user2[k_fold]
        
        #by_user2, user=Augment(by_user2, exerDict, user) #enable to have augmentation
        
        X=DataPreprationWindow(by_user2,wordmodel,user, exercise, windowsize=windowsize)
        net, loss_tot=trainRNNAct(X)
        
        out_dist_temp=[]
        Xtest=DataPreprationWindow({k_fold:test},wordmodel,user, exercise, windowsize=windowsize)
        
        """ finetune the network with user profile """
        finetuningusers=IdentifySimilarUsers(user, k_fold)
        findata=dict()
        for x in finetuningusers:
            findata[x]=by_user2[x]
        Xfine=DataPreprationWindow(findata,wordmodel,user, exercise)
        net=FineTune(net, Xfine)
        
        for param in net.parameters():
            param.requires_grad = False
        
        
        for j in range(len(Xtest)):
            a, b=net(Xtest[j][0].view(-1,windowsize,48),Xtest[j][2].view(-1,windowsize,4),Xtest[j][3].view(-1,windowsize,3))
            groundtruth=Xtest[j][1].detach().numpy()[0]
            recommended=F.softmax(a[0,windowsize-1,:],dim=0).detach().numpy()
            
            
            R_before.CheckList(groundtruth,recommended)
            
            
            """ Finding distribution as input to dirichlet distribution file"""
            # eee=np.sort(F.softmax(a[0,:,:44]).detach().numpy(), axis=1)
            # ent=[]
            # xn=[]
            # for et in range(3):
                
                
            #     ent.append(eee[et,-1]-eee[et,-2])
            #     xn.append(eee[et,-2])
                
            # distribution.append([ent,100*R_before.identified_counter1/R_before.total_counter, xn, eee])
            
            """ applying active learning """
            eee=np.sort(F.softmax(a[0,:,:44],dim=1).detach().numpy(), axis=1)
            marg=eee[2,-1]-eee[2,-2]
            
            if marg<0.19:
                net=UpdateNet2(net,Xtest[j],X) 
                a, b=net(Xtest[j][0].view(-1,windowsize,48),Xtest[j][2].view(-1,windowsize,4),Xtest[j][3].view(-1,windowsize,3))
                groundtruth=Xtest[j][1].detach().numpy()[0]
                recommended=F.softmax(a[0,windowsize-1,:],dim=0).detach().numpy()
                #print(groundtruth, np.argmax(recommended))
            else:
                R.CheckList(groundtruth,recommended)
                
                
                
            
            

        top1_before.append(100*R_before.identified_counter1/R_before.total_counter)  
        top5_before.append(100*R_before.identified_counter5/R_before.total_counter)
        top10_before.append(100*R_before.identified_counter10/R_before.total_counter)
        top20_before.append(100*R_before.identified_counter20/R_before.total_counter)   
        
        top1.append(100*R.identified_counter1/R.total_counter)
        top5.append(100*R.identified_counter5/R.total_counter)
        top10.append(100*R.identified_counter10/R.total_counter)
        top20.append(100*R.identified_counter20/R.total_counter)    
        
        
        
        
        
        
        
            
        
        
        
        test_sim=[]
        
        
   
 
#pickle.dump([distribution],open('marginal_distr2.p','wb'))
FF=np.array([top1_before, top5_before,top10_before])
print('before', np.mean(FF,axis=1))

FF=np.array([top1, top5,top10])
print('bafter', np.mean(FF,axis=1))

pickle.dump([top1_before, top5_before,top10_before,top1, top5,top10],open('res_end2end_act_inituser.p','wb'))    
    
    
    
    



