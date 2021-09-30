# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:21:17 2021

@author: arash
"""

import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import torch.nn.functional as F


def CleanUser(user_profile):
    user=pd.read_csv(user_profile).dropna()
    a=user.max(axis=0)[1:]
    for i, c in enumerate(user.columns[1:]):
        user[c]=user[c].div(a.iloc[i])
    return user
    
def GroupByUser(dataframe,user_id,remove_cols=None):
    a=dataframe.groupby(user_id)
    X=dict()
    for name, group in a:
        if remove_cols!=None:
            group=group.drop(remove_cols, axis=1)
        X[name]=group
    return X    

def CleanData(data_file):
    data=pd.read_csv(data_file).dropna()
    
    gr=data.groupby('id')
    lists=[]
    by_user=dict()
    for nam, gro in gr:
        lists.append(list(gro['activity']))
        by_user[nam]=gro
        by_user[nam]=by_user[nam].reset_index(drop=True)
    return by_user


def CleanDataSorted(data_file):
    
    data=pd.read_csv(data_file).dropna()
       
    gr=data.groupby('id')
    
    by_user=dict()
    for nam, gro in gr:
        for i in [0,2,4,7,9,11,14,16,18,21,23,25]:
            d=gro.loc[gro['day']==i]
            d.sort_values(by=['activity'],inplace=True)
            gro.loc[gro['day']==i]=d
        
        by_user[nam]=gro
        by_user[nam]=by_user[nam].reset_index(drop=True)
    return by_user


def WordEmbedding(training):
    # Create CBOW model
    model = gensim.models.Word2Vec(training, min_count = 1, vector_size=5 , window = 1)
    model.train(training, total_examples=10, epochs=5)
    return model


def DataPreprationWindow(data,wordmodel,user,exercise,windowsize=3):
    xtemp=[]
    ytemp=[]
    ztemp=[]
    Xtemp=[]
    for k in data.keys():
        if len(data[k])>windowsize:
            
            for i in range(windowsize,len(data[k])):
                temp=[]
                te=[]
                excer=[]
                for j in range(windowsize):
                    #print(data[k].loc[i+j-windowsize,'activity'])
                    acti=data[k].loc[i+j-windowsize,'activity']
                    a=wordmodel[acti].tolist()
                    
                    a.append(data[k].loc[i+j-windowsize,'success'])
                    a.append(data[k].loc[i+j-windowsize,'difficulty'])
                    a.append(data[k].loc[i+j-windowsize,'normlag'])
                    a.append(data[k].loc[i+j-windowsize,'normfreq'])
                    temp.append(a)
                    excer.append(exercise[exercise['name']==acti].to_numpy()[0,1:])
                    te.append(user[user['id']==k].to_numpy()[0,1:])
                    
                    
                    
                xtemp.append(temp)
                #a=wordmodel[data[k].loc[i,'activity']].tolist()
                a=np.argmax(wordmodel[data[k].loc[i,'activity']])
                # a.append(data[k].loc[i,'success'])
                # a.append(data[k].loc[i,'difficulty'])
                # a.append(data[k].loc[i,'normlag'])
                # a.append(data[k].loc[i,'normfreq'])
                ytemp.append([a])
                ztemp.append(te)  
                Xtemp.append(excer)  
   
    #print(len(ztemp))
    #print(len(xtemp))
    tensor_x = torch.Tensor(xtemp)
    tensor_y = torch.Tensor(ytemp)
    tensor_ex=torch.Tensor(Xtemp)
    
    #X = TensorDataset(tensor_x,tensor_y,torch.Tensor(ztemp),tensor_ex,transform=transforms.Compose([transforms.ToTensor(), AddGaussianNoise()]))
    #X = TensorDataset(tensor_x,tensor_y,torch.Tensor(ztemp),tensor_ex,transform=AddGaussianNoise())
    X = TensorDataset(tensor_x,tensor_y,torch.Tensor(ztemp),tensor_ex)
    return X        



class Recall:
    def __init__(self, wordmodel, dic=None):
        self.identified_counter1=0
        self.identified_counter5=0
        self.identified_counter10=0
        self.identified_counter20=0
        self.total_counter=0
        self.wordmodel=wordmodel
        self.invers_rank=0
        self.Dic=dic
        self.exc=list(wordmodel.keys())[0]
    
    def top10(self,groundtruth, recommended):
        #groundtruth=np.argmax(groundtruth)
        groundtruth=int(groundtruth)
        
        flag=False 
        for i in range(10):
            if np.argmax(recommended[i])==groundtruth:
                flag=True
        return flag        
    
    def CheckList(self,groundtruth, recommended):
        #groundtruth=np.argmax(groundtruth)
        #print(groundtruth)
        groundtruth=int(groundtruth)
        
        self.total_counter+=1
        
        flag=False
        Tem=np.array(recommended)
        if np.argmax(Tem)==groundtruth:
            self.identified_counter1+=1
        
        
        for i in range(5):
            #print(np.argmax(recommended[i]))
            if np.argmax(Tem)==groundtruth:
                flag=True
            else:
                Tem[np.argmax(Tem)]=0
                
        if flag:
            self.identified_counter5+=1
        flag=False    
        Tem=np.array(recommended)
        for i in range(10):
            if np.argmax(Tem)==groundtruth:
                flag=True
            else:
                Tem[np.argmax(Tem)]=0
        if flag:
            self.identified_counter10+=1
        
        flag=False     
        Tem=np.array(recommended)
        for i in range(20):
            if np.argmax(Tem)==groundtruth:
                flag=True
            else:
                Tem[np.argmax(Tem)]=0
        if flag:
            self.identified_counter20+=1     
        
        


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
    
from sklearn.metrics.pairwise import cosine_similarity
import math
def Uncertainty(recommended,wvmodel): #recommended is the predicted vector by RNN
    recommendedlist2=wvmodel.wv.most_similar([recommended[0:5]],topn=20)
    recommendedlist=[]
    X=[]
    Xang=[]
    for i in range(20):
        recommendedlist.append(recommendedlist2[i][0])
        X.append(recommendedlist2[i][1])
        Xang.append(math.acos(recommendedlist2[i][1]))
    

    return recommendedlist, X, Xang    
        
            
def DistanceExcer(wvmodel,recom,ground):
    return np.linalg.norm(np.array(recom)- ground)/5
    #return np.linalg.norm(np.array(wvmodel.decod(np.array(recom)))- np.array(wvmodel.decod(ground)))
    
    

def IdentifySimilarUsers(Userprofiles, Newuser):
    #Userprofiles=Userprofiles.reset_index()
    Y=Userprofiles.loc[Userprofiles['id']==Newuser]
    #Y = Y.set_index("id")
    
    data=pd.DataFrame(Userprofiles)
    #data = data.set_index("id")
    #data=data.drop(Newuser)
    ind=data.index
    data=data.drop(ind[data['id']==Newuser])
    
    data=data.reset_index()
    data=data.drop(labels="index", axis=1)
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(Y)
    a=list(data.loc[indices[0],'id'])
    return [int(x) for x in a]
     
    
 
    
def IdentifyUsersHistory(by_user2, test, indexsofar):
    sim=[]
    for k in by_user2.keys():
        x=by_user2[k]
        count=0
        for i in range(indexsofar):
            if x.loc[i,'activity']==test.loc[i,'activity']:
                count+=1
        sim.append(count)        
            
    L=list(by_user2.keys())
    U=[]
    
    for i in range(3):
        idx=np.argmax(sim)
        U.append(L[idx])
        sim[idx]=0
        
    return U
         


def ExerLookup(exer):
    co=0
    e=[0]
    dic=dict()
    for l in exer:
        e=[0]*44
        #print(co,e)
        e[co]=1
        co+=1
        dic[l]=torch.from_numpy(np.array(e)).float()
    return dic 















