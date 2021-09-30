# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:10:46 2021

@author: arash
"""
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


def AugmentDic():
    expert=pd.read_csv('expert.csv').dropna()
    for a in expert.columns[1:]:
        expert[a]=expert[a]/max(expert[a])
    A=expert.to_numpy()
    P=pairwise_distances(A[:,1:])   
    D=dict()
    for ind, row in expert.iterrows():
        P[ind,ind]=10000
        a=np.argmin(P[ind,:])
        D[row['name']]=expert.loc[a,'name']
    return D


def Augment(by_user, D, user):
    lis=list(by_user.keys())
    for k in lis:
        X=pd.DataFrame(by_user[k])
        random=np.random.randint(3000,7000)
        u=user[user['id']==k]
        u['id']=random
        user=user.append(u)
        idx=np.random.randint(0, len(X), size=10)
        Y=X.copy()
        for idx2 in idx:
            try:
                tempD=D[Y.loc[idx2,'activity']]
                Y.loc[idx2,'activity']=np.random.choice(tempD)
                Y['id']=random
            except:
                pass    
        by_user[random]=Y
        
    return by_user, user   
        
        
            
        
def AugmentDicRule(rules, window=2):
    rules2=rules[(rules['antecedent_len']==window) & (rules['consequents_len']==1) ]
    uni_rule=rules2.groupby('antecedents')
    senten_temp=[]
    senten=[]
    for name, group in uni_rule:
        d=group['consequents']
        temp=[list(x)[0] for x in d]
        temp.sort()
        senten.append(temp)
        #temp=frozenset(temp)
        #senten_temp.append(temp)
    
    
    D=dict()
    for j in range(len(senten)):
        x=senten[j]
        #print(j)
        for i in range(len(x)):
            #print(len(D))
            x2=x.copy()
            del x2[i]
            if x[i] in D.keys():
                [D[x[i]].append(yy) for yy in x2]
            else:
                D[x[i]]=x2
            
    for k in D.keys():
        D[k]=np.unique(D[k])
    
    return D













