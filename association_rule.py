# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:30:30 2020

@author: arash
"""
from __future__ import print_function

#from apyori import apriori, association_rules
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import GroupByUser



def RuleMining(dstress):

    data=pd.read_csv(dstress).dropna()
    df=GroupByUser(data,'id')
    
    """ ******************************************************"""
    """ Frequent item mining, Finding rules """
    dataset=[]
    for k in df.keys():
        temp=[]
        for index, row in df[k].iterrows():
            temp.append(row['activity'])
        dataset.append(temp)    
    
    
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df_training = pd.DataFrame(te_ary, columns=te.columns_)
    
    #frequent_itemsets=apriori(training, min_support=0.5, min_confidence=0.7, min_lift=1.2, min_length=2)
    frequent_itemsets=apriori(df_training, min_support=0.7, use_colnames=True)    
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))
    return rules, frequent_itemsets







