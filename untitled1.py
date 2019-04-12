#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:11:22 2019

@author: pan
"""

import numpy as np
import pandas as pd

from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import svm
# 模型评价 混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report,precision_score
if __name__ == "__main__":
    def f(x):
        if x=="-":
            return -1
            pass
        else:
            return 1
        pass
    data2=pd.read_csv("2.2.txt",header=None,sep="\t")
    data4=pd.read_csv("2.4.txt",header=None,sep="\t")
    data6=pd.read_csv("2.6.txt",header=None,sep="\t")
    data8=pd.read_csv("2.8.txt",header=None,sep="\t")
    
    name=[str(i) for i in range(16)]
    #数据转换
    def convdata(data): 
        mydata=[]
        temp=[]
        for i in range(len(data.iloc[0,:])):
            temp.append(pd.DataFrame(data.iloc[:,i].apply(lambda x:x.split(" ")).values.tolist()))
            pass
        mydata=pd.concat(temp,axis=1)
        
        temp=[]
        for i in range(0,len(mydata.iloc[0,:]),3):
            temp.append(mydata.iloc[:,i].apply(float))
            a=mydata.iloc[:,i+1].apply(f)
            b=mydata.iloc[:,i+2].apply(lambda x:x[:-1]).apply(float)
            c=a*b
            temp.append(c)
            pass
        mydata=pd.concat(temp,axis=1)
        mydata.columns=name
        return mydata
    
    data=[]
    temp=convdata(data2)
    temp["tag"]=0
    data.append(temp)
    
    temp=convdata(data4)
    temp["tag"]=1
    data.append(temp)
    
    temp=convdata(data6)
    temp["tag"]=2
    data.append(temp)
    
    temp=convdata(data8)
    temp["tag"]=3
    data.append(temp)
    
    data=pd.concat(data)
    
    
    #分割测试数据和训练数据
    test, train = train_test_split(data, train_size=0.3, random_state=1)
    
    #训练数据
    y=train['tag']
    X=train
    del train['tag']
    
    #测试数据
    TagTest=test['tag']
    XdataTest=test
    del test['tag']
    
    
    # 使用SVM作为分类器
    clf = svm.SVC()
    
    
    
    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
    
    
    # 设置可能学习的参数
    param_dist = {"kernel": ["linear","rbf","sigmoid"],
                  "C": np.logspace(-2, 2, 10),
                  "gamma": np.logspace(-2, 2, 10)
                  }
    
    # 随机搜索， randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    #起始时间
    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
#    report(random_search.cv_results_)
    print("=============================================")
    
#    # use a full grid over all parameters
#    param_grid = {"kernel": ["linear","rbf","sigmoid"],
#                  "C": np.logspace(-2, 2, 10),
#                  "gamma": np.logspace(-2, 2, 10)
#                  }
#    
#    # 网格搜索， grid search
#    grid_search = GridSearchCV(clf, param_grid=param_grid)
#    start = time()
#    grid_search.fit(X, y)
#    
#    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#          % (time() - start, len(grid_search.cv_results_['params'])))
#    report(grid_search.cv_results_)
    
    #对比分类效果
    #random_search
    #预测结果        
    print("============================random_search=========================================")
    Result = random_search.predict(XdataTest)
    print('The accuracy is:',accuracy_score(TagTest,Result))
    #混淆矩阵
    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
    # 3 precision, recall, f1 score
    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
    #all
    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))
    
#    #预测结果        
#    print("============================grid_search=========================================")
#    Result = grid_search.predict(XdataTest)
#    print('The accuracy is:',accuracy_score(TagTest,Result))
#    #混淆矩阵
#    print('The confusion matrix is:\n',confusion_matrix(TagTest,Result))
#    # 3 precision, recall, f1 score
#    print('The precision, recall, f1 score are:\n',classification_report(TagTest,Result))
#    #all
#    print('The precision are:\n',precision_score(TagTest,Result,average='micro'))    
    
    
    pass