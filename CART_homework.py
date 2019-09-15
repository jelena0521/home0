#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:52:15 2019

@author: liujun
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#先观察数据
digits=load_digits()
data=digits.data
print(data.shape) 
print(digits.images[0]) #image 为像素矩阵  
print(digits.target[0]) #target为数字
plt.gray() #灰色打印
plt.imshow(digits.images[0])
plt.show()

#分割数据
train_x,test_x,train_y,test_y=train_test_split(data,digits.target,test_size=0.2,random_state=4)

#标准化数据 只需对X
ss=preprocessing.StandardScaler()
train_x_ss=ss.fit_transform(train_x)
test_x_ss=ss.transform(test_x)

#创建模型 采用gini系数，对树深不做要求，最小叶子数为1
cart=tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_leaf=1)
cart.fit(train_x_ss,train_y)
predict_y=cart.predict(test_x_ss)
print('cart的准确率为:{}'.format(accuracy_score(predict_y,test_y)))



#from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
#from sklearn.metrics import accuracy_score
#from sklearn.datasets import load_digits
#from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt
#
#
#
## 加载数据
#digits = load_digits()
#data = digits.data
## 数据探索
#print(data[0])
#print(data.shape)
## 查看第一幅图像
#print(digits.images[0])
## 第一幅图像代表的数字含义
#print(digits.target[0])
## 将第一幅图像显示出来
#plt.gray()
#plt.imshow(digits.images[0])
#plt.show()
#
## 分割数据，将25%的数据作为测试集，其余作为训练集
#train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)
#
## 采用Z-Score规范化
#ss = preprocessing.StandardScaler()
#train_ss_x = ss.fit_transform(train_x)
#test_ss_x = ss.transform(test_x)
#
## 创建LR分类器
#lr = LogisticRegression()
#lr.fit(train_ss_x, train_y)
#predict_y=lr.predict(test_ss_x)
#print('SVM准确率: %0.4lf' % accuracy_score(predict_y, test_y))