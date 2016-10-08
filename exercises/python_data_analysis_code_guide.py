# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 27 Aug 2016

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

#Module 2 Basic of Numpy

# numpy can do arithematic, but python cannot
# a = [1,1,1,1]
# b = [2,2,2,2]
# #print(a-b)
# a1 = np.array(a)
# b1 = np.array(b)
# print(a1/b1)

# numpy ensure all elements are same format
# a = [1,1,1,1,'hi',5.9]
# print(a)
# a1 = np.array(a)
# print(a1)

# numpy can specify the precision 
# a = [1,4,6,7]
# a1 = np.array(a,dtype=np.float64)
# print(a1)

# 1D Numpy Array
# a = [1,4,6,7]
# a1 = np.array(a,dtype=np.float32)
# print(a1.ndim)
# print(a1.shape)
# print(len(a1))
# print(a1.dtype)

# a =[[1,1],[2,2],[3,3]]
# b =[[3,3],[2,2],[1,1]]
# print(a+b)
# a1 = np.array(a,dtype=np.float32)
# b1 = np.array(b)
# print(a1+b1)
# print(a1)
# print(a1.ndim)
# print(a1.shape)
# print(len(a1))
# print(a1.dtype)

# Numpy can do logical indexing
# a = [3,4,7,-1,-2,6,8,-9,3]
# s = 0
# for i in a:
# 	if i>0:
# 		s = s+i
# print(s)
# a1 = np.array(a)
# print(sum(a1[a1>0]))

# Useful numpy functions
# for i in range(1,10,3):
# 	print(i)
#a = np.arange(1,10,3)
# a = np.linspace(1,10,8)
# print(a)

# Reshape
# a = np.arange(12).reshape(4,-1)
# print(a)

# Math
#a = [4,5,6]
#print(np.sin(5))
#print(np.mean(a))
# a = np.arange(12).reshape(4,-1)
# print(a)
# print(np.mean(a,axis=1)) #row wise 
# print(np.mean(a,axis=0)) #col wise

# Slicing
# a = [3,4,5,6,7,8]
# print(a[0:6:2])
# a1 = np.array(a)
# print(a1[0:6:2])

# Linear Algebra

# Matrix Multiplication

# a = np.array([[1,1],[1,1]])
# print(a)
# b = np.array([[2,2],[2,2]])
# print(b)
# print(np.dot(a,b))

# a = np.matrix([[1,1],[1,1]])
# b = np.matrix([[2,2],[2,2]])
# print(a*b)
# a = np.array([[3,2],[4,-2]])
# b = np.array([8,6]).T 
# [x,y] = np.linalg.solve(a,b)
# print(x)
# print(y)

# Module 3: Basic of Matplotlib

# Basic Plots
# x = np.linspace(0,4*np.pi,200)
# y = np.sin(x)
# y2 = np.cos(x)
# y3 = y*y2
# y4 = y*y-y2*y2
# plt.subplot(2,2,1)
# plt.plot(x,y,color='red',marker='o',label='sine')
# plt.subplot(2,2,2)
# plt.plot(x,y2,color='blue',marker='^',label='cosine')
# plt.subplot(2,2,3)
# plt.plot(x,y3)
# plt.subplot(2,2,4)
# plt.plot(x,y4)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('A sine curve')
# plt.legend(loc='upper left')
# plt.show()

# Scatter Plots
# x = np.linspace(0,10,200)
# y = x+np.random.randn(len(x))
# plt.scatter(x,y)
# plt.show()

# Bar Plot

# people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
# y_pos = np.arange(len(people))
# performance = 3 + 10 * np.random.rand(len(people))
# error = np.random.rand(len(people))

# plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
# plt.yticks(y_pos, people)
# plt.xlabel('Performance')
# plt.title('How fast do you want to go today?')

# plt.show()

# month = ('May','Jun','Jul','Aug')
# x = np.arange(len(month))
# PSI = np.array([30,40,50,200])
# plt.bar(x,PSI)
# plt.xticks(x,month)
# plt.show()

# Histogram 

# x = np.random.randn(1000)
# plt.hist(x)
# plt.show()

# Module 4 Pandas 

# Series
# Difference between Numpy Array and Pandas Series
# a = [1,2,3,4]
# b = [4,3,2,1]
# a1 = np.array(a)
# b1 = np.array(b)
#print(a1+b1)
# a2 = pd.Series(a,index=['jan','feb','mar','aug'])
# print(a2)
# b2 = pd.Series(b,index=['apr','may','feb','jan'])
# print(b2)
# print(a2+b2)

#a = pd.Series(np.random.randn(1000))
#print(a.head(10))
#print(a.tail(10))
#print(a[300:350])

# a = [3,4,5,7,9,1,2]
# a1 = pd.Series(a)
# print(a1[[1,2,5]])

# Data Frame

# a = [[3,4],[5,6]]
# b = [[6,5],[4,3]]
# a1 = np.array(a)
# b1 = np.array(b)
# #print(a1+b1)
# a2 = pd.DataFrame(a,index=[1,2],columns=['d','b'])
# print(a2)
# b2 = pd.DataFrame(b,index=[3,2],columns=['c','b'])
# print(b2)
# print(a2+b2)

# Data filtering
#print(data[(data.Price>100) & (data.Sector == 'Health Care')]['Book Value'])

# Select row (index) data
# print(data[2:5])
# print(data.ix[['ACT','YHOO']])

# Select column data
#print(data['Price'])
#print(data[['Price','Book Value']])
#print(data.Price)

# Select row and column
#print(data.ix[['ACT','YHOO']][['Price','Book Value']])
#print(data[['Price','Book Value']].ix[['ACT','YHOO']])

# aapl = pd.read_csv('aapl.csv',index_col='Date')
# msft = pd.read_csv('msft.csv',index_col='Date')

# ts = aapl['Close']
# ts.plot()
# plt.show()
# aapl.insert(0,'Symbol','AAPL')
# msft.insert(0,'Symbol','MSFT')

# data = pd.concat([aapl,msft]).reset_index()

# p = data.pivot(index='Date',columns='Symbol',values='Close')
# print(p)

# aapl1 = aapl[['Close','Volume']][:20].reset_index()
# msft1 = msft[['Close','Volume']][10:30].reset_index()
#print(msft1)
#print(aapl1)
#print(msft1)
#print(pd.concat([aapl1,msft1],keys=['AAPL','MSFT'],axis=1,join='inner'))

#print(pd.merge([aapl1,msft1],on='Date'))


# Module 5 Import/Export Data 

# from pandas_datareader import data,wb

# aapl = data.DataReader("AAPL",'google','2016-08-01','2016-08-26')
# print(aapl[aapl.Close>107])

# import Quandl

# dji = Quandl.get("YAHOO/INDEX_DJI",trim_start='2016-01-01',trim_stop='2016-08-26')
# print(dji.tail(10))

# Import data from csv
#data = pd.read_csv('sp500.csv',index_col='Symbol',usecols=[0,2,3,7])


# Module 7 Scikits Learn

# from sklearn import datasets

# iris = datasets.load_iris()

# from sklearn.cross_validation import train_test_split

# X,y = iris.data,iris.target

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#Step 1: Load the model
#from sklearn import svm
# from sklearn import naive_bayes
# #machine = svm.SVC()
# machine = naive_bayes.GaussianNB()

# #Step 2: Training
# machine.fit(X_train,y_train)

# #Step 3: Testing
# # print(machine.predict(X_test))
# # print(y_test) 

# # Step4 : Deployment
# flower = [[4,5,3,1]]
# print(machine.predict(flower))

# x = iris.data[:,0]
# y = iris.data[:,1]

# plt.scatter(x,y,c=iris.target)
# plt.show()

#print(iris.data)
#print(iris.target)
