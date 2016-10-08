import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

#Module 2 Numpy

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

# Module 3: Matplotlib

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

# Module 3 Pandas

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

# from pandas_datareader import data,wb

# aapl = data.DataReader("AAPL",'google','2016-08-01','2016-08-26')
# print(aapl[aapl.Close>107])

import Quandl

dji = Quandl.get("YAHOO/INDEX_DJI",trim_start='2016-01-01',trim_stop='2016-08-26')
print(dji.tail(10))


