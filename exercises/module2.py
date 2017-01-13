# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 10 Oct 2016

# Module 2 Numpy

import numpy as np

# a = [2,3,4]
# a1 = np.array(a)
# b = [4,3,2]
# b1 = np.array(b)
# Differences between Numpy and Python List
# print(a+b)
# print(a1+b1)
# print(a-b)
# print(a1-b1)

# a = [2, 'hi']
# print(a)
# a1 = np.array(a)
# print(a1)

# a = [2,3,5]
# a1 = np.array(a,dtype=np.int16)
# print(a1)
# b1 = a1.astype(np.float64)
# print(b1)
# print(a1.dtype)
# print(b1.dtype)

# a = [[1,5,2],[4,5,6]]
# b = [[6,5,4],[2,5,1]]
# print(a+b)
# a1 = np.array(a)
# b1 = np.array(b)
# print(a1+b1)

# a = [[1,5,2],[4,5,6]]
# a1 = np.array(a)
# print(a1.ndim)
# print(a1.shape)
# print(len(a1))
# print(a1.dtype)

#a = np.linspace(1,10,40) 
#a = np.logspace(-10,-1,10)
#a = np.arange(1,10,2)
#print(a)

# reshape
# a = np.arange(1,12,2)
# print(a)
# b = a.reshape(3,2)
# print(b)
# c = b.ravel()
# print(c)
# c = a.reshape(-1,2)
# print(c)

#stacking arrays
# a = np.array([[1,1],[2,2],[3,3]])
# b = np.array([[4,4],[5,5],[6,6]])
# print(np.vstack([a,b]))
# print(np.hstack([a,b]))

# a = [[1,5,2],[4,5,6],[7,8,9]]
# a1 = np.array(a)
# print(a1)
# print(a1[[0,2],:][:,[0,2]])

# a = [3,4,2,8,6]
# a1 = np.array(a)
# print(a1[a1>3])

# a = [3,4,7,-1,-2,6,8,-9,3]
# a1 = np.array(a)
#print(sum(a1[a1>0]))

# Numpy Math Functions
# print(np.exp(1))
# print(np.sin(np.pi/2))
# print(np.cos(np.pi/2))
# print(np.sqrt(4))

# Numpy Statistical Functions
#a = [3,4,2,8,6]
# a = [[1,5,2],[4,5,6],[7,8,9]]
# a1 = np.array(a)
# print(a1)
# print(np.mean(a1,axis=0))
# print(np.std(a1))

# Random Number
# np.random.seed(10)
#print(np.random.rand(5))
# print(np.random.randn(5))
#print(np.random.randint(1,6,2))

# Linear Algebra
# a = np.array([[1,1],[1,1]])
# b = np.array([[2,2],[2,2]])
# print(a*b)
# a = np.matrix([[1,1],[1,1]])
# b = np.matrix([[2,2],[2,2]])
# print(a*b)

# from numpy import linalg
# a = np.matrix([[2,4],[3,-1]])
# print(a)
# b = linalg.inv(a)
# print(b)
# print(a*b)

