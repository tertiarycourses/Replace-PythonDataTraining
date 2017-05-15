# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Update: 13 Jan 2017

# Module 2 Overview of Numpy

import numpy as np

# Create 1D Numpy Array
# a = [1,2,3,4]
# a1 = np.array(a,dtype=np.int16)
# print(a1)

# Create 2D Numpy Array
# a = [[1,2],[3,4]]
# a1 = np.array(a,dtype=np.float32)
# print(a1)

# Numpy Array Attributes
# print(a1.dtype)
# print(a1.ndim)
# print(a1.shape)

# Numpy Arithmetics
# a = [1,2,3,4]
# b = [4,3,2,1]
# a1 = np.array(a)
# b1 = np.array(b)
# print(a1+b1)

# a = [[1,2],[3,4]]
# b = [[4,3],[2,1]]
# a1 = np.array(a)
# b1 = np.array(b)
# print(a1-b1)
# print(a+b)
# print(a1+b1)

# Special functions
# a = np.arange(2,20,3)
# a = np.linspace(2,20,3)
# np.random.seed(25)
# a = np.random.randn(1000)*4+3
# print(a)

# Math functions 
# a = np.exp(2)
# a = np.sqrt(4)
# a = np.sin(np.pi/2)
# a = np.cos(np.pi/2)
# print(a)

# Selecting and Slicing Array Elements
# a = np.arange(2,20,1.5)
# print(a[:3])
# print(a[3:])
# print(a[:])
# print(a[2:7])
# print(a[-1])

# Filtering Array Elements
# a = np.arange(-10,10,2)
# print(a)
# print(a>0)
# print(a[a>0])

# Exercise - filter out all those numbers
# divisible by 3
# a = np.arange(1,100,1)
# print(a[a%3==0])

# Transforming Array
# a = np.arange(24).reshape(-1,6)
# print(a)
# b = a.ravel()
# print(b)

# Statistics 
# a = np.random.randn(100)*2+5
# print(a.mean())
# print(a.std())

# a = np.arange(24).reshape(6,4)
# print(a.mean(axis=0))
# print(a.mean(axis=1))
# print(a.std(axis=0))
# print(a.std(axis=1))

