<<<<<<< HEAD
=======
# Code file for Pandas Essential Training for Finance
# Copyright: Tertiary Infotech Pte Ltd
# Author: Alfred Ang
# Date: 24 Aug 2016
>>>>>>> test

import numpy as np
import matplotlib.pyplot as plt 

<<<<<<< HEAD
#Module 3: Import/Export Data

make some changes here
=======
# Module 1: Basics of Numpy

#1-D Numpy Array

# a = [1,2,-4,3,4,-2,3.4,-3.5]
# b = [4,3,2,1]
#print(a)
#print(a-b)

# Compute the sum of elements in a that are greater than 0
# s = 0
# for i in a:
# 	if i>0:
# 		s = s + i
# print(s)

# a1 = np.array(a,dtype=np.float32)
# print(a1)
# print(a1.dtype)
# print(sum(a1[a1>0]))
# s = 0
# for i in a1:
# 	if i>0:
# 		s = s + i
# print(s)

# b1 = np.array(b)

# Reshape
# a = np.arange(1,13,1)
# b = a.reshape(4,3)
# print(a)
# print(b)

#2D Numpy Array
# a = [[1,2],[3,4],[5,6]]
# a = [1,2,3,4]
# print(a)
# print(a[1][1])

# b = np.array(a)
# print(b)
# print(len(b))
# print(b[1,1])

# Statistical Functions in Numpy
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a)
# b = a[[0,2],:]
# print(b)
# print(b[:,[0,2]])
# print(a[1:3,:])
# print(np.mean(a,axis=0))

# a = np.arange(3,10,3)
# print(a)
# print(np.std(a))





#print(a1/b1)
>>>>>>> test

#Module 2: Basics of Matplotlib

#Scatter plot
# x = np.arange(1,11,1)
# y = np.random.rand(1,10)
#plt.scatter(x,y)

#Bar Chart
# x = [1,2,3,4]
# y = [3,4,7,8]
# plt.bar(x,y)

#Histogram
# x = np.random.randn(1000)
# plt.hist(x,10)

# plt.show()

# def myplot(x,y):
# 	plt.plot(x,y,linestyle="-",marker='o',color="red",label="sine")
# 	plt.grid()
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	plt.title('A simple sine and cosine curve')
# 	plt.legend(loc="upper left")

# x = np.linspace(0,4*np.pi,200)
# y = np.sin(x)
# y2 = np.cos(x)
# plt.subplot(2,1,1)
# myplot(x,y)
# #plt.plot(x,y,linestyle="-",marker='o',color="red",label="sine")
# plt.subplot(2,1,2)
# myplot(x,y2)
#plt.plot(x,y2,linestyle="-",marker='^',color="green",label="cosine")
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('A simple sine and cosine curve')
# plt.legend(loc="upper left")
<<<<<<< HEAD
plt.show()


# Module 1: Basics of Numpy
# Statistical Functions in Numpy
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a)
# b = a[[0,2],:]
# print(b)
# print(b[:,[0,2]])
#print(a[1:3,:])
#print(np.mean(a,axis=0))

# a = np.arange(3,10,3)
# print(a)
# print(np.std(a))

# Reshape
#a = np.arange(1,13,1)
#b = a.reshape(4,3)
#print(a)
#print(b)
#2D Numpy Array
#a = [[1,2],[3,4],[5,6]]
#a = [1,2,3,4]
# print(a)
# print(a[1][1])

# b = np.array(a)
# print(b)
# print(len(b))
#print(b[1,1])

#1-D Numpy Array
# a = [1,2,-4,3,4,-2,3.4,-3.5]
# b = [4,3,2,1]
#print(a)
#print(a-b)
# Compute the sum of elements in a that are greater than 0
# s = 0
# for i in a:
# 	if i>0:
# 		s = s + i
# print(s)

# a1 = np.array(a,dtype=np.float32)
# print(a1)
# print(a1.dtype)
#print(sum(a1[a1>0]))
# s = 0
# for i in a1:
# 	if i>0:
# 		s = s + i
# print(s)

# b1 = np.array(b)

#print(a1/b1)
=======

plt.show()


#Module 3: Import/Export Data

#Module 4: Pandas

#Module 5: Time Series

>>>>>>> test
