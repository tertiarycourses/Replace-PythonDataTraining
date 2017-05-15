# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Update: 13 Jan 2017

# Module 5 Intro to Machine Leanrning with Scikit Learn

# import sklearn as sk 

# from sklearn import datasets

# iris = datasets.load_iris()
# X,y = iris.data,iris.target
#print(iris.target)
#print(iris.data)

# Supervised Learning: Classfification
# Step 1 Model

# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier()

# Step 2 Training
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# clf.fit(X_train,y_train)

# Step 3 Testing
# y2 = clf.predict(X_test)
# print(y2)
# print(y_test)

# import numpy as np

# y3 = clf.predict([[5,3.2,2,4.3]])
# print(y3)

# Supervised Learning: Regression

# import numpy as np 
# import matplotlib.pyplot as plt 

# X = np.linspace(1,20,100).reshape(-1,1)
# y = X + np.random.randn(len(X)).reshape(-1,1)
# plt.scatter(X,y)
# plt.show()

# Step 1: Model

# from sklearn import linear_model

# lm = linear_model.LinearRegression()

#  Step 2: Fitting

# lm.fit(X,y)

# Step 3: Prediction

# y2 = lm.predict(X)
# plt.plot(X,y2,'-r')
# plt.scatter(X,y)
# plt.show()

# print(lm.predict([[10]]))

# Unsupervised Learning: Clustering

# iris = datasets.load_iris()
# X,y = iris.data,iris.target

# Step 1 Mdoel

# from sklearn import cluster

# clf = cluster.KMeans(n_clusters=3)

# Step 2 : Training

# clf.fit(X)

# Step 3

# print(clf.labels_[::10])
# print(y[::10])
