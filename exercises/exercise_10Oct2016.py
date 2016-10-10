# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 10 Oct 2016

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

#Module 2 Basic of Numpy

#Differences between Python List and Numpy Array
# First Difference: Python list cannot do arithmetic
# a = [1,1,1,1]
# b = [2,2,2,2]
# print(a)
# print(b)
# print(a+b)
#print(a-b)
# a1 = np.array(a)
# b1 = np.array(b)
# print(a1)
# print(b1)
# print(a1+b1)
# print(a1/b1)

# Second difference between Numpy and Python list
# For Numpy, all elements are same data type
# a = [1,1,1,1,'hi',5.9]
# print(a)
# a1 = np.array(a)
# print(a1)

# Third difference between Numpy and Python list
# Numpy can specify the precision for data type
# a = [1,4,6,7]
# a1 = np.array(a,dtype=np.int16)
# print(a1.dtype)

# 1D Numpy Array
# a = [1,4,6,7]
# a1 = np.array(a,dtype=np.float32)
# print(a1.ndim)
# print(len(a1))
# print(a1.dtype)

# 2D Numpy Array
# a =[[1,1],[2,2],[3,3]]
# b =[[3,3],[2,2],[1,1]]
# print(a+b)
# a1 = np.array(a,dtype=np.float32)
# b1 = np.array(b,dtype=np.float64)
# c1 = a1+b1
#print(c1)
# print(c1.dtype)
# print(c1.ndim)
# print(c1.shape)
# print(len(c1))
# print(c1.dtype)

# Fourth difference between Numpy and Python list
# Numpy can do logical indexing
a = [3,4,7,-1,-2,6,8,-9,3]
# Python use a for loop to sum the elements greater than 0
# s = 0
# for i in a:
# 	if i>0:
# 		s = s+i
# print(s)
# a1 = np.array(a)
# print(a1>0)
# print(a1[a1>0])
# print(a1[a1>5])
# print(sum(a1[a1>0]))

# Useful numpy 1D functions
# for i in range(1,20,3):
# 	print(i)
# a = np.arange(1,20,3)
# print(a)
# a = np.linspace(1,20,8)
# print(a)
# a = np.logspace(-10,-1,10)
# print(a)

# Reshape a Numpy array
# a = np.arange(12)
# print(a)
# b = a.reshape(4,3)
# print(b)
# b = a.reshape(4,-1)
# print(b)
# b = a.reshape(-1,3)
# print(b)
# c = a.reshape(12,-1)
# print(c)

# Math Functions in Numpy Array
# import math
# print(math.sin(math.pi/2))
#print(np.sin(np.pi/2))
# a = [4,5,6]
# print(np.mean(a))
# a = np.arange(12).reshape(4,-1)
# print(a)
# print(np.mean(a,axis=1)) #row wise 
# print(np.mean(a,axis=0)) #col wise

#Random Numbers
# a = 3+np.random.randn(2,3)*0.1
# print(a)

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
# plt.plot(x,y,'o-',color='#ff0000')
# y2 = np.cos(x)
# fig = plt.figure()
# fig1 = fig.add_subplot(2,1,1)
# fig1.plot(x,y2,'o-.',color='blue')
# y3 = y*y2
# fig2=fig.add_subplot(2,1,2)
# fig2.plot(x,y3,'^--',color='green')
# plt.show()
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
# Horizontal Bar Plot
# people = ['Tom', 'Dick', 'Harry', 'Slim', 'Jim']
# height = 170 + 10 * np.random.randn(len(people))
# x = np.arange(len(people))
# plt.barh(x,height,align="center")
# plt.yticks(x,people)
# plt.show()

# Vertical Bar Plot
#people = ['Tom', 'Dick', 'Harry', 'Slim', 'Jim']
# height = 170 + 10 * np.random.randn(len(people))
# x = np.arange(len(people))
# plt.bar(x,height,align="center")
# plt.xticks(x,people)
# plt.show()


# Histogram 

# x = np.random.randn(100000)
# plt.hist(x,100)
# plt.show()

# Module 4 Pandas 

# Series
# Difference among Python List, Numpy Array and Pandas Series
# a = [1.1,2.4,3.5,4.6]
# print(a[0])
# b = [4.6,3.2,2.5,1.8,5.9]
# a1 = pd.Series(a,index=[1,2,3,4])
# print(a1)
# b1 = pd.Series(b,index=[5,2,4,1,3])
# print(b1)
# print(a1+b1)


# Access Pandas Series elements
# a = pd.Series(np.random.randn(1000))
# print(a.head(10))
# print(a.tail(10))
# print(a[300:350])

# a = [3,4,5,7,9,1,2]
# a1 = np.array(a)
# print(a1[[0,3,5]])
# a1 = pd.Series(a,index=['a','b','c','d','e','f','g'])
# print(a1[['a','d','f']])

# Data Frame

# a = [[3,4],[5,6]]
# b = [[6,5],[4,3]]
# a2 = pd.DataFrame(a,index=[1,2],columns=['d','b'])
# print(a2)
# b2 = pd.DataFrame(b,index=[3,2],columns=['c','b'])
# print(b2)
# print(a2+b2)

# a = pd.DataFrame(
#     {
#     'name': ['Ally','Jane','Belinda'],
#     'height': [160,155,163],
#     'age': [40,35,42]
#     },
#     columns = ['name','height','age'],
#     index = ['a1','a2','a3']
# 	)
# print(a)
# print(a.shape)
# print(a.columns)
# print(a.index)
# print(a.values)

# Accessing elements in Dataframe
#print(a)

# Accessing column data
# print(a.name)
# print(a['name'])
# print(a[[0]])

# Accessing row data
# print(a.ix[0])
# print(a.iloc[0])
# print(a.loc['a1'])

# Accessing scalar data
# print(a.ix[1]['height'])
# print(a.ix[1,1])

# Reindexing Data

# Missing Data

# Import data from CSV file
# sp500 = pd.read_csv("data/sp500.csv", 
#                     index_col='Symbol', 
#                     usecols=['Symbol', 'Sector', 'Price', 'Book Value'])

# print(sp500.head())
# df_ex1=pd.read_csv('ex_06-02.txt',sep="\t",header=None,index_col=4)
# print(df_ex1)

# a = pd.DataFrame({'A':[2,4,6,8],'B':[1,3,5,9]})
# print(a)
# a.to_csv('test2.csv',sep=',')

# a = pd.read_csv('F://....../test.csv',index_col=0)
# print(a)
# b = pd.DataFrame({'B':[1,6,9,10],'C':[1,5,8,5]})

# Data filtering
# print(sp500[(sp500.Price>100) & (sp500.Sector == "Health Care")] )
# print(data[(data.Price>100) & (data.Sector == 'Health Care')]['Book Value'])

# Data Merge
# sp888 = pd.read_csv("data/sp500.csv", 
#                     index_col='Symbol', 
#                     usecols=['Symbol', 'Dividend Yield'])

#print(sp888.head())
# data = pd.concat([s1,s2])

#Inner Join
# s1 = sp500.head().reset_index()
# s2 = sp888.head(3).reset_index()
# data = pd.merge(s1,s2,on="Symbol",how="inner")
# print(data)

#Left Join
# s1 = sp500.head().reset_index()
# s2 = sp888.head(3).reset_index()
# data = pd.merge(s1,s2,on="Symbol",how="left")
# print(data)

#Right Join
# s1 = sp500.head().reset_index()
# s2 = sp888.head(8).reset_index()
# data = pd.merge(s1,s2,on="Symbol",how="right")
# print(data)
#print(pd.merge([aapl1,msft1],on='Date'))

# Outer Join
# s1 = sp500[sp500.Price>200].reset_index()
# s2 = sp888.head(8).reset_index()
# data = pd.merge(s1,s2,on="Symbol",how="outer")
# print(data)
#print(pd.merge([aapl1,msft1],on='Date'))

# Data Pivot
# s1 = sp500.head(5).reset_index()
# #print(s1)
# data = s1.pivot(index='Sector',columns='Symbol',values='Price')
# print(data)

#print(data)
# aapl = pd.read_csv('aapl.csv',index_col='Date')
# msft = pd.read_csv('msft.csv',index_col='Date')

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

#Import data from the web
# pip install pandas-datareader
# from pandas_datareader import data,wb
# aapl = data.DataReader("AAPL",'yahoo','2016-10-01','2016-10-09')
# print(aapl.head())

# import Quandl
# dji = Quandl.get("YAHOO/INDEX_DJI",trim_start='2016-01-01',trim_stop='2016-08-26')
# print(dji.tail(10))

# Import data from csv
#data = pd.read_csv('sp500.csv',index_col='Symbol',usecols=[0,2,3,7])

# Export data to csv
# print(data)
# data.to_csv('test.csv')

# Module 6 Time Series

# import datetime
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pytz
# import random

# Date and Time
# =============

# print(datetime.datetime(2000, 1, 1))
# print(datetime.datetime.strptime("2000/1/1", "%Y/%m/%d"))
# print(datetime.datetime(2000, 1, 1, 0, 0).strftime("%Y%m%d"))

# to_datetime
# ===========

# print(pd.to_datetime("4th of July"))
# print(pd.to_datetime("13.01.2000"))
# print(pd.to_datetime("7/8/2000"))
# print(pd.to_datetime("7/8/2000", dayfirst=True))
# print(issubclass(pd.Timestamp, datetime.datetime))

# ts = pd.to_datetime(946684800000000000)

# print(ts.year, ts.month, ts.day, ts.weekday())

# index = [pd.Timestamp("2000-01-01"),
#          pd.Timestamp("2000-01-02"),
#          pd.Timestamp("2000-01-03")]

# ts = pd.Series(np.random.randn(len(index)), index=index)
# print(ts)
# print(ts.index)

# ts = pd.Series(np.random.randn(len(index)),
#                index=["2000-01-01", "2000-01-02", "2000-01-03"])
# print(ts.index)

# index = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"])
# ts = pd.Series(np.random.randn(len(index)), index=index)
# print(ts.index)

# print(pd.date_range(start="2000-01-01", periods=3, freq='H'))
# print(pd.date_range(start="2000-01-01", periods=3, freq='T'))
# print(pd.date_range(start="2000-01-01", periods=3, freq='S'))
# print(pd.date_range(start="2000-01-01", periods=3, freq='B'))
# print(pd.date_range(start="2000-01-01", periods=5, freq='1D1h1min10s'))
# print(pd.date_range(start="2000-01-01", periods=5, freq='12BH'))

# bh = pd.tseries.offsets.BusinessHour(start='07:00', end='22:00')
# print(bh)


# print(pd.date_range(start="2000-01-01", periods=5, freq=12 * bh))
# print(pd.date_range(start="2000-01-01", periods=5, freq='W-FRI'))
# print(pd.date_range(start="2000-01-01", periods=5, freq='WOM-2TUE'))


# s = pd.date_range(start="2000-01-01", periods=10, freq='BAS-JAN')
# t = pd.date_range(start="2000-01-01", periods=10, freq='A-FEB')
# s.union(t)
# index = pd.date_range(start='2000-01-01', periods=200, freq='B')
# print(index)

# ts = pd.Series(np.random.randn(len(index)), index=index)
# walk = ts.cumsum()
# walk.plot()
# plt.savefig('random_walk.png')

# print(ts.head())
# print(ts[0])
# print(ts[1:3])
# print(ts['2000-01-03'])
# print(ts[datetime.datetime(2000, 1, 3)])
# print(ts['2000-01-03':'2000-01-05'])
# print(ts['2000-01-03':datetime.datetime(2000, 1, 5)])
# print(ts['2000-01-03':datetime.date(2000, 1, 5)])
# print(ts['2000-02'])
# print(ts['2000-03':'2000-05'])

# small_ts = ts['2000-02-01':'2000-02-05']

# print(small_ts)
# print(small_ts.shift(2))
# print(small_ts.shift(-2))

# Downsampling
# ============

# rng = pd.date_range('4/29/2015 8:00', periods=600, freq='T')
# ts = pd.Series(np.random.randint(0, 100, len(rng)), index=rng)

# print(ts.head())
# print(ts.resample('10min').mean().head())
# print(ts.resample('10min').sum().head())
# print(ts.resample('1h').sum().head())
# print(ts.resample('1h').max().head())


# print(ts.resample('1h').apply(lambda m: random.choice(m)).head())
# print(ts.resample('1h', how='ohlc').head())

# Upsampling
# ==========

# rng = pd.date_range('4/29/2015 8:00', periods=10, freq='H')
# ts = pd.Series(np.random.randint(0, 100, len(rng)), index=rng)

# print(ts.head())
# print(ts.resample('15min'))
# print(ts.head())
# print(ts.resample('15min', fill_method='ffill').head())
# print(ts.resample('15min', fill_method='bfill').head())
# print(ts.resample('15min', fill_method='ffill', limit=2).head())
# print(ts.resample('15min', fill_method='ffill', limit=2, loffset='5min').head())

# tsx = ts.resample('15min')
# print(tsx.interpolate().head())

# Time zone handling
# ==================

# t = pd.Timestamp('2000-01-01')
# print(t.tz is None)

# t = pd.Timestamp('2000-01-01', tz='Europe/Berlin')
# print(t.tz)

# rng = pd.date_range('1/1/2000 00:00', periods=10, freq='D', tz='Europe/London')
# print(rng)


# tz = pytz.timezone('Europe/London')
# rng = pd.date_range('1/1/2000 00:00', periods=10, freq='D', tz=tz)
# print(rng)

# rng = pd.date_range('1/1/2000 00:00', periods=10, freq='D')
# ts = pd.Series(np.random.randn(len(rng)), rng)
# print(ts.index.tz is None)

# ts_utc = ts.tz_localize('UTC')

# print(ts_utc.index.tz)
# print(ts_utc.tz_convert('Europe/Berlin').index.tz)
# print(ts_utc.tz_convert(None).index.tz is None)
# print(ts_utc.tz_localize(None).index.tz is None)

# # Time deltas
# # ===========

# print(pd.Timedelta('1 days'))
# print(pd.Timedelta('-1 days 2 min 10s 3us'))
# print(pd.Timedelta(days=1,seconds=1))
# print(pd.Timedelta(days=1) + pd.Timedelta(seconds=1))
# print(pd.to_timedelta('20.1s'))
# print(pd.to_timedelta(np.arange(7), unit='D'))

# # Time series plotting
# # ====================

# rng = pd.date_range(start='2000', periods=120, freq='MS')
# ts = pd.Series(np.random.randint(-10, 10, size=len(rng)), rng).cumsum()

# print(ts.head())

# plt.clf()
# ts.plot(c='k', title='Example time series')
# plt.savefig('time_series_1.png')

# ts.resample('2A').plot(c='0.75', ls='--')
# ts.resample('5A').plot(c='0.25', ls='-.')

# plt.clf()

# tsx = ts.resample('1A')
# ax = tsx.plot(kind='bar', color='k')
# plt.savefig('time_series_2.png')

# ax.set_xticklabels(tsx.index.year)
# plt.clf()
# ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
# df = df.cumsum()
# df.plot(color=['k', '0.75', '0.5', '0.25'], ls='--')
# plt.savefig('time_series_3.png')


# Module 7 Scikits Learn

#from sklearn import datasets

#iris = datasets.load_iris()
#print(iris)
#from sklearn.cross_validation import train_test_split

#X,y = iris.data,iris.target
#print(y)
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
# print(X_train)

#Step 1: Load the model
#from sklearn import svm
#model = svm.SVC()

# #Step 2: Training
#model.fit(X_train,y_train)

# #Step 3: Testing
# print(model.predict(X_test))
# print(y_test) 

# print(model.score(X_test,y_test))

# Step4 : Deployment
#flower = [[3,2,3,5]]
#print(model.predict(flower))

# x = iris.data[:,0]
# y = iris.data[:,1]

# plt.scatter(x,y,c=iris.target)
# plt.show()

#print(iris.data)
#print(iris.target)

# Machine Learning Exercise
# digits = datasets.load_digits()
# X,y= digits.data,digits.target

# from sklearn.cross_validation import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#Step 1: Load the model
# from sklearn import svm
#from sklearn import neighbors
#model = svm.SVC()
#model = neighbors.KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree')

# #Step 2: Training
# model.fit(X_train,y_train)

# #Step 3: Testing
# print(model.predict(X_test))
# print(y_test) 

# print(model.score(X_test,y_test))

# Step4 : Deployment
#flower = [[3,2,3,5]]
#print(model.predict(flower))

# Regression
# ==========

# X = [[1], [2], [3], [4], [5], [6], [7], [8]]
# y = [1, 2.5, 3.5, 4.8, 3.9, 5.5, 7, 8]

# #plt.clf()
# plt.scatter(X, y, c='0.25')

#plt.savefig('regression_1.png')

# clf = LinearRegression()
# clf.fit(X, y)
# print(clf.coef_)

#
#plt.plot(X, clf.predict(X), '--', color='0.10', linewidth=1)
#plt.savefig('regression_2.png')
# plt.show()
# # K-Means
# # =======

# km = KMeans(n_clusters=3)
# km.fit(iris.data)
# print(km.labels_)
# print(iris.target)

# tr = {1: 0, 2: 1, 0: 2}
# predicted_labels = np.array([tr[i] for i in km.labels_])
# sum([p == t for (p, t) in zip(predicted_labels, iris.target)])

# # PCA
# # ===

# pca = PCA(n_components=2)    
# X = StandardScaler().fit_transform(iris.data)
# Y = pca.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
# clf = SVC()
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))

# clf = SVC()
# scores = cross_val_score(clf, iris.data, iris.target, cv=5)
# print(scores)
# print(scores.mean())