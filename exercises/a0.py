import mysql.connector

#import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pandas_datareader import data,wb
#
# a = pd.Series([1,2,3,4],index=[2,3,4,5])
# b = pd.Series([1,2],index=[3,4])
# print(a+b)
# Pandas

#Time Series

# date = pd.date_range("2016-6-1",periods=30,freq='D')
# a = pd.Series(np.random.randn(len(date)),index=date)
# print(a['2016-6-3':'2016-6-10'])

# Resamling the time series

# Upsampling
# time = pd.date_range('1/1/2016 8:00',periods=600, freq='H')
# a = pd.Series(np.random.randint(0,100,len(time)),index=time)
# b = a.resample('10min').bfill()
# print(b.head(30))

# Downsampling
# b = a.resample('1h').mean()
# print(b.head())
# Import data from the web

# stock = data.DataReader("yhoo","yahoo",'2016-5-1','2016-7-1')

# print(stock.tail())

# Exporting data
# a = pd.DataFrame(np.random.rand(4,2))
# a.columns=['a','b']
# a.to_csv('test.out',sep='\t')

#Aggregation



#b = pd.DataFrame(np.random.rand(4,2))

#create a pivot

# a = pd.read_csv('data/sunshine.tsv', header=None, sep='\t')
# a.columns =['country','city','date','sunshine']
#print(a.describe())


# b= a.groupby('city').apply(lambda x:max(x)-min(x))
# print(b)

# Pivot table

#print(a)
#b = a.pivot('country', ['country','city','date'],'sunshine')

#print(b)

# groupby city and average sun shine 

#Grouping data 


# a = pd.read_csv('data/cities.tsv', header=None, sep='\t')
# a.columns =['date','city','value']

# b= a.groupby('city').max()
# print(b)
# Pivot table

# a = pd.read_csv('data/cities.tsv', header=None, sep='\t')
# a.columns =['date','city','value']
# b = a.pivot('date','city','value')
# print(b)
#print(b.mean(axis=0))

# Merging data 

# DF1 = pd.DataFrame({'a':[1,2,3,4],'b':[2.1,2.2,2.3,2.4]})
# print(DF1)
# DF2 = pd.DataFrame({'a':[1,2,1,2],'b':[4.1,4.2,4.3,4.4]})
# print(DF2)
# print('----- merge on a on the left -----')
# print(pd.merge(DF1,DF2,on='a',how='left'))
# print('---- merge on a on the right ------')
# print(pd.merge(DF1,DF2,on='a',how='right'))
#Append or Concat data
# print(a.append(b))
# print(b.append(a))
#print(pd.concat([a,b]))

#a = pd.Series([10,20,30,40,50])
#a.index = [2,3,4,5,6]
# print(a)
# b = a.reindex([2,3,4,5,6],method='ffill')
# print(b)
# Viewing DataFrame

# Data Filtering
# sp500 = pd.read_csv("data/sp500.csv",index_col='Symbol',usecols=[0,2,3,7])
# print(sp500[(sp500.Sector == 'Health Care') & (sp500.Price>55)])

#DF = pd.DataFrame(np.random.rand(20,5),columns=['a','b','c','d','e'])


# Selecting Rows
#DF = pd.DataFrame(np.random.rand(20,5),columns=['a','b','c','d','e'])
# b= DF.c
# print(b[b>0.5])
#print(DF[-2:])
#print(DF.ix[[3,4,2],])
# print(DF.loc[3])

# Selecting Columns
#DF = pd.DataFrame(np.random.rand(20,5),columns=['a','b','c','d','e'])
#print(a.ix[[3,5],['a','c']])
#print(a[['c']])
#print(DF.c)

# Creating DataFrame

# a = pd.Series(np.random.randn(5))
# b = pd.Series(np.random.randn(5))
# c = pd.DataFrame({1:a,2:b})

# d = pd.Series(np.random.randn(5))
# c[3] = d
# print(c)

# DataFrame
# a = [[1,2],[3,4]]
# b = [[4,3],[2,1]]
# a1 = pd.DataFrame(a,index=[1,2],columns=['a','b'])
# b1 = pd.DataFrame(b,index=[3,2],columns=['b','a'])
# print(a1)
# print(b1)
# c = a1+b1
# print(c)
# d = c.dropna()
# print(d)

#a = pd.Series(np.random.randn(20))
# print(a[:9])

# View Pandas Series data
# a = pd.Series(np.random.randn(20))
# print(a.values)
#print(a.tail(10))

# Pandas Series Creation

# a = pd.Series({'a':1,'b':2,'c':3,'d':4})
# print(a)

# a = pd.Series(np.random.randn(20))
# print(a)
# a = [1,2,3,4,6,8,9]
# b = [4,3,2,1]
# a1 = np.array(a)
# b1 = np.array(b)
#print(a1+b1)

# a2 = pd.Series(a,index=[1,2,3,4,5,6,7])
# b2 = pd.Series(b,index=[1,2,3,4])
# print(a2+b2)

# Histogram

# x = np.random.normal(0,1,1000)
# plt.hist(x)
# plt.show()
# Contour Plot

# x = np.linspace(-1,1,255)
# y = np.linspace(-2,2,300)

# X,Y = np.meshgrid(x,y)

# z = np.sin(X)*np.cos(Y)
# plt.contour(X,Y,z,255)
# plt.show()
# Bar chart

# x =np.arange(0,10,1)
# y = x + np.random.randint(10)
# plt.bar(x,y)
# plt.show()

# Scatter plot
# x = np.arange(0,30,2)
# y = x + 2*np.random.normal(0,1,len(x))
# plt.scatter(x,y)
# plt.show()
# Simple plot
# x = np.linspace(0,4*np.pi,200)
# y = np.sin(x)
# y2 = np.cos(x)
# y3 = y*y2
# y4 = y*y-y2*y2
# plt.subplot(2,2,1)
# plt.plot(x,y,color='darkgreen',marker='o',linestyle='-',label='sine')
# plt.subplot(2,2,2)
# plt.plot(x,y2,color='#ee33dd',marker='^',linestyle='-',label='cosine')
# plt.subplot(2,2,3)
# plt.plot(x,y3,color='darkgreen',marker='o',linestyle='-',label='sine')
# plt.subplot(2,2,4)
# plt.plot(x,y4,color='#ee33dd',marker='^',linestyle='-',label='cosine')

# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('sine and cosine curves')
# plt.axis([0,4*np.pi,-2,2])
# plt.legend(loc='upper left')
# plt.grid()
# plt.show()

# Random number

#print(np.random.randint(6))
# Linear Alegbra

# A = np.array([[2,3],[3,-1]])
# b = np.array([12,7]).T
# x = np.linalg.solve(A,b)
# print(x)
# a = np.array([[1,4,5],[5,2,2],[-1,6,8]])
# w,v = np.linalg.eig(a)
# print(w)
# print(v)



#a = [2,4,-6,8,10,-12,14,-16,18]
# print(a>0)

# Logical indexing
# b = np.array(a)
# b = b[b>0]
# b = b[b>8]
# print(b)

# Indexing & Slicing
# a = np.arange(0,24,2).reshape(-1,3)
# print(a)

# Numpy basic stats
# print(a[2:,[0,2]])
# print(np.sum(a,axis=1))

#a = np.arange(4,30,3).reshape(3,3)
# a = np.linspace(4,40,10).reshape(2,-1)
# b = a.ravel()
# print(a)
# print(b)

# print(np.full((5,4),10))

# def f(x,y):
# 	return 10*x*y+3

# b= np.fromfunction(f,(5,4),dtype=int)
# print(b)

# a = [[1,1],[1,1],[1,1]]
# a1 = np.array(a,dtype=np.float32)
# print(len(a1))
# print(a1.shape)
# print(a1.dtype)

# b1 = a1.astype(np.float64)
# print(b1.dtype)

# b = [2,2,2,2,2]
# b = [2,2,2,2,2]
# print(a)

#a1 = np.matrix(a,dtype=np.float32)
# b1 = np.matrix(b,dtype=np.float64)
# c1 = a1+b1
# print(c1.dtype)
# print(a1/b1)