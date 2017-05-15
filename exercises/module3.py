# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 13 Jan 2017

# Module 4 Pandas

import numpy as np
import pandas as pd

# # Series

# a = [2,5,6,7,3]
# b = [3,4,7,9,10]
# a1 = np.array(a)
# b1 = np.array(b)
# a2 = pd.Series(a,index=['a','b','c','d','e'])
# b2 = pd.Series(b,index=['a','b','c','d','e'])
#print(a1+b1)
# # print(a2)
# # print(b2)
# # print(a2+b2)
#print(a2['c'])

# print(a2.index)
# print(a2.values)

#a = pd.Series(np.random.randn(1000))
#print(a.head())
#print(a.tail(10))
#print(a[300:305])

# Data Frame

# a = [[3,4],[5,6]]
# b = [[6,5],[4,3]]
# a1 = np.array(a)
# b1 = np.array(b)
# print(a1+b1)

# a2 = pd.DataFrame(a, index=[1,2], columns=['a','b'])
# b2 = pd.DataFrame(b, index=[1,2], columns=['a','c'])
# print(a2+b2)

# a = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Belinda'],
# 	'height':[160,155,163],
# 	'age': [40,35,42]
# 	},
# 	columns = ['name','height','age'],
# 	index = ['101','105','108']
# 	)
# print(a)
# print(a.index)
# print(a.columns)
# print(a.values)

# Column data
#print(a['name'])
#print(a.name)
#print(a[[0]])

# Row data
# print(a.ix['105'])
# print(a.ix[1])
# print(a.loc['105'])

# Scalar data
# print(a.ix[1]['height'])
# print(a.ix['105','height'])

# a.index = ['108','105','110']
# print(a)

# Re-index
# a3 = a.reindex(['108','105','110'],fill_value='NA')
# a3 = a.reindex(['108','105','110'])
# print(a3)
# a4 = a3.dropna()
# print(a4)
# a = pd.DataFrame(np.random.randn(20,5))
# print(a.head())
# print(a.tail())


#sp500 = pd.read_csv('data/sp500.csv',index_col='Symbol',usecols=[0,2,3,7])
#print(sp500.head())
#print(sp500[sp500.Price<100])

#print(sp500[(sp500.Price>100) & (sp500.Sector=='Health Care')])


# a = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Belinda'],
# 	'height':[160,155,163],
# 	'age': [40,35,42]
# 	},
# 	columns = ['name','height','age']
# 	)

# b = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Alfred'],
# 	'weight': [55,50,80]
# 	},
# 	columns = ['name','weight']
# 	)

# print(a)
# print(b)

# Inner Join
# c = pd.merge(a,b,on='name',how='inner')
# print(c)

# Left Join
# c = pd.merge(a,b,on='name',how='left')
# print(c)

# Right Join
# c = pd.merge(a,b,on='name',how='right')
# print(c)

# Outer Join
# c = pd.merge(a,b,on='name',how='outer')
# print(c)

# a = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Belinda'],
# 	'height':[160,155,163],
# 	'age': [40,35,42]
# 	},
# 	columns = ['name','height','age']
# 	)

# Data Merge Challenges
# sp888 = pd.read_csv("data/sp500.csv", 
#                     index_col='Symbol', 
#                     usecols=['Symbol', 'Dividend Yield'])

# print(sp888.head())
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
# print(pd.merge([aapl1,msft1],on='Date'))

# Outer Join
# s1 = sp500[sp500.Price>200].reset_index()
# s2 = sp888.head(8).reset_index()
# data = pd.merge(s1,s2,on="Symbol",how="outer")
# print(data)
# print(pd.merge([aapl1,msft1],on='Date'))

# Data Pivot
# s1 = sp500.head(5).reset_index()
# print(s1)
# data = s1.pivot(index='Sector',columns='Symbol',values='Price')
# print(data)

# print(data)
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

# Basic Statistics in Pandas
# a = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Belinda'],
# 	'height':[160,155,163],
# 	'age': [40,35,42]
# 	},
# 	columns = ['name','height','age'],
# 	index = ['101','105','108']
# 	)
# print(a.describe())

a = pd.Series([4,5,7,9,13])
# a1 = a.pct_change()
# print(a1.head())

b = pd.Series([5,7,9,13,15])
print(a.cov(b))
print(a.corr(b))






