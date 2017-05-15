# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 13 Jan 2017

# Module 3 Data Analysis with Pandas

import numpy as np
import pandas as pd

# Pandas Series
# Create Pandas Series from List
# a = [3,4,5,6]
# s = pd.Series(a)
# s = pd.Series(a,index=['A1','A2','A3','A4'])
# print(s)

# Create Pands Series from Numpy Array
# a = np.random.randn(100)*5+100
# date = pd.date_range('20170101',periods=100)
# s = pd.Series(a,index=date)
# print(s)

# Create Pandas Series from Dictionary
# a = {'A1':5,'A2':3,'A3':6,'A4':2}
# s = pd.Series(a)
# print(s)

# Pandas Series Arithematic
# a = [2,5,6,7,3]
# b = [3,4,7,9,10]
# s1 = pd.Series(a,index=['a','b','c','d','e'])
# s2 = pd.Series(b,index=['a','b','c','d','e'])
# print(s1+s2)

# Pandas Series Attributes
# s = pd.Series({'a': 1, 'b': 3, 'c': 5, 'd': 6})
# print(s.index)
# print(s.values)
# print(len(s))


# Viewing Pandas Series Data
# a = np.random.randn(100)*5+100
# date = pd.date_range('20170101',periods=100)
# s = pd.Series(a,index=date)
#print(s.head())
#print(s.tail(10))
#print(s[30:35])

# Selecting Pandas Series Data
# s = pd.Series({'a': 1, 'b': 3, 'c': 5, 'd': 6})
# print(s['b'])
# print(s['b'])
# print(s[1])
# print(s[[1]])
# print(s[['b','d']])
# print(s[[1,3]])

# Slicing Pandas Series Data
# print(s[1:4])
# print(s[:4])
# print(s[2:])

# Challenge
# date = pd.date_range('20170101',periods=20)
# s = pd.Series(np.random.randn(20),index=date)
# print(s['20170105':'20170110'])

# Pandas DataFrame
# Create Pandas DataFrame from List
# d = [[1,2],[3,4]]
# df = pd.DataFrame(d1, index=[1,2], columns=['a','b'])

# Create Pandas DataFrame from List
# d = [[1,2],[3,4]]
# df = pd.DataFrame(d1, index=[1,2], columns=['a','b'])

# Create Pandas DataFrame from Numpy Array
# d = np.arange(24).reshape(6,4)
# col_name = ['Q1','Q2','Q3','Q4']
# df = pd.DataFrame(d,index=list('ABCDEF'),columns=col_name)
# print(df)


# Create Pandas DataFrame from Dictionary
# df = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Belinda'],
# 	'height':[160,155,163],
# 	},
# 	columns = ['name','height'],
# 	index = ['A1','A2','A3']
# 	)
# print(df)

# Create Pandas DataFrame from Series
# date = pd.date_range('20170101',periods=6)
# s1 = pd.Series(np.random.randn(6),index=date)
# s2 = pd.Series(np.random.randn(6),index=date)
# df = pd.DataFrame({'Asia':s1,'Europe':s2})
# print(df)


# Challenge
# df = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Belinda'],
# 	'height':[160,155,163],
# 	'age': [40,35,42]
# 	},
# 	columns = ['name','height','age'],
# 	index = ['A1','A2','A3']
# 	)
# print(df)

# DataFrame Attributes
# print(df.index)
# print(df.columns)
# print(df.values)


# Append and Insert Columns
# d = [[1,2],[3,4]]

# df = pd.DataFrame(d, index=[1,2], columns=['a','b'])

# df['c']=[9,10]
# print(df)

# df.insert(1,'d',[11,12])
# print(df)

# Arithematics

# d1 = np.arange(12).reshape(4,-1)
# d2 = np.arange(0,24,2).reshape(3,-1)
# df1 = pd.DataFrame(d1)
# df2 = pd.DataFrame(d2)
# print(df1)
# print(df2)
# print(df1+df2)
	
# Selecting Column data
# df = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Belinda'],
# 	'height':[160,155,163],
# 	'age': [40,35,42]
# 	},
# 	columns = ['name','height','age'],
# 	index = ['A1','A2','A3']
# 	)
# print(df[['name','height']])
# print(df.name)
# print(df[[1,2]])

# Selecting Row data
# print(df.ix[0])
# print(df.ix['A1'])

# print(df.ix[['A1','A2']])
# print(df.ix[[0,1]])

# print(df.ix[1:3])
# print(df.ix[1:])
# print(df.ix[:3])

# Scalar data
# print(df.ix['A1']['height'])
# print(df.ix[1]['height'])
# print(df.ix[['A1','A2'],['name','height']])

# a.index = ['108','105','110']
# print(a)


# Import/Export data

# Import data from csv file
# sp500 = pd.read_csv('data/sp500.csv',index_col ='Symbol', usecols=['Symbol','Sector','Price'])
# print(sp500.head)
# print(sp500.Price)

# Import data from excel file
# pip install xlrd
sp500 = pd.read_excel('data/sp500.xlsx',index_col='Symbol',usecols=[0,2,3,7])
print(sp500.head())


# Import data from internet
# pip install pandas-datareader
# from pandas_datareader import data,wb
# msft = data.DataReader("MSFT", "yahoo","2017-1-1","2017-1-11")
# print(msft.tail())

# import quandl
# d = quandl.get("FRED/GDP")
# print(d.tail())

# d = quandl.get("YAHOO/INDEX_DJI",trim_start='2016-01-01',trim_stop='2016-08-26')
# print(d.tail(10))

# Export data to csv file
# sp500.to_csv('data/test.csv')


# Filtering Data

# Filtering Pandas Series
# date = pd.date_range('20170101',periods=10)
# d = np.random.randn(10)*4+5
# ts = pd.Series(d,index=date)
# print(ts[ts>5])

# Filtering Pandas DataFrame
# sp500 = pd.read_excel('data/sp500.xlsx',index_col ='Symbol', usecols=['Symbol','Sector','Price'])
# print(sp500.Price>100)
# print(sp500[sp500.Price == sp500.Price.max()])

# Challenge
#sp500 = pd.read_csv('data/sp500.csv',index_col='Symbol',usecols=[0,2,3,7])
#print(sp500[(sp500.Price>100) & (sp500.Sector=='Health Care')])

# Missing Data
# missing = np.nan
# s = pd.Series([3,4,missing,6,missing,8])
# print(s.isnull())

# np.random.seed(25)
# df = pd.DataFrame(np.random.randn(36).reshape(6,6))
# df.ix[3:5, 0] = missing
# df.ix[2:4, 5] = missing
# print(df)
# print(df)
# df2 = df.fillna(0)
# df2 = df.fillna({0: 0.1, 5: 1.25})
# print(df2)
# df2 = df.fillna(method='ffill')
# print(df2)
# df2 = df.fillna(method='bfill')
# print(df2)

# Counting the number of missing data
# print(df.isnull().sum())

# Filtering out the missing data
# df2 = df.dropna()
# df2 = df.dropna(axis=1)
# df2 = df.dropna(how='all')
# print(df2)


# Duplicate Data
# df = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 3],
#                    'B': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
#                    'C': ['A', 'A', 'B', 'B', 'C', 'C', 'C']})
# print(df)
# print(df.duplicated())
# df2 = df.drop_duplicates()
# print(df2)

# df = pd. DataFrame({'c1': [1, 1, 2, 2, 3, 3, 3],
#                   'c2': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
#                   'c3': ['A', 'A', 'B', 'B', 'C', 'D', 'C']})
# print(df)
# df2 = df.drop_duplicates('c3')
# print(df2)
# Re-index
# a3 = a.reindex(['108','105','110'],fill_value='NA')
# a3 = a.reindex(['108','105','110'])
# print(a3)
# a4 = a3.dropna()
# print(a4)
# a = pd.DataFrame(np.random.randn(20,5))
# print(a.head())
# print(a.tail())

# Concat 2 Series
# s1 = pd.Series(['a', 'b'])
# s2 = pd.Series(['c', 'd'])
# s3 = pd.concat([s1, s2])
# s3 = pd.concat([s1, s2], ignore_index=True)
# s3 = pd.concat([s1, s2], keys=['s1', 's2'])
# print(s3)


# Concat 2 DataFrames

# df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['A', 'B'])
# df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['A', 'B'])
# df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['A', 'C'])
# df3 = pd.concat([df1,df2],join="inner")
# df3 = pd.concat([df1,df2],join="outer")
# df3 = pd.concat([df1,df2],axis=1)
# df3 = pd.concat([df1, df2], keys=['s1', 's2',])
# df3 = pd.concat([df1,df2],ignore_index=True)
# print(df3)

# Concat 2 DataFrames - 2nd example
# d1 = np.arange(24).reshape(6,4)
# dates = pd.date_range('20170101',periods=6)
# df1 = pd.DataFrame(d1,index=dates)
# print(df1)
# d2 = np.arange(12).reshape(3,4)
# dates = pd.date_range('20170201',periods=3)
# df2 = pd.DataFrame(d2,index=dates)
# print(df2)
# df3 = pd.concat([df1,df2],axis=0,keys=['Jan','Feb'])
# print(df3)


# Challenge: Concat
#sp500 = pd.read_csv('data/sp500.csv',index_col='Symbol',usecols=[0,2,3,7])

# Append 2 DataFrames
# df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['A', 'B'])
# df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['A', 'C'])
# df3 = df1.append(df2)
# df3 = df1.append(df2,ignore_index=True)
# print(df3)

# Join 2 DataFrames
# df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['A', 'B'])
# df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['C', 'D'])
# df3 = pd.DataFrame.join(df1,df2)
# print(df3)

# Merge 2 DataFrames
# df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['A', 'B'])
# df2 = pd.DataFrame([['a', 3], ['d', 4]], columns=['A', 'B'])

# Inner Join
# df3 = df1.merge(df2,on='A',how='inner')
# print(df3)

# Outer Join
# df3 = df1.merge(df2,on='A',how='outer')
# print(df3)

# Left Join
# df3 = df1.merge(df2,on='A',how='left')
# print(df3)

# Right Join
# df3 = df1.merge(df2,on='A',how='right')
# print(df3)

# df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['A', 'B'])
# df2 = pd.DataFrame([['a', 3], ['d', 4]], columns=['C', 'D'])

# Left and Right On
# df3 = df1.merge(df2,left_on='A',right_on="C",how='inner')
# print(df3)


# Data Merge Challenge
# df1 = pd.read_csv("data/sp500.csv", 
#                     index_col='Symbol', 
#                     usecols=['Symbol','Price'])
# df2 = pd.read_csv("data/sp500.csv", 
#                     index_col='Symbol', 
#                     usecols=['Symbol', 'Dividend Yield'])


# df1 = df1.ix[:10].reset_index()
# df2 = df2.ix[5:15].reset_index()
# df3 = df1.merge(df2,on="Symbol",how="outer")
# print(df3)

#Dropping Data

# d = np.arange(24).reshape(6,4)
# df = pd.DataFrame(d,columns=['a','b','c','d'])
# print(df)
# df2 = df.drop([2,4])
# print(df2)
# df2 = df.drop(['b','d'],axis=1)
# print(df2)

#Sorting Data
# np.random.seed(25)
# d = np.random.randn(24).reshape(12,2)
# df = pd.DataFrame(d,columns=['a','b'])
# df2 = df.sort_values(by=['a'])
# print(df2)

# Challenge: Sorting data
# df = pd.read_csv("data/sp500.csv", 
#                     index_col='Symbol', 
#                     usecols=['Symbol','Price','Dividend Yield'])

# df2 = df.sort_values(by=['Price'])
# df2 = df.sort_values(by=['Price'],ascending=[False])
# print(df2)

# Grouping and Aggregating Data

# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','hp'])
# print(df.head)

# df2 = df.groupby('cyl')
# print(df2.mean())

# Aggregating data with groupby
# df=pd.read_csv('data/sunshine.tsv',header=None,sep="\t")
# df.columns=['country','city','date','hours']

# print(df.groupby('city').describe())
# print(df.groupby('city').mean())
# print(df.groupby(['country','date']).describe())
# print(df.groupby('city').agg(np.mean))
# print(df.groupby('city').agg(lambda a:max(a)-min(a)))

# Challenge: Groupby 
# df=pd.read_csv('data/cities.tsv',header=None, sep='\t')
# df.columns = ['date','cities','temperature']
# print(df)
# df2 = df.groupby('cities')
# print(df2.max())

# df = pd.DataFrame({'foo': ['one','one','one','two','two','two'],
#                        'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
#                        'baz': [1, 2, 3, 4, 5, 6]})
# p = df.pivot(index='foo', columns='bar', values='baz')
# print(p)

# Pivot Data
# df = pd.read_csv("data/sp500.csv", usecols=['Symbol','Price','Sector'])
# p = df.pivot(index='Sector',columns='Symbol',values='Price')
# print(p2)

# df=pd.read_csv('data/cities.tsv',header=None, sep='\t')
# df.columns = ['date','cities','temperature']
# p = df.pivot(index='date',columns='cities',values='temperature')
# print(p)


# Challenge: Pivot Data
# aapl = pd.read_csv('data/aapl.csv',index_col='Date',usecols=['Date','Volume','Close'])
# msft = pd.read_csv('data/msft.csv',index_col='Date',usecols=['Date','Volume','Close'])

# df = pd.concat([aapl,msft],keys=['APPL','MSFT']).reset_index()
# df.columns = ['Symbol','Date','Close','Volume']
# p = df.pivot(index='Date',columns='Symbol',values='Close')
# print(p)


# Basic Statistics in Pandas
# df = pd.DataFrame(
# 	{
# 	'name': ['Ally','Jane','Belinda'],
# 	'height':[160,155,163],
# 	'age': [40,35,42]
# 	},
# 	columns = ['name','height','age'],
# 	index = ['101','105','108']
# 	)

# df = pd.read_csv("data/mtcars.csv",index_col='car_names',usecols=['car_names','mpg','cyl','hp','wt'])
# print(df.head)
# print(df.sum(axis=1))
# print(df.mean())

# print(df.describe())
# s1 = pd.Series([4,5,7,9,13])
# print(s1.pct_change())

#s2 = pd.Series([5,7,9,13,15])
#print(s1.cov(s2))
#print(s1.corr(s2))