# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 10 Oct 2016

# Module 5 Import/Export Data

import pandas as pd 

# Import data

# data = pd.read_csv('data/ex_data.csv')
# print(data.head())
# print(data.ix[1])
# print(data.columns)
# print(data.index)

# sp500 = pd.read_csv('data/sp500.csv',index_col='Symbol',usecols=[0,2,3,7])
# print(sp500.head())

# Export data
# sp500.to_csv('data/test.csv')

# from pandas_datareader import data,wb

# msft = data.DataReader("MSFT", "yahoo","2017-1-1","2017-1-11")
# print(msft.tail())

# import quandl
# data = quandl.get("FRED/GDP")
# print(data.tail())
