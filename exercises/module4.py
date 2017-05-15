# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 13 Jan 2017

# Module 4 Basic of Data Visualization

# Setup
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# A simple demo of matplot
# x = np.linspace(-4,4,100)
# y = np.sin(x)
# plt.plot(x,y,color='#334411',marker='o',linestyle='-')
# plt.plot(x,y,'ro-',label='sine',x,y2,'g^-',label='cosine')
# plt.subplot(2,1,1)
# plt.plot(x,y,'ro-',label='sine')
# plt.subplot(2,1,2)
# plt.plot(x,y2,'g^-',label='cosine')
# plt.grid()
# plt.legend(loc='upperleft')
# plt.legend(bbox_to_anchor=(1.1,1.05))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('sine curve')
# plt.show()

# Scatter Plot
# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','wt'])
# plt.scatter(df.mpg,df.wt)
# plt.show()

# Multple Plots
# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','wt'])
# plt.plot(df[['mpg','cyl','wt']])
# plt.show()

# Bar plot
# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','wt'])
# mpg = df['mpg']
# mpg.plot(kind='bar')
# plt.show()

# Pie Chart
# x = [1,2,3,4,0.5]
# plt.pie(x)
# type = ['bicycle', 'motorbike','car', 'van', 'stroller']
# plt.pie(x, labels= type)
# plt.show()

# Histogram
# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','wt'])
# mpg = df['mpg']
# plt.hist(mpg)
# sb.distplot(mpg)
# plt.show()

# Scatter Plot with Regression
# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','hp'])
# df.plot(kind='scatter', x='hp', y='mpg', c=['darkgray'], s=150)
# sb.regplot(x='hp', y='mpg', data=df, scatter=True)
# plt.show()

# Pair Plot
#df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','hp'])
#sb.pairplot(df)
#plt.show()

# Box Plot
# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','wt','am'])
# df.boxplot(column='mpg', by='am')
# df.boxplot(column='wt', by='am')
# plt.show()

# Object-Oriented Plotting
# x = range(1,10)
# y = [1,2,3,4,0,4,3,2,1]
# fig = plt.figure()
# ax = fig.add_axes([.1, .1, 0.8, 0.8])
# ax.set_xlim([1,9])
# ax.set_ylim([0,5])
# ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
# ax.set_yticks([0,1,2,3,4,5])
# ax.grid()
# ax.plot(x,y)
# plt.show()

# Subplot
# x = range(1,10)
# y = [1,2,3,4,0,4,3,2,1]
# fig = plt.figure()
# fig,(ax1,ax2) = plt.subplots(1,2)
# ax1.plot(x)
# ax2.plot(x,y)
# plt.show()

# Limits, Ticks, Grid, Colors, Linestyles, Linewidth
# x = range(1,10)
# y = [1,2,3,4,0,4,3,2,1]
# fig = plt.figure()
# ax = fig.add_axes([.1, .1, 0.8, 0.8])
# ax.set_xlim([1,9])
# ax.set_ylim([0,5])
# ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])
# ax.set_yticks([0,1,2,3,4,5])
# ax.grid()
# ax.plot(x,y)
# ax.plot(x,y,color='salmon')
# ax.plot(x,y, ls='--', lw=2)
# plt.show()

# Challenge - Object Orient Plottin
# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','wt'])
# df = df[['mpg','cyl','wt']]
# color_theme = ['darkgray', 'lightsalmon', 'powderblue']
# df.plot(color=color_theme)
# plt.show()


# Label, Title
# df = pd.read_csv("data/mtcars.csv",usecols=['car_names','mpg','cyl','wt'])
# mpg = df['mpg']
# fig = plt.figure()
# ax = fig.add_axes([.1, .1, 0.8, 0.8])
# mpg.plot()

# Time Series
# from pandas_datareader import data,wb
# aapl = data.DataReader("AAPL", 'yahoo', '2016-1-1', '2016-8-17')
# ts = aapl[['Close']]
# ts.plot()
# plt.show()

# Challenge Time Series Plot
# df = pd.read_csv('data/aapl.csv',index_col='Date')
# df2 = df['Close']
# df3 = df2.rolling(window=20).mean()
#df2 = df2.sample(frac=0.1)
# df2.plot()
# df3.plot()
# plt.show()

# y = x + np.random.randn(len(x))
# plt.scatter(x,y)
# plt.show()

# Bar Plot
# Horizontal Bar Plot
# people = ['Tom', 'Dick', 'Harry', 'Slim', 'Jim']
# height = 170 + 20 * np.random.randn(len(people))
# x = np.arange(len(people))
# plt.barh(x,height,align="center",color='yellow')
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
# plt.hist(x,10)
# plt.show()

# Contour Plot
# x = np.linspace(-1,1,255)
# y = np.linspace(-2,2,300)
# X,Y = np.meshgrid(x,y)
# z = np.sin(X)*np.cos(Y)
#plt.contour(X,Y,z,10)
# plt.show()

# Pie Plot
# x = [45,50,20]
# plt.pie(x)
# plt.show()

