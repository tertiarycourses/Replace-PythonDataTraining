import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pandas_datareader import data, wb
import datetime
import pytz
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [1, 2.5, 3.5, 4.8, 3.9, 5.5, 7, 8]

plt.clf()
plt.scatter(X, y, color='blue')
#plt.savefig('regression_1.png')

clf = LinearRegression()
clf.fit(X, y)
print(clf.coef_)

#plt.clf()
plt.plot(X, clf.predict(X), '--', color='red', linewidth=1)
plt.show()
#plt.savefig('regression_2.png')


# data = {'Age':[24,26,25,36],'Height':[160,155,180,150]}
# label = ['Ally','Belinda','Jane','Alfred']
# df = pd.DataFrame(data,index=label)
# df.plot(kind='bar',subplots=True,sharex=True)
# plt.show()
#date = datetime.datetime(2016,5,5)

# date = pd.to_datetime('2016-05-01')
# date = pd.date_range('2015-05-01',periods=100,freq='H')
# ts = pd.Series(np.random.randint(0,100,len(date)),index=date)
# ts.cumsum().plot(ls='--')
# plt.show()
#print(ts.head(20))
#print(ts.resample('15min').ffill(limit=2).head(10))

# msft = data.DataReader("MSFT", 'yahoo', '2016-4-1', '2016-5-1')
# print(msft.tail())

# sp500 = pd.read_csv('data/sp500.csv',index_col='Symbol',usecols=[0,2,3,7])
# print(sp500[(sp500.Sector=='Health Care')&(sp500.Price>100)])

# df=pd.read_csv('data/sunshine.tsv',header=None, sep='\t')
# df.columns=['country','city','date','hours']
# print(df.hours.apply(np.cumsum))
#p = df.pivot('date','city','temp')
# p = df.groupby('country').agg(lambda a: max(a)-min(a))
# print(p)
#print(p.mean(axis=0))

#a = pd.DataFrame({'A':[2,6,4,8],'B':[1,9,5,3]})
# a = pd.Series([2,6,4,8,1,9,5,3],index=[3,4,1,2,9,10,4,6])
# print(a)
# print(a.sort_index())
#print(a.apply(lambda a:a**2))
# b = pd.DataFrame({'A':[1,6,8,10],'B':[3,4,7,8]})
# print(b)
# print(pd.merge(a,b,on='A',how='left'))

# a = pd.Series(np.arange(10),index=[1,2,3,4,5,6,7,8,9,10])
# a2 = a.reindex([1,2,3,11,5,6,12,8,9,13])
# print(a2.dropna())

# data = pd.DataFrame(np.random.randn(10,5),columns=['a','b','c','d','e'])
# print(data.head())
# print(data.at[3,'d'])

#print(a.ix[1:5,['d','a']])

# a1 = pd.Series(np.random.randn(20))
# a2 = pd.Series(np.random.randn(20))
# dates = pd.date_range('20160506',periods=6)
# a = pd.DataFrame(np.random.randn(6),index=dates,columns=['a'])
# a['b']=np.random.randn(6)
# print(a)

# x = np.array([4,2,6,7,3])
# plt.pie(x)
# plt.show()
# x = np.random.randn(2000)
# plt.hist(x,100)
# plt.show()

# x = np.linspace(-1,1,255)
# y = np.linspace(-2,2,300)
# X, Y = np.meshgrid(x, y)

# z = np.sin(X)*np.cos(Y)
# plt.plot_trisurf(X,Y,z)
# plt.show()

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

# print(a.append(b))
# x = np.arange(5)
# y = np.array([2.4,5.6,8.3,10.2,12.6])
#plt.scatter(x,y)
# plt.bar(x,y,color='yellow')
# plt.show()

# x = np.linspace(0,4*np.pi,200)
# y = np.sin(x)
# y2 = np.cos(x)
# y3 =  y*y2
# y4 = y**2 - y2**2
# plt.plot(x,y,color='#ff4433',marker='o',linestyle='-',label='sin')
# plt.plot(x,y2,color='#2233ff',marker='^',linestyle='-',label='cos')
# plt.plot(x,y3,color='#2233ff',marker='^',linestyle='-',label='sin.cos')
# plt.plot(x,y4,color='#2233ff',marker='^',linestyle='-',label='sin^2-cos^2')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Sine curve')
# plt.grid()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
# plt.axis([0,2*np.pi,-1,1])
# plt.show()


# b1 = [[3,4],[6,7],[8,9]]
# b2 = [[1,1],[2,2],[3,3]]
# print(b1+b2)

# A = np.array([[1,4,5],[5,2,2],[-1,6,8]])
# print(A)
# w,v = np.linalg.eig(A)
# print(w)
# print(v)

# print(np.random.binomial(100,0.9,3))
# A = np.array([[5,3],[3,-1],[4,6]])
# print(A)
# print(np.sort(A,axis=1))

#print(A)
#b = np.array([12,7]).T
#print(b)
# x = np.linalg.solve(A,b)
# print(x)
#a = np.array([[1,1],[2,2],[3,3]])
# b = np.array([[4,4,],[5,5],[6,6]])
# print(a)
# print(b)
# print(np.hstack([a,b]))
# b = [3,-2,1,6,-2,8,9,-5]
# a = np.array(b)
# print(a[a%3==0])

#Ex: list out all the elements that divisible by 3 in b



# print(a)
# print(a[[1,2],[0,2]])
# a = np.array([[1,1,1],[2,2,2]])
# print(a)
# print(a.std(axis=0))
#a *= np.pi
#a = np.arange(1,41,2).reshape(10,2)
# print(a)
# print(np.cos(a))

# a = np.linspace(1,20,40)
# print(a)
