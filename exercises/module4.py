# Code guide for Python Data Analysis
# Copyright: Tertiary Infotech Pte Ltd
# Author: Dr Alfred Ang
# Date: 13 Jan 2017

# Module 3 Matplolib

# import numpy as np 
# import matplotlib.pyplot as plt

# x = np.linspace(-4,4,100)
# y = np.sin(x)
# y2 = np.cos(x)
# y3 = y*y2
# y4 = y*y -y2*y2
#plt.plot(x,y,color='#334411',marker='o',linestyle='-')
#plt.plot(x,y,'ro-',label='sine',x,y2,'g^-',label='cosine')
# plt.subplot(2,1,1)
# plt.plot(x,y,'ro-',label='sine')
# plt.subplot(2,1,2)
# plt.plot(x,y2,'g^-',label='cosine')
# plt.grid()
#plt.legend(loc='upperleft')
# plt.legend(bbox_to_anchor=(1.1,1.05))
# plt.xlabel('x')
# plt.ylabel('y')
#plt.title('sine curve')
# plt.show()

# Challenge
# plt.subplot(2,2,1)
# plt.plot(x,y,'ro')
# plt.subplot(2,2,2)
# plt.plot(x,y2,'g^')
# plt.subplot(2,2,3)
# plt.plot(x,y3,'b^')
# plt.subplot(2,2,4)
# plt.plot(x,y4,'ko')
# plt.show()

# Other Plots

# Scatter Plot
# x = np.linspace(0,10,200)
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

