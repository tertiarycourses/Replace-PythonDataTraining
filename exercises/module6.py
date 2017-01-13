#import datetime

# date = datetime.datetime(2017,1,13)
# print(date)

import pandas as pd 

#date = pd.to_datetime('2017-1-13')
#print(date)

#date = pd.date_range('2017-1-1',periods=30)
# date = pd.date_range('2017-1-1',periods=12,freq='M')
# print(date)

# import numpy as np
# import matplotlib.pyplot as plt

# date = pd.date_range('4/29/2015 8:00',periods=600,freq='T')

# ts = pd.Series(np.random.randint(0,100,len(date)), index=date)

#print(ts.head())
# ts.plot()
# plt.show()

#Re-sampling
# ts1 = ts.resample('10min').mean()
# ts1.plot()
# plt.show()