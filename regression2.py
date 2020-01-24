import pandas as pd
import numpy as np
import quandl
import math
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

desired_width=1000
pd.set_option('display.width',desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',100)

import matplotlib.pyplot as plt
import yfinance as yf

stock_name='VOO'
start_date = '2019-01-01'

data=yf.download(stock_name,start_date)
df = data[['Open','Close','High','Low','Volume']]

df['HL_PCT'] = (df['High']-df['Low'])/df['Low'] * 100
df['PCT_change'] = (df['Close']-df['Open'])/df['Open'] * 100

forecast_col = 'Close'
#forecast_out = int(math.ceil(0.0004*len(df)))
forecast_out=1
df.fillna(-99999,inplace=True)
df['label']=df[forecast_col].shift(-forecast_out)
print(df.tail())
df.dropna(inplace=True)

##################################################

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
print(len(X_train),len(X_test),len(y_train), len(y_test))
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(data)
print("Days in advance =", forecast_out)
print("Accuracy percentage =", accuracy)

clf.fit(X[:-forecast_out],y[:-forecast_out])
y_new = clf.predict(X[-forecast_out:])
print("Todays stock price for",stock_name,"=",data['Close'][-forecast_out])
print("Tomorrows stock price for",stock_name,"=",y_new[0])
percent_change = (y_new[0]-data['Close'][-forecast_out])/data['Close'][-forecast_out]
print("Percentage change =",percent_change)
if percent_change>0:
    print(stock_name,":",percent_change)

#################################################

data.Close.plot()
plt.show()