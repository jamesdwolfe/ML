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

quandl.ApiConfig.api_key="xqbWeW5uNNCNRNNnZ7sR"
df = quandl.get('WIKI/GOOGL')
print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.0027*len(df))) #predicting days in advance count
df.fillna(-99999,inplace=True) #replace null fields in your data

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

######################################################################

X = np.array(df.drop(['label'],1)) #features
y= np.array(df['label']) #labels

X = preprocessing.scale(X) #convert every number to something between 0 and 1, normalization
y=np.array(df['label'])
#print(len(X),len(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.6)
clf = svm.SVR() #support vector machines, mcuh worse
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
print(forecast_out)
print(accuracy)

