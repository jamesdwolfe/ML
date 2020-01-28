import pandas as pd
import numpy as np
import quandl
import math
import datetime
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

# desired_width=1000
# pd.set_option('display.width',desired_width)
# np.set_printoptions(linewidth=desired_width)
# pd.set_option('display.max_columns',100)

style.use('ggplot')

quandl.ApiConfig.api_key="xqbWeW5uNNCNRNNnZ7sR"
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df))) #predicting days in advance count
forecast_out = 30
df.fillna(-99999,inplace=True) #replace null fields in your data

df['label'] = df[forecast_col].shift(-forecast_out)

######################################################################

X = np.array(df.drop(['label'],1)) #features
X = preprocessing.scale(X) #convert every number to something between 0 and 1, normalization
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y=np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = svm.SVR() #support vector machines, mcuh worse
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
forecast_set = clf.predict(X_lately)

print(forecast_set,accuracy,forecast_out)

df['Forecast'] = np.nan
last_date=df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


