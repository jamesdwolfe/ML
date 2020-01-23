import pandas as pd
import numpy as np
import quandl

desired_width=1000
pd.set_option('display.width',desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',100)

quandl.ApiConfig.api_key="xqbWeW5uNNCNRNNnZ7sR"
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(df.head())




