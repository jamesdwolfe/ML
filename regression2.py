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

data=yf.download('PINS')
print(data)

data.Close.plot()
plt.show()