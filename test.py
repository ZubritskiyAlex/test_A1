import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error

register_matplotlib_converters()


df = pd.read_csv('Timeseries.csv',nrows=1645 )
df.info()

timeseries = df[0:1644]
test = df[1550:]

dd = np.asarray(timeseries.series1)
lastvalue = dd[len(dd)-1]
y_hat = test.copy()
y_hat['naive'] = lastvalue
y_hat.describe()


plt.figure(figsize=(15, 9))
plt.plot(timeseries.index, timeseries['series1'], label='Timeseries')
plt.plot(test.index, test['series1'], label='Test')
plt.legend(loc='best')
plt.title('Naive Forecast')
plt.show()

