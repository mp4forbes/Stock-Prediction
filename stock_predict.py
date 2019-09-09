#!/usr/local/bin/python3

import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

'''
Confirm the following packages are installed.
531	sudo pip3 install scipy
532	sudo pip3 install sklearn

Loading YahooFinance Dataset
Pandas web data reader (https://pandas-datareader.readthedocs.io/en/latest/) is an extension of pandas library to communicate with most updated financial data. This will include sources as: Yahoo Finance, Google Finance, Enigma, etc.
We will extract Apple Stocks Price using the following codes:
'''
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2019, 9, 6)

df = web.DataReader("AAPL", 'yahoo', start, end)

'''
Feature Engineering
We will use these three machine learning models to predict our stocks: 
Simple Linear Analysis, Quadratic Discriminant Analysis (QDA), and 
K Nearest Neighbor (KNN). But first, let us engineer some features: 
High Low Percentage and Percentage Change.
'''
dfreg = df.loc[:,['Adj Close', 'Volume']]

dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

'''
Pre-processing & Cross Validation
We will clean up and process the data using the following steps before putting them into the prediction models:
Drop missing value
Separating the label here, we want to predict the AdjClose
Scale the X so that everyone can have the same distribution for linear regression
Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
Separate label and identify it as y
Separation of training and testing of model by cross validation train test split
Please refer the preparation codes below.
***Import math
'''
import math
import numpy as np
from sklearn import preprocessing

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

'''
Model Generation — Where the prediction fun starts
But first, let’s insert the following imports for our Scikit-Learn:
'''
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


'''
Simple Linear Analysis & Quadratic Discriminant Analysis
Simple Linear Analysis shows a linear relationship between two or more variables. When we draw this relationship within two variables, we get a straight line. Quadratic Discriminant Analysis would be similar to Simple Linear Analysis, except that the model allowed polynomial (e.g: x squared) and would produce curves.
Linear Regression predicts dependent variables (y) as the outputs given independent variables (x) as the inputs.
We will plug and play the existing Scikit-Learn library and train the model by selecting our X and y train sets. The code will be as following.
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

'''
Evaluation
A simple quick and dirty way to evaluate is to use the score method 
in each trained model. The score method finds the mean accuracy of 
self.predict(X) with y of the test data set.
'''
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)
print('The linear regression confidence is', confidencereg)
print('The quadratic regression 2 confidence is', confidencepoly2)
print('The quadratic regression 3 confidence is', confidencepoly3)
print('The knn regression cnfidence is:', confidenceknn)

'''
This shows an enormous accuracy score (>0.95) for most of the models. 
However this does not mean we can blindly place our stocks. There are 
still many issues to consider, especially with different companies that 
have different price trajectories over time.
For sanity testing, let us print some of the stocks forecast.
'''
forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan
print (forecast_set)

'''
Plotting the Prediction
Based on the forecast, we will visualize the plot with our existing
historical data. This will help us visualize how the model fares to 
predict future stocks pricing.
'''
last_date = dfreg.iloc[-1,:].name
last_unix = last_date
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
