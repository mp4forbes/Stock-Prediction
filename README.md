#Stock Prediction
This project is the second homework assignment for implementing three regression methods for stock prediction.
1) Linear Regression method which ultimately generates a linear best-fit for the scatter plot of price -vs- time
2) Quadratic (Square - 3) and (cube - 3) best fit,  this involves curves but is yet a regressive best fit for the training data
3) KNN Regression

Here are the basic steps:
1)Loading YahooFinance Dataset
Pandas web data reader (https://pandas-datareader.readthedocs.io/en/latest/) is an extension of pandas library to communicate with most updated financial data. This will include sources as: Yahoo Finance, Google Finance, Enigma, etc.
We will extract Apple Stocks Price using the following codes:

2) Feature Engineering
We will use these three machine learning models to predict our stocks:
Simple Linear Analysis, Quadratic Discriminant Analysis (QDA), and
K Nearest Neighbor (KNN).

3)Pre-processing & Cross Validation
First we need to clean up and process the data using the following steps before putting them into the prediction models:
  a) Drop missing value
  b) Separating the label here, we want to predict the AdjClose
  c) Scale the X so that everyone can have the same distribution for linear regression
  d) Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
  e) Separate label and identify it as y
  f) Separation of training (99%) and testing (1%) of model by cross validation train test split

Once you have the data, you can clean and apply the various predictive regression models on a percentage (in our case 99%)   of the data for 
training and confirm accuracy by using the test data (in our case 1%)  for testing.

Output:
This program prints out confidencence levels of each model:
  1) The linear regression confidence is: <0-1>
  2) The quadratic regression 2 confidence is <0-1>
  3) The quadratic regression 3 confidence is', <0-1>
  4) The knn regression cnfidence is:', <0-1>

A plot of the historical data against the future stock prices reserved for testing is generated using matplotlib.
