import random
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import statistics
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

seed = 577
valuedf = pd.read_csv("VALUE.csv")

######################
# Question 1         #
######################

# 1a
valuedf.plot.scatter(x = 'Gender', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'Married', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'Income', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'FirstPurchase', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'LoyaltyCard', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'WalletShare', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'TotTransactions', y = 'CustomerLV', s = 100)
plt.show(block=True)
valuedf.plot.scatter(x = 'LastTransaction', y = 'CustomerLV', s = 100)
plt.show(block=True)

# No obvious significant response is visible with changes to the Gender, Married, Income, & LoyaltyCard.
# A slight positive correlation between FirstPurchase and the response is possible.
# A strong positive correlation is noticeable between CustomerLV and WalletShare. Increasing levels
# of totTransactions also relates to a strong positive response. Finally, there is a noticeable positive correlation
# between CustomerLV and LastTransaction, as well as an increase in variance of the response alongside an increase in
# this predictor

# 1b
predictor_list = ['Gender', 'Married', 'Income', 'FirstPurchase', 'LoyaltyCard', 'WalletShare', 'TotTransactions',
                  'LastTransaction']

ohe = OneHotEncoder(sparse = False)
valuedf['Gender'] = ohe.fit_transform(valuedf[['Gender']])
valuedf['Married'] = ohe.fit_transform(valuedf[['Married']])
valuedf['LoyaltyCard'] = ohe.fit_transform(valuedf[['LoyaltyCard']])
le = preprocessing.LabelEncoder()
valuedf['Income'] = le.fit_transform(valuedf['Income'])

x = valuedf.loc[:,predictor_list]
y = valuedf['CustomerLV']

model = LinearRegression()
scores = []
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 577)
model.fit(x,y)
intercept = model.intercept_
coefficients = model.coef_
print("intercept: " + str(intercept))
print(list(zip(model.coef_, predictor_list)))
print("coefficients: ")
print(coefficients)

# Encoding key:
# Gender: Male = 0, Female = 1
# Married: Single = 0, Married = 1
# LoyaltyCard: Yes = 0, No = 1
#
# After running the full multiple regression, we see that an increasing number of transactions predicts an increasing
# CustomerLV in this model.
# Using One Hot Encoding, Male = 0 and Female = 1. Therefore, in this model Men predict a positive response in
# CustomerLV.
# Finally, Income30t45 was encoded as 3 on a scale from 0 to 5, which puts it slightly on the higher end.
# Therefore, considering the fairly high positive correlation between this predictor and the response, we can
# conclude that the Income30t45 would predict a CustomerLV higher than average, all other predictors being equal.

# 1c

