import random
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import statistics
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

random.seed(577)
valuedf = pd.read_csv("VALUE.csv")

######################
# Question 1         #
######################
'''
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

'''
# 1c
# split the dataset
x = valuedf.loc[:, valuedf.columns!='CustomerLV']
y = valuedf.loc[:,"CustomerLV"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.05, random_state=0)

# Multiple Regression Model


# Ridge Regression
alpha = 1
n, m = x.shape
rr = Ridge(alpha=1)
rr.fit(x, y)
w = rr.coef_
plt.scatter(x, y)
plt.plot(x, w*x, c='red')

# Lasso Regression
reg = Lasso(alpha)
reg.fit(x_train, y_train)