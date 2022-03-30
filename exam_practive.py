import random
import numpy
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt

# Import CSV file as dataframe
valuedf = pd.read_csv("VALUE.csv")

# Scatterplot of predictor vs response
valuedf.plot.scatter(x = 'Gender', y = 'CustomerLV', s = 100)
plt.show(block=True)

# Histogram of predictor data
plt.hist(valuedf.loc[:,""], bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

# Box plot
boxplot = valuedf.boxplot(column=['C1', 'C2', 'C3'])
plt.show()

# Get quantile from each column
print(valuedf[''].quantile([.2,.4,.6,.8]))

# Perform stats on predictor
box = list(valuedf.loc[:,"BoxOffice"])
print(valuedf['BoxOffice'].unique)
print(valuedf['BoxOffice'].value_counts())
print(statistics.median([x for x in box if x < 177 and x > 128]))
print(max(box[:3049]))
comedy = sum(1 for x in box if x == "Comedy")

# Query data from df
movie2 = moviedf.query('Budget >= 10 and Budget <= 20', inplace = False)

# Remove bad data from predictor list
myList = list(filter(("NA").__ne__, box))

# Test train split
X = valuedf.drop(['Income'], axis = 'columns', inplace = True)
y = valuedf.Income
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 577)

#One Hot Encoder for Nominal Variables
ohe = OneHotEncoder(sparse = False)
add_columns = pd.get_dummies(valuedf['Gender'])
valuedf.drop(['',''], axis = 'columns', inplace = True) # drop one dummy variables and original column
valuedf.join(add_columns)
#merged = pd.concat([df1, df2], axis='columns')

# LabelEncoder
#df=df.sort_values(by=[''])
#df['encoded'] = pd.factorize(df['position'])[0]

# Scale the continuous variables. Standard scaler assumes normally distributed data
# Encoded predictors should be dropped before scaling
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LinearRegression()
model = model.fit(X,y)

# Assess model accuracy
score=model.score(X_test,y_test)

# Assess coefficients
intercept = model.intercept_
coefficients = model.coef_
print("intercept: " + str(intercept))
print(list(zip(model.coef_, predictor_list)))

# Make predictions based on model
predict = model.predict([[288,0,76]])

# Actual vs Predicted plot
y_pred= model.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100)
plt.show(block=True)

# kfold cross-validation- RMSE evaluation
k = 5
cv = KFold(n_splits=k, random_state=None, shuffle=True)
model = LinearRegression()
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)
sqrt(mean(absolute(scores))) #view RMSE



