import random
import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import model_selection
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm
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
# split the dataset
predictor_list = ['Gender', 'Married', 'Income', 'FirstPurchase', 'LoyaltyCard', 'WalletShare', 'TotTransactions',
                  'LastTransaction']
x = valuedf.loc[:, valuedf.columns!='CustomerLV']
y = valuedf.loc[:,"CustomerLV"]

# Normalizing predictor data to avoid bias from scales/ranges in predictors
scaler = StandardScaler()
x_max_scaled = x.copy()
for column in x_max_scaled.columns:
    x_max_scaled[column] = x_max_scaled[column] / x_max_scaled[column].abs().max()
x = x_max_scaled

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=577)
'''

# Multiple Regression Model
print("Multiple Linear Regression")
model = LinearRegression()
scores = []
model.fit(x_train, y_train)
intercept = model.intercept_
coefficients = model.coef_
print("intercept: " + str(intercept))
print(list(zip(model.coef_, predictor_list)))
y_pred= model.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100);
plt.show(block=True)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# summarize feature importance
for i,v in enumerate(coefficients):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(coefficients))], coefficients)
plt.show()

# RMSE for this model was 163. 


# Ridge Regression
alpha = 1
n, m = x.shape
rr = Ridge(alpha=1)
rr.fit(x_train, y_train)
rr_coef = rr.coef_
rr_pred = rr.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': rr_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100);
plt.show(block=True)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rr_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rr_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rr_pred)))
# summarize feature importance
for i,v in enumerate(rr_coef):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(rr_coef))], rr_coef)
plt.show()

# Using an alpha = 1, RMSE was 164. Gender and FirstPurchase predictors are of very low importance.

# Lasso Regression
alpha = 1
lass = Lasso(alpha)
lass.fit(x_train, y_train)
lass_coef = lass.coef_
lass_pred = lass.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': lass_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100)
plt.show(block=True)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lass_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, lass_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lass_pred)))
# summarize feature importance
for i,v in enumerate(lass_coef):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(lass_coef))], lass_coef)
plt.show()

# Using an alpha = 1, RMSE was 164. Gender and FirstPurchase predictors are of very low importance.

# Elastic Net Regression
alpha = 0.1
ee_rmse = []
while alpha < 1:
    e_net = ElasticNet(alpha=alpha)
    e_net.fit(x_train, y_train)
    e_net_coeff = e_net.coef_
    e_net_pred = e_net.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, e_net_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, e_net_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, e_net_pred)))
    ee_rmse.append(np.sqrt(metrics.mean_squared_error(y_test, e_net_pred)))
    alpha += 0.1
print(ee_rmse)

e_net = ElasticNet(alpha=0.1)
e_net.fit(x_train, y_train)
e_net_coeff = e_net.coef_
e_net_pred = e_net.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': e_net_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100)
plt.show(block=True)
# summarize feature importance
for i,v in enumerate(e_net_coeff):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(e_net_coeff))], e_net_coeff)
plt.show()
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, e_net_pred)))

# The minimum rmse was found at alpha = 0.1. At this alpha, rmse was 242. ender and FirstPurchase predictors are
# of very low importance.


# Principle Components Regression
pca = PCA()
pcr = pca.fit(x_train, y_train)
# generate the Scree plot for PCs
fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(len(predictor_list))
plt.plot(sing_vals, np.cumsum(pca.explained_variance_ratio_), linewidth=2)
plt.title(' cumulative variation  Plot')
plt.xlabel('Principal Component')
plt.ylabel('cumulative variation ')
plt.show()

#scale the training and testing data
X_reduced_train = pca.fit_transform(scale(x_train))
X_reduced_test = pca.transform(scale(x_test))[:,:5]

#train PCR model on training data
model = LinearRegression()
model.fit(X_reduced_train[:,:5], y_train)
pcr_pred = model.predict(X_reduced_test)
coefficients = model.coef_

df = pd.DataFrame({'Actual': y_test, 'Predicted': pcr_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100);
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pcr_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pcr_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pcr_pred)))
print(abs( pca.components_ ))
feature_imp = []
for i in range(1,8):
    feature_list = []
    for comp in pca.components_:
        feature_list.append(comp[i])
    feature_imp.append(statistics.mean(feature_list))
print(feature_imp)
# plot feature importance
plt.bar([x for x in range(len(feature_imp))], feature_imp)
plt.show()

# There was negligible improvement in explained variance with 7 PCs over 6 PCs, while substantial improvement was
# seen with additional PCs up to 6. Therefore, 6 PCs were used in the model. In the final PCR model, the rmse was 285.
# It is difficult to assess predictor importance in PCR regressions. However, I have calculated the mean importances
# of each feature across all components and output a bar graph. By this method, the most important features
# are Gender, FirstPurchase, and WalletShare


'''
# Partial Least Squares
plsr = PLSRegression(n_components=len(predictor_list), scale=True)
plsr.fit(x_train, y_train)
rmse_plot = []
for n_comp in range(1, len(predictor_list)):
    plsr = PLSRegression(n_components=n_comp, scale=True)
    plsr.fit(x_train, y_train)
    pls_pred = plsr.predict(x_test)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pls_pred)))
    rmse_plot.append(np.sqrt(metrics.mean_squared_error(y_test, pls_pred)))
plt.plot(range(1, len(predictor_list)), rmse_plot)
plt.show()

plsr = PLSRegression(n_components=6, scale=False)
y_cv = cross_val_predict(plsr, x_test, y_test, cv=10)
# Calculate scores
r2 = metrics.r2_score(y_test, y_cv)
mse = metrics.mean_squared_error(y_test, y_cv)
rpd = y.std()/np.sqrt(mse)
print('R2: %0.4f, MSE: %0.4f, RPD: %0.4f' %(r2, mse, rpd))

plt.figure(figsize=(6, 6))
with plt.style.context('ggplot'):
    plt.scatter(y_test, y_cv, color='red')
    plt.plot(y_test, y_test, '-g', label='Expected regression line')
    z = np.polyfit(y_test, y_cv, 1)
    plt.plot(np.polyval(z, y_test), y_test, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()
    plt.plot()
    plt.show()

# There was negligible improvement in RMSE when increasing the number of components from 6 to 7, therefore 6 
# components were used. In this model, the RMSE was 164. 

'''
# K Nearest Neighbors
neighbors = np.arange(1, 25)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train, y_train)
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(x_train, y_train)

plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, knn_pred)))
df = pd.DataFrame({'Actual': y_test, 'Predicted': knn_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100);
plt.show()


# The highest model accuracy score on the training data was found when k = 1. Therefore, k = 1 was used in the final
# model. In this model, rmse was 267.
'''

######################
# Question 2         #
######################

