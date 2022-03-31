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
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import statistics
import numpy as np
import matplotlib.pyplot as plt
from heapq import nsmallest

######################################################################################################################
# Question 1         #
######################

energydf = pd.read_csv("energy_efficiency_homework3.csv")
energydf.rename(columns={'X1': 'RelativeCompactness', 'X2': 'SurfaceArea', 'X3': 'WallArea',
                         'X4': 'OverallHeight', 'X5': 'Orientation', 'X6': 'GlazingArea',
                         'X7': 'GlazingAreaDistribution', 'Y': 'HeatingLoad'}, inplace=True)
print(energydf.head())
X = energydf.drop(['HeatingLoad'], axis = 'columns')
y = energydf.HeatingLoad

# Normalizing predictor data to avoid bias from scales/ranges in predictors
scaler = StandardScaler()
x_max_scaled = X.copy()
for column in x_max_scaled.columns:
    x_max_scaled[column] = x_max_scaled[column] / x_max_scaled[column].abs().max()
X = x_max_scaled

# Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 577)

####################################
# Multiple Linear Regression Model #
####################################
model = LinearRegression()
model = model.fit(X_train,y_train)
y_pred= model.predict(X_test)
# Assess model accuracy
print('R Squared:', model.score(X_test,y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

##########################
# Ridge Regression Model #
##########################
'''
alpha = 0.1
rr_rmse = {}
n, m = X.shape

# Loop through potential alpha values to find the lowest RMSE model
while alpha < 5:
    rr = Ridge(alpha=alpha)
    rr.fit(X_train, y_train)
    rr_pred = rr.predict(X_test)
    rr_rmse[alpha] = (np.sqrt(metrics.mean_squared_error(y_test, rr_pred)))
    alpha += 0.1
print(rr_rmse)
best_alpha = nsmallest(1, rr_rmse, key = rr_rmse.get)
print(best_alpha)

# Use best tuning parameter to fit the RR model
rr = Ridge(alpha=alpha)
rr.fit(X_train, y_train)
rr_pred = rr.predict(X_test)

# Determine model perfomance metrics
df = pd.DataFrame({'Actual': y_test, 'Predicted': rr_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100);
plt.show(block=True)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rr_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rr_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rr_pred)))

# summarize feature importance
rr_coef = rr.coef_
for i,v in enumerate(rr_coef):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(rr_coef))], rr_coef)
plt.show()

##########################
# Lasso Regression Model #
##########################
alpha = 0.1
lass_rmse = {}
n, m = X.shape

# Loop through potential alpha values to find the lowest RMSE model
while alpha < 5:
    lass = Lasso(alpha=alpha)
    lass.fit(X_train, y_train)
    lass_pred = lass.predict(X_train)
    lass_rmse[alpha] = (np.sqrt(metrics.mean_squared_error(y_train, lass_pred)))
    alpha += 0.1
print(lass_rmse)
best_alpha = nsmallest(1, lass_rmse, key = lass_rmse.get)
print(best_alpha)

# Use best tuning parameter to fit the lass model
lass = Ridge(alpha=best_alpha)
lass.fit(X_train, y_train)
lass_pred = lass.predict(X_test)

# Determine model perfomance metrics
df = pd.DataFrame({'Actual': y_test, 'Predicted': lass_pred})
df.plot.scatter(x = 'Actual', y = 'Predicted', s = 100);
plt.show(block=True)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lass_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, lass_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lass_pred)))

# summarize feature importance
lass_coef = lass.coef_
for i,v in enumerate(lass_coef):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(lass_coef))], lass_coef)
plt.show()

#####################
# Elastic Net Model #
#####################
alpha = 0.1
e_net_rmse = {}

# Loop through potential alpha values to find the lowest RMSE model
while alpha < 1:
    e_net = ElasticNet(alpha=alpha)
    e_net.fit(X_train, y_train)
    e_net_pred = e_net.predict(X_train)
    e_net_rmse[alpha] = (np.sqrt(metrics.mean_squared_error(y_train, e_net_pred)))
    alpha += 0.1
print(e_net_rmse)
best_alpha = nsmallest(1, e_net_rmse, key = e_net_rmse.get)
print(best_alpha)

# Use best tuning parameter to fit the elastic net model
e_net = ElasticNet(alpha=best_alpha)
e_net.fit(X_train, y_train)
e_net_pred = e_net.predict(X_test)

e_net_coeff = e_net.coef_
e_net_pred = e_net.predict(X_test)
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
'''
###################################
# Principle Components Regression #
###################################
regr = LinearRegression()
mse = []
ncomp = list(range(1, len(X.columns) -1))
print(ncomp)
print(ncomp[-1])
i = 1

while i <= ncomp[-1]:
    pca = PCA(n_components=i)
    X_reduced_train = pca.fit_transform(X_train)
    regr.fit(X_reduced_train, y_train)
    y_c = regr.predict(X_reduced_train)
    mse_c = metrics.mean_squared_error(y_train, y_c)
    mse.append(mse_c)
    i += 1
plt.plot(ncomp,mse)
plt.show()

ncomp = mse.index(min(mse))
pca = PCA(n_components=ncomp)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_test = pca.fit_transform(X_test)
regr.fit(X_reduced_train, y_train)
pred = regr.predict(X_reduced_test)
mse = metrics.mean_squared_error(y_test, pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

##################
# PLS Regression #
##################