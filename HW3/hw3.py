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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
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

# summarize feature importance
model_coef = model.coef_
for i,v in enumerate(model_coef):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(model_coef))], model_coef)
plt.show()

# MSE = 8.8, RMSE = 2.97.
# In this linear model, the most important predictor was SurfaceArea followed by RelativeCompactness.
# Orientation and GlazingAreaDistribution were negligible.

''''''
##########################
# Ridge Regression Model #
##########################

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

# MSE = 12.01, RMSE = 3.48
# The most significant predictor in this model was OverallHeight, followed by WallArea. Orientation and
# GlazingAreaDIstribution were negligible in the model


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

# MSE = 9.24, RMSE = 3.04
# In the lasso model, the most important predictors were OverallHeight followed by WallArea.
# Orientation was negligible and GlazingAreaDistribution very minor 

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
print(' Mean Squared Error:', (metrics.mean_squared_error(y_test, e_net_pred)))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, e_net_pred)))

# MSE = 27.48, RMSE = 5.24
# In the elactic net model, OverallHeight was by far the most significant predictor.
# Orientation was removed as a predictor altogether, and GlazingAreaDistribution was minor.

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
pca = PCA(n_components=5)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_test = pca.fit_transform(X_test)
regr.fit(X_reduced_train, y_train)
pred = regr.predict(X_reduced_test)
mse = metrics.mean_squared_error(y_test, pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

# MSE = 303.1, RMSE = 17.4


##################
# PLS Regression #
##################
plsr = PLSRegression(n_components=len(X.columns), scale=True)
plsr.fit(X_train, y_train)
rmse_plot = []
for n_comp in range(1, len(X.columns)):
    plsr = PLSRegression(n_components=n_comp, scale=True)
    plsr.fit(X_train, y_train)
    pls_pred = plsr.predict(X_test)
    rmse_plot.append(np.sqrt(metrics.mean_squared_error(y_test, pls_pred)))
plt.plot(range(1, len(X.columns)), rmse_plot)
plt.show()

plsr = PLSRegression(n_components=5, scale=False)
plsr.fit(X_train, y_train)
pred = plsr.predict(X_test)
mse = metrics.mean_squared_error(y_test, pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

# MSE = 9.39, RMSE = 3.06


#############################
# K Nearest Neighbors       #
#############################
neighbors = np.arange(1, 26)
print(neighbors)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
rmse_plot = []
for i in neighbors:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    rmse_plot.append(np.sqrt(metrics.mean_squared_error(y_test, knn_pred)))
plt.plot(neighbors, rmse_plot)
plt.show()

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
mse = metrics.mean_squared_error(y_test, pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

# MSE = 6.58, RMSE = 2.57

# The best performing models were the KNN model, followed by the multiple linear regression model. Most other models 
# performed similarly, except the elastic net model was slightly worse, and the PCR model was very poor. The PCR
# model performed substantially worse than other models, possibly because PCR does not train itself on the response
# variable, but only on variance of predictors. In this case, it resulted in a poor model, possibly because there was
# large variance in predictors that were not truly important. 


######################################################################################################################
# Question 2         #
######################
predictors = np.arange(0, 8)
leukdf = pd.read_csv("leukem_std.csv", usecols=predictors, header=None)
X = leukdf
y = pd.read_csv("leukem_std.csv", usecols=[7129], header=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 577)

#############################
# KNN Classifier Model      #
#############################
neighbors = np.arange(1, 26)
score_plot = []
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    score_plot.append(metrics.accuracy_score(y_test, knn_pred))
plt.plot(neighbors, score_plot)
plt.show()

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
score = metrics.accuracy_score(y_test, knn_pred)
print("Accuracy score: ", score)

# Accuracy: 66.7%

#####################################
# Linear Discriminant Analysis      #
#####################################
lda = LinearDiscriminantAnalysis()
print(y_train.head())
lda.fit(X_train, y_train)
lda_pred = lda.predict(X_test)
score = metrics.accuracy_score(y_test, lda_pred)
print("Accuracy score: ", score)

# Accuracy: 73.3%

########################################
# Quadratic Discriminant Analysis      #
########################################
qda = QuadraticDiscriminantAnalysis()
print(y_train.head())
qda.fit(X_train, y_train)
qda_pred = qda.predict(X_test)
score = metrics.accuracy_score(y_test, qda_pred)
print("Accuracy score: ", score)

# Accuracy: 73.3%

###########################
# Logistic Regression     #
###########################
logreg = LogisticRegression(random_state = 0, penalty='none')
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
score = metrics.accuracy_score(y_test, logreg_pred)
print("Accuracy score: ", score)

# summarize feature importance
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': logreg.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()

# Accuracy: 73.3%

#######################################
# Regularized Logistic Regression     #
#######################################
weights, params = [], []
accuracy_scores = []
for c in np.arange(-5, 5, dtype=float):
   lr = LogisticRegression(C=10**c, random_state=0)
   lr.fit(X_train, y_train)
   lr_pred = lr.predict(X_test)
   score = metrics.accuracy_score(y_test, lr_pred)
   print("Accuracy score: ", score)
   print("c = ", c)
   accuracy_scores.append(score)
   params.append(10 ** c)
weights = np.array(weights)
print(params)

# Decision region drawing
plt.plot(params, accuracy_scores, color='blue', marker='x', label='Accuracy')
plt.ylabel('accuracy score')
plt.xlabel('C')
plt.legend(loc='right')
plt.xscale('log')
plt.show()

lr = LogisticRegression(C=10**(-1), random_state=0)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
score = metrics.accuracy_score(y_test, lr_pred)
print("Accuracy score: ", score)

# summarize feature importance
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': lr.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()

# Using a hyperparameter of 0.1, the accuracy was 80%. Important features were genes 2, 3, and 4. 5 is negligible

#####################################
# Linear Support Vector Machine    #
#####################################
weights, params = [], []
accuracy_scores = []
for c in np.arange(-5, 5, dtype=float):
   lsvc = SVC(C=10**c, kernel='linear')
   lsvc.fit(X_train, y_train)
   lr_pred = lsvc.predict(X_test)
   score = metrics.accuracy_score(y_test, lr_pred)
   print("Accuracy score: ", score)
   print("c = ", c)
   accuracy_scores.append(score)
   params.append(10 ** c)
weights = np.array(weights)
print(params)

# Decision region drawing
plt.plot(params, accuracy_scores, color='blue', marker='x', label='Accuracy')
plt.ylabel('accuracy score')
plt.xlabel('C')
plt.legend(loc='right')
plt.xscale('log')
plt.show()

lsvc = SVC(C=0.1, kernel='linear')
lsvc.fit(X_train, y_train)
lsvc_pred = lsvc.predict(X_test)
score = metrics.accuracy_score(y_test, lsvc_pred)
print("Accuracy score: ", score)

# summarize feature importance
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': lsvc.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()

# Using a tuning hyperparameter C of 0.1, Accuracy: 80%

####################################
# Radial Support Vector Machine    #
####################################
weights, params = [], []
accuracy_scores = []
for c in np.arange(-5, 5, dtype=float):
   lsvc = SVC(C=10**c, kernel='rbf', gamma='auto')
   lsvc.fit(X_train, y_train)
   lr_pred = lsvc.predict(X_test)
   score = metrics.accuracy_score(y_test, lr_pred)
   print("Accuracy score: ", score)
   print("c = ", c)
   accuracy_scores.append(score)
   params.append(10 ** c)
weights = np.array(weights)
print(params)

# Decision region drawing
plt.plot(params, accuracy_scores, color='blue', marker='x', label='Accuracy')
plt.ylabel('accuracy score')
plt.xlabel('C')
plt.legend(loc='right')
plt.xscale('log')
plt.show()

clf = SVC(kernel='rbf', C=10, gamma='auto')
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, clf_pred)
print("Accuracy score: ", score)

# Accuracy: 73.3%


# The best performing model as assessed by classification accuracy were the linear SVC and the regularized LDA models,
# both of which tied with 80% accuracy. Important features in the Reguylarized LDA model were genes 2, 3, and 4.
# Gene 5 is negligible. The linear SVC model was very different. In this model, genes 4, 8, and 5 were most important.
#

######################################################################################################################
# Question 3         #
######################
winedf = pd.read_csv("WINE.csv")
le = preprocessing.LabelEncoder()
winedf['Quality'] = le.fit_transform(winedf['Quality'])
X = winedf.drop(['Quality'], axis = 'columns')
y = winedf.Quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 577)

#############################
# KNN Classifier Model      #
#############################
neighbors = np.arange(1, 26)
score_plot = []
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    score_plot.append(metrics.accuracy_score(y_test, knn_pred))
plt.plot(neighbors, score_plot)
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
score = metrics.accuracy_score(y_test, knn_pred)
print("Accuracy score: ", score)

# roc curve for models
knn_prob = knn.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, knn_prob[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
auc_score = roc_auc_score(y_test, knn_prob[:,1])
print(auc_score)

# plot roc curves
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

# Accuracy: 75.5%, AUC = 0.80

#####################################
# Linear Discriminant Analysis      #
#####################################
lda = LinearDiscriminantAnalysis()
print(y_train.head())
lda.fit(X_train, y_train)
lda_pred = lda.predict(X_test)
score = metrics.accuracy_score(y_test, lda_pred)
print("Accuracy score: ", score)

# roc curve for models
roc_prob = lda.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, roc_prob[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
auc_score = roc_auc_score(y_test, roc_prob[:,1])
print(auc_score)

# plot roc curves
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='LDA')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

# Accuracy: 83.7%, AOC Score: 0.898

########################################
# Quadratic Discriminant Analysis      #
########################################
qda = QuadraticDiscriminantAnalysis()
print(y_train.head())
qda.fit(X_train, y_train)
qda_pred = qda.predict(X_test)
score = metrics.accuracy_score(y_test, qda_pred)
print("Accuracy score: ", score)

# roc curve for models
roc_prob = qda.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, roc_prob[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
auc_score = roc_auc_score(y_test, roc_prob[:,1])
print(auc_score)

# plot roc curves
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='QDA')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

# Accuracy: 83.7%, AUC Score: 0.922

###########################
# Logistic Regression     #
###########################
logreg = LogisticRegression(random_state = 0, penalty='none')
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
score = metrics.accuracy_score(y_test, logreg_pred)
print("Accuracy score: ", score)

# summarize feature importance
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': logreg.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()

# roc curve for models
roc_prob = logreg.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, roc_prob[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
auc_score = roc_auc_score(y_test, roc_prob[:,1])
print(auc_score)

# plot roc curves
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Log Reg')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

# Accuracy: 79.4%, AUC Score: 0.882


#######################################
# Regularized Logistic Regression     #
#######################################
weights, params = [], []
accuracy_scores = []
for c in np.arange(-5, 5, dtype=float):
   lr = LogisticRegression(C=10**c, random_state=0)
   lr.fit(X_train, y_train)
   lr_pred = lr.predict(X_test)
   score = metrics.accuracy_score(y_test, lr_pred)
   print("Accuracy score: ", score)
   print("c = ", c)
   accuracy_scores.append(score)
   params.append(10 ** c)
weights = np.array(weights)
print(params)

# Decision region drawing
plt.plot(params, accuracy_scores, color='blue', marker='x', label='Accuracy')
plt.ylabel('accuracy score')
plt.xlabel('C')
plt.legend(loc='right')
plt.xscale('log')
plt.show()

lr = LogisticRegression(C=10**(-1), random_state=0)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
score = metrics.accuracy_score(y_test, lr_pred)
print("Accuracy score: ", score)

# summarize feature importance
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': lr.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()

# roc curve for models
roc_prob = lr.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, roc_prob[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
auc_score = roc_auc_score(y_test, roc_prob[:,1])
print(auc_score)

# plot roc curves
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Log Reg')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

# Using a hyperparameter of 0.1, the accuracy was 81.9%. AUC Score: 0.868

#####################################
# Linear Support Vector Machine    #
#####################################
weights, params = [], []
accuracy_scores = []
for c in np.arange(-5, 0, dtype=float):
   lsvc = SVC(C=10**c, kernel='linear', probability=True)
   lsvc.fit(X_train, y_train)
   lr_pred = lsvc.predict(X_test)
   score = metrics.accuracy_score(y_test, lr_pred)
   print("Accuracy score: ", score)
   print("c = ", c)
   accuracy_scores.append(score)
   params.append(10 ** c)
weights = np.array(weights)
print(params)

# Decision region drawing
plt.plot(params, accuracy_scores, color='blue', marker='x', label='Accuracy')
plt.ylabel('accuracy score')
plt.xlabel('C')
plt.legend(loc='right')
plt.xscale('log')
plt.show()

lsvc = SVC(C=0.01, kernel='linear', probability=True)
lsvc.fit(X_train, y_train)
lsvc_pred = lsvc.predict(X_test)
score = metrics.accuracy_score(y_test, lsvc_pred)
print("Accuracy score: ", score)

# roc curve for models
roc_prob = lsvc.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, roc_prob[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
auc_score = roc_auc_score(y_test, roc_prob[:,1])
print(auc_score)

# plot roc curves
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Linear SVC')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

# Using a tuning hyperparameter C of 1, Accuracy: 82.1%. AUC Score: 0.862

####################################
# Radial Support Vector Machine    #
####################################
weights, params = [], []
accuracy_scores = []
for c in np.arange(-5, 5, dtype=float):
   lsvc = SVC(C=10**c, kernel='rbf', gamma='auto')
   lsvc.fit(X_train, y_train)
   lr_pred = lsvc.predict(X_test)
   score = metrics.accuracy_score(y_test, lr_pred)
   print("Accuracy score: ", score)
   print("c = ", c)
   accuracy_scores.append(score)
   params.append(10 ** c)
weights = np.array(weights)
print(params)

# Decision region drawing
plt.plot(params, accuracy_scores, color='blue', marker='x', label='Accuracy')
plt.ylabel('accuracy score')
plt.xlabel('C')
plt.legend(loc='right')
plt.xscale('log')
plt.show()

clf = SVC(kernel='rbf', C=10, gamma='auto', probability=True)
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, clf_pred)
print("Accuracy score: ", score)

# roc curve for models
roc_prob = clf.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, roc_prob[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
auc_score = roc_auc_score(y_test, roc_prob[:,1])
print(auc_score)

# plot roc curves
plt.style.use('seaborn')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Linear SVC')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()

# Accuracy: 78.4%, AUC Score: 0.835

# The highest performing classification model as judged by the AUC metric was the Quadratic Discriminant Analysis
# with an AUC Score of 0.922 and 83.7% overall accuracy on the test dataset.