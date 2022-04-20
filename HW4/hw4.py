import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import statistics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import tree # for decision tree models
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
from heapq import nsmallest
import sklearn.neural_network
from keras.models import Sequential
from keras.layers import Dense

######################################################################################################################
# Preliminary Analysis        #
###############################
nbadf = pd.read_csv("nba_merged_subset.csv")
#print(nbadf.isnull().any()) # Missing values found in the ThreeP column
threep = list(nbadf.loc[:,"ThreeP."])
threep_median = statistics.median(threep)
nbadf.fillna(threep_median, inplace=True)
#print(nbadf.isnull().any()) # Missing values found in the ThreeP column
nbadf.drop_duplicates(inplace=True, keep='first') # remove duplicate rows

X = nbadf.drop(['y', 'Name'], axis = 'columns')
y = nbadf.y
X_features = X.columns.values.tolist()

#define cross-validation method to use
cv = KFold(n_splits=10, random_state=577, shuffle=True)

# Normalizing predictor data to avoid bias from scales/ranges in predictors
#scaler = StandardScaler()
#x_max_scaled = X.copy()
#for column in x_max_scaled.columns:
#    x_max_scaled[column] = x_max_scaled[column] / x_max_scaled[column].abs().max()
#X = x_max_scaled

# Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 577)

'''
######################################################################################################################
# Question 1a         #
#######################
# List of values to try for max_depth:
max_depth_range = list(range(1, 10))  # List to store the accuracy for each value of max_depth:
accuracy = []
cv_score = []
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth=depth,
                                 random_state=0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accuracy.append(score)
    cv_score.append(statistics.mean(cross_val_score(clf, X_train, y_train, cv=cv)))
plt.plot(max_depth_range,cv_score)
plt.show()
plt.clf()
print("Estimated Accuracy:",max(cv_score))

# Create Optimal decision tree based on highest accuracy at max_depth = 2
# Estimated generalization accuracy of this model was 71%
clf = DecisionTreeClassifier(criterion='gini', max_depth = 2, random_state = 0)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Final Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Plot the decision tree model
plt.figure(figsize = (20,16))
tree.plot_tree(clf, fontsize = 16,rounded = True , filled = True, feature_names=X_features)
plt.show()

# Plot variable importance
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
plt.bar(importances.feature, importances.importance)
plt.show()

# Final model accuracy: 69.23%


######################################################################################################################
# Question 1b         #
#######################
# Train bagging ensemble on iterations of n_estimators=i
# and iterations of stump max_leaf_nodes=j
max_n_ests = 40
max_leaves = [10,100,1000,99999]
accuracy_matrix = []
for j in [0,1,2,3]:
    clf_stump=DecisionTreeClassifier(max_features=None,max_leaf_nodes=max_leaves[j])
    accuracy_list = []
    for i in np.arange(1,max_n_ests):
        baglfy=BaggingClassifier(base_estimator=clf_stump,n_estimators= i, max_samples=1.0)
        baglfy=baglfy.fit(X_train,y_train)
        bag_tr_err=y==baglfy.predict(X)
        accuracy_list.append(statistics.mean(cross_val_score(baglfy, X_train, y_train, cv=cv)))
    accuracy_matrix.append(accuracy_list)

plt.plot(np.arange(1,max_n_ests), accuracy_matrix[0], label='10 Leaves')
plt.plot(np.arange(1,max_n_ests), accuracy_matrix[1], label='100 Leaves')
plt.plot(np.arange(1,max_n_ests), accuracy_matrix[2], label='1000 Leaves')
plt.plot(np.arange(1,max_n_ests), accuracy_matrix[3], label='99999 Leaves')
plt.xlabel('Number of Estimators', fontsize=15)
plt.ylabel('Accuracy',  color='blue', fontsize=15)
plt.legend()
plt.show()

# The highest accuracy model after k-fold validation was the model with a maximum of 10 leaves and using 6 estimators,
# an estimated generalized accuracy of 73.9%
clf_stump=DecisionTreeClassifier(max_features=None,max_leaf_nodes=10)
baglfy=BaggingClassifier(base_estimator=clf_stump,n_estimators= 6, max_samples=1.0)
baglfy=baglfy.fit(X_train,y_train)
y_pred = baglfy.predict(X_test)
print("Final Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Plot variable importance
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.mean([
    tree.feature_importances_ for tree in baglfy.estimators_], axis=0)})
print(np.mean([tree.feature_importances_ for tree in baglfy.estimators_], axis=0))
importances = importances.sort_values('importance',ascending=False)
plt.bar(importances.feature, importances.importance)
plt.show()

# Final model accuracy: 68.4%

######################################################################################################################
# Question 1c         #
#######################
max_n_ests = 50
accuracy_matrix = []
accuracy_list = []
for i in [10,50,100,1000]:
    accuracy_list = []
    for j in np.arange(1,max_n_ests):
        rf_classifier = RandomForestClassifier(
                              max_leaf_nodes=i,
                              n_estimators=j,
                              bootstrap=True,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=577,
                              max_features='auto')
        rf_classifier = rf_classifier.fit(X_train, y_train)
        accuracy_list.append(statistics.mean(cross_val_score(rf_classifier, X_train, y_train, cv=cv)))
    print(accuracy_list)
    accuracy_matrix.append(accuracy_list)
print(accuracy_matrix)
plt.plot(np.arange(1,max_n_ests), accuracy_matrix[0], label='10 Leaves')
plt.plot(np.arange(1,max_n_ests), accuracy_matrix[1], label='50 Leaves')
plt.plot(np.arange(1,max_n_ests), accuracy_matrix[2], label='100 Leaves')
plt.plot(np.arange(1,max_n_ests), accuracy_matrix[3], label='1000 Leaves')
plt.xlabel('Number of Estimators', fontsize=15)
plt.ylabel('Accuracy',  color='blue', fontsize=15)
plt.legend()
plt.show()
'''
# The highest accuracy was achieved with 10 leaves and 5 estimators

rf_classifier = RandomForestClassifier(
                              max_leaf_nodes=10,
                              n_estimators=5,
                              bootstrap=True,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=577,
                              max_features='auto')
rf_classifier = rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
print("Final Accuracy:",metrics.accuracy_score(y_test, rf_pred))

# Plot variable importance
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf_classifier.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
plt.bar(importances.feature, importances.importance)
plt.show()

# Final model accuracy: 70.6%

'''
######################################################################################################################
# Question 1d         #  Gradient Boosted Tree
#######################
gradient_booster = GradientBoostingClassifier()
parameters = {
    "n_estimators":[5,50,250,500],
    "max_depth":[1,3,5,7,9],
    "learning_rate":[0.01,0.1,1,10,100]
}
#cv = GridSearchCV(gradient_booster,parameters,cv=5)
#cv.fit(X_train,y_train)
#print(f'Best parameters are: {cv.best_params_}')
# Best parameters are: {'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 500}
gradient_booster = GradientBoostingClassifier(learning_rate=0.1, max_depth=1, n_estimators=500)
gradient_booster.fit(X_train,y_train)
y_pred = gradient_booster.predict(X_test)
print("Final Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Plot variable importance
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(gradient_booster.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
plt.bar(importances.feature, importances.importance)
plt.show()

# Final model accuracy: 70.33%

######################################################################################################################
# Question 1e         #  Neural Net
#######################
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# Final model accuracy: 70.63%
'''
######################################################################################################################
# Question 1f         #  Summary
#######################
# Most models performed very similarly, all around 70% accuracy after tuning. As evidenced by the tuning of models that
# performed predictor selection (like the tree depth parameters of the tree models), many predictors were not
# significant. For example, in the bagging tree,PTS and GamesPlayed were much more significant that the other predictors
# All predictors having to do with 3-pointers, as well as FGA, BLK, and STL were marginal or insignificant.
# Similarly, in the simple tree model, the most significant predictors were FGM, GamesPlayed, and DREB. Other models
# were insignificant. The low significance of so many of the predictors likely contributed to the fact that
# very simple models performed about as well as very advanced models, like the neural net. Without useful predictors,
# even the most sophisticated models will not be able to perform well.