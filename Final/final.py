import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import statistics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import stats
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


class final():
    def main(self):
        #X_train, X_test, y_train, y_test, cv, X, y = final.prelim()
        #final.onea(X_train, X_test, y_train, y_test, cv, X, y)
        #final.oneb(X_train, X_test, y_train, y_test, cv, X, y)
        #final.onec(X_train, X_test, y_train, y_test, cv, X, y)
        #final.oned(X_train, X_test, y_train, y_test, cv, X, y)
        #final.onee(X_train, X_test, y_train, y_test, cv, X, y)
        #final.onef(X_train, X_test, y_train, y_test, cv, X, y)
        #final.oneg(X_train, X_test, y_train, y_test, cv, X, y)
        X_train, X_test, y_train, y_test, cv, X, y = final.prelim2()
        #final.twoa(X_train, X_test, y_train, y_test, cv, X, y)
        #final.twob(X_train, X_test, y_train, y_test, cv, X, y)
        #final.twoc(X_train, X_test, y_train, y_test, cv, X, y)
        final.twod(X_train, X_test, y_train, y_test, cv, X, y)
        #final.twoe(X_train, X_test, y_train, y_test, cv, X, y)
        #final.twof(X_train, X_test, y_train, y_test, cv, X, y)
        #final.twog(X_train, X_test, y_train, y_test, cv, X, y)
        #final.twoh(X_train, X_test, y_train, y_test, cv, X, y)
        #krogerdf, originaldf = final.prelim3()
        #final.threea(krogerdf, originaldf)
        #final.threeb(krogerdf, originaldf)

######################################################################################################################
# Preliminary Analysis        #
###############################
    def prelim(self):
        elecdf = pd.read_csv("Electricity.csv")
        elecdf.drop(['ID', 'day', 'YearMonth'], axis = 'columns', inplace=True)
        # Label Encoding for Ordinal Variables
        le = preprocessing.LabelEncoder()
        elecdf['PayPlan'] = le.fit_transform(elecdf['PayPlan'])
        plt.hist(elecdf.Usage)
        plt.clf()
        X = elecdf.drop(['Usage'], axis='columns')
        y = elecdf.Usage
        # Normalizing predictor data to avoid bias from scales/ranges in predictors
        scaler = StandardScaler()
        x_max_scaled = X.copy()
        for column in x_max_scaled.columns:
            x_max_scaled[column] = x_max_scaled[column] / x_max_scaled[column].abs().max()
        X = x_max_scaled
        # Test train split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5000, random_state=577)
        cv = KFold(10)
        return X_train, X_test, y_train, y_test, cv, X, y

######################################################################################################################
# Question 1a         #
#######################
    def onea(self, X_train, X_test, y_train, y_test, cv, X, y):
        model = LinearRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_train)
        accuracy = -1 * np.mean(cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error'))
        print("Generalized Accuracy score: ", accuracy)
        y_pred = model.predict(X_test)
        # Assess model accuracy
        print('R Squared:', model.score(X_test,y_test))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        # summarize feature importance
        predictor_list = X_train.columns
        predictor_importance = list(zip(model.coef_, predictor_list))
        model_coef = model.coef_

        # plot feature importance
        plt.bar([x[1] for x in predictor_importance], [x[0] for x in predictor_importance])
        plt.show()

        # Generalized RMSE: 26.73; Holdout RMSE: 26.17
        # By far the most significant predictors are mean (positive), heatinghouyrs (positive), and coolinghours (neg)

######################################################################################################################
# Question 1b         #
#######################
    def oneb(self, X_train, X_test, y_train, y_test, cv, X, y):
        alpha_list = np.arange(0,1.1,0.1)
        rmse_list = []
        for alpha in alpha_list:
            rr = Ridge(alpha=alpha)
            rr.fit(X_train, y_train)
            rr_coef = rr.coef_
            rr_pred = rr.predict(X_test)
            rmse_list.append(np.sqrt(metrics.mean_squared_error(y_test, rr_pred)))
        plt.plot(alpha_list, rmse_list)
        plt.show()
        plt.clf()
        # Optimal model found at alpha=0.0
        rr = Ridge(alpha=0.0)
        rr.fit(X_train, y_train)
        accuracy = -1 * np.mean(cross_val_score(rr, X, y, cv=cv, scoring='neg_root_mean_squared_error'))
        print("Generalized Accuracy score: ", accuracy)
        rr_coef = rr.coef_
        rr_pred = rr.predict(X_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rr_pred)))
        predictor_list = X_train.columns
        predictor_importance = list(zip(rr.coef_, predictor_list))
        # plot feature importance
        plt.bar([x[1] for x in predictor_importance], [x[0] for x in predictor_importance])
        plt.show()
        plt.clf()
        # Generalized RMSE: 26.76; Holdout RMSE: 26.17
        # By far the most significant predictors are mean (positive), heatinghours (positive), and coolinghours (neg)

######################################################################################################################
# Question 1c         #
#######################
    def onec(self, X_train, X_test, y_train, y_test, cv, X, y):
        neighbors_list = [1,15,40,80,120]
        knn_rmse = []
        for neighbors in neighbors_list:
            knn = KNeighborsRegressor(n_neighbors=neighbors)
            knn.fit(X_train, y_train)
            knn_pred = knn.predict(X_test)
            knn_rmse.append(np.sqrt(metrics.mean_squared_error(y_test, knn_pred)))
        plt.plot(neighbors_list, knn_rmse)
        plt.show()
        plt.clf()
        # Elbow in RMSE plot at n=40
        knn = KNeighborsRegressor(n_neighbors=40)
        knn.fit(X_train, y_train)
        accuracy = -1 * np.mean(cross_val_score(knn, X, y, cv=cv, scoring='neg_root_mean_squared_error'))
        print("Generalized Accuracy score: ", accuracy)
        knn_pred = knn.predict(X_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, knn_pred)))

        #Generalized RMSE: 27.00
        #Holdout RMSE: 26.32


    ######################################################################################################################
# Question 1d         #
#######################
    def oned(self, X_train, X_test, y_train, y_test, cv, X, y):
        tree = DecisionTreeRegressor()
        tree.fit(X_train, y_train)
        accuracy = -1 * np.mean(cross_val_score(tree, X, y, cv=cv, scoring='neg_root_mean_squared_error'))
        print("Generalized Accuracy score: ", accuracy)
        tree_pred = tree.predict(X_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, tree_pred)))
        predictor_importance = list(tree.feature_importances_)
        predictor_list = X_train.columns
        # plot feature importance
        plt.bar(predictor_list, predictor_importance)
        plt.show()
        plt.clf()
        #
        # By far the most significant predictor was mean, followed by heatinghours
        # Generalized RMSE:: 27.00757968963032
        # Holdout RMSE: 26.702


######################################################################################################################
# Question 1e         #
#######################
    def onee(self, X_train, X_test, y_train, y_test, cv, X, y):
        mtry_list = [2,3,8,12]
        tree_rmse = []
        for mtry in mtry_list:
            regressor = RandomForestRegressor(min_samples_split=mtry)
            regressor.fit(X_train, y_train)
            tree_pred = regressor.predict(X_test)
            tree_rmse.append(np.sqrt(metrics.mean_squared_error(y_test, tree_pred)))
        plt.plot(mtry_list, tree_rmse)
        plt.show()
        plt.clf()
        # Minimum RMSE found at mtry=12
        regressor = RandomForestRegressor(min_samples_split=12)
        regressor.fit(X_train, y_train)
        accuracy = -1 * np.mean(cross_val_score(regressor, X, y, cv=cv, scoring='neg_root_mean_squared_error'))
        print("Generalized Accuracy score: ", accuracy)
        tree_pred = regressor.predict(X_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, tree_pred)))
        predictor_importance = list(regressor.feature_importances_)
        predictor_list = X_train.columns
        # plot feature importance
        plt.bar(predictor_list, predictor_importance)
        plt.show()
        plt.clf()
        # The most important predictors were heatinghours and mean
        # Generalized RMSE: 26.93
        # Holdout RMSE: 26.59

######################################################################################################################
# Question 1f         #
#######################
    def onef(self, X_train, X_test, y_train, y_test, cv, X, y):
        gradient_booster = GradientBoostingRegressor()
        parameters = {
            "n_estimators": [5, 50, 100],
            "max_depth": [10, 50, 100],
            "learning_rate": [0.1, 1, 100]
        }
        #cv = GridSearchCV(gradient_booster,parameters,cv=5)
        #cv.fit(X_train,y_train)
        #print(f'Best parameters are: {cv.best_params_}')
        # Best parameters are: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 50}
        parameters = {
            "n_estimators": [50],
            "max_depth": [10],
            "learning_rate": [0.1]
        }
        gradient_booster.fit(X_train, y_train)
        y_pred = gradient_booster.predict(X_train)
        cv_rmse = cross_val_score(gradient_booster, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        rmse = np.mean(cv_rmse) * -1
        print('Generalized Root Mean Squared Error:', rmse)
        boost_pred = gradient_booster.predict(X_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, boost_pred)))
        # Generalised RMSE: 26.7; Holdout RMSE: 26.1

######################################################################################################################
# Question 1g         #
#######################
    def oneg(self, X_train, X_test, y_train, y_test, cv, X, y):
        nodes = np.arange(1,6)
        model_rmse = []
        for node in nodes:
            model = Sequential()
            model.add(Dense(units=node, input_dim=12, activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
            history = model.fit(X_train, y_train)
            print(history.history)
            model_rmse.append(np.sqrt(history.history['mse']))
        plt.plot(nodes, model_rmse)
        plt.show()
        plt.clf()
        # Lowest RMSE achieved with 5 nodes
        model = Sequential()
        model.add(Dense(units=5, input_dim=12, activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])
        cv_rmse = cross_val_score(model,X,y,cv=cv, scoring='neg_root_mean_squared_error')
        rmse = np.mean(cv_rmse)
        print('Generalized Root Mean Squared Error:', rmse)
        history = model.fit(X_test, y_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(history.history['mse']))
        # Generalised RMSE: 38.8; Holdout RMSE: 31.7

######################################################################################################################
# Question 1h         #
#######################
#
# Out of all of the tested model, the one with the lowest generalized RMSE was the decision tree regressor.
# However, the linear regression model performed nearly identically with a RMSE of 26.7. Therefore, due to the
# simplicity and interpretability of a linear regression model, I would select this model as the optimal one
#

######################################################################################################################
# Preliminary Analysis 2      #
###############################
    def prelim2(self):
        purchasedf = pd.read_csv("PURCHASE.csv")
        X = purchasedf.drop(['Purchase'], axis = 'columns')
        columns = X.columns
        y = purchasedf.Purchase
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        X = pd.DataFrame(scaled, columns=[columns])
        print(X.head())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=577)
        cv = StratifiedKFold(10)
        return X_train, X_test, y_train, y_test, cv, X, y

######################################################################################################################
# Question 2a         #
#######################
    def twoa(self, X_train, X_test, y_train, y_test, cv, X, y):
        y_pred = ["No"] * len(y)
        print('Generalized Naive Accuracy:', metrics.accuracy_score(y, y_pred))
        y_pred = ["No"] * len(y_test)
        print('Holdout Naive Accuracy:', metrics.accuracy_score(y_test, y_pred))
        # General Naive accuracy 74%
        # Holdout accuracy 74.79%

######################################################################################################################
# Question 2b         #
#######################
    def twob(self, X_train, X_test, y_train, y_test, cv, X, y):
        logreg = LogisticRegression(random_state=0, penalty='none')
        logreg.fit(X_train, y_train)
        logreg_pred = logreg.predict(X_test)
        accuracy = np.mean(cross_val_score(logreg,X,y,cv=cv, scoring='accuracy'))
        print("Generalized Accuracy score: ", accuracy)
        score = metrics.accuracy_score(y_test, logreg_pred)
        print("Holdout Accuracy score: ", score)
        print(X_train.head())
        print(list(X_train.columns.levels[0]))
        print(logreg.coef_[0])
        # summarize feature importance
        importances = pd.DataFrame(data={
            'Attribute': list(X_train.columns.levels[0]),
            'Importance': logreg.coef_[0]
        })
        importances = importances.sort_values(by='Importance', ascending=False)
        plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
        plt.title('Feature importances obtained from coefficients', size=20)
        plt.xticks(rotation='vertical')
        plt.show()

        # The most important predictors are CloseStore, followed by Spent
        # Generalized Accuracy: 74.8%
        # Holdout accuracy: 74.4%

######################################################################################################################
# Question 2c         #
#######################
    def twoc(self, X_train, X_test, y_train, y_test, cv, X, y):
        weights, params = [], []
        accuracy_scores = []
        for c in np.arange(-5, 5, dtype=float):
            lr = LogisticRegression(C=10 ** c, random_state=0)
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

        # Highest accuracy found at C=10^-5
        lr = LogisticRegression(C=10 ** (-5), random_state=0)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        accuracy = np.mean(cross_val_score(lr, X, y, cv=cv, scoring='accuracy'))
        print("Generalized Accuracy score: ", accuracy)
        score = metrics.accuracy_score(y_test, lr_pred)
        print("Holdout Accuracy score: ", score)

        # summarize feature importance
        importances = pd.DataFrame(data={
            'Attribute': X_train.columns.levels[0],
            'Importance': lr.coef_[0]
        })
        importances = importances.sort_values(by='Importance', ascending=False)
        plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
        plt.title('Feature importances obtained from coefficients', size=20)
        plt.xticks(rotation='vertical')
        plt.show()

        #Generalized Accuracy score: 0.755
        #Holdout Accuracy score: 0.748
        # Feature importance: CloseStores, Closest, Spent

######################################################################################################################
# Question 2d         #
#######################
    def twod(self, X_train, X_test, y_train, y_test, cv, X, y):
        dtree = DecisionTreeClassifier()
        parameters = {
            "criterion": ['gini', 'entropy'],
            "max_depth": [2, 5, 10, 50, 100, None],
            "min_samples_split" : [2, 6, 10, 18, 50, 100],
        }
        grid = GridSearchCV(dtree, parameters, cv=5)
        grid.fit(X_train,y_train)
        print(f'Best parameters are: {grid.best_params_}')
        # Best parameters are: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 2}
        dtree = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=2)
        accuracy = np.mean(cross_val_score(dtree, X, y, cv=cv, scoring='accuracy'))
        print("Generalized Accuracy score: ", accuracy)
        dtree = dtree.fit(X_train, y_train)
        y_pred = dtree.predict(X_test)
        print("Holdout Final Accuracy:", metrics.accuracy_score(y_test, y_pred))

        # summarize feature importance
        importances = pd.DataFrame(data={
            'Attribute': X_train.columns.levels[0],
            'Importance': dtree.feature_importances_
        })
        importances = importances.sort_values(by='Importance', ascending=False)
        plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
        plt.title('Feature importances obtained from coefficients', size=20)
        plt.xticks(rotation='vertical')
        plt.show()
        plt.clf()

        # Plot the decision tree model
        plt.figure(figsize=(20, 16))
        tree.plot_tree(dtree, fontsize=16, rounded=True, filled=True, feature_names=X_train.columns.levels[0])
        plt.show()
        plt.clf()
        # The only used predictors are CloseStores and Closest
        #Generalized Accuracy score: 0.755
        #Holdout Accuracy score: 0.748

######################################################################################################################
# Question 2e         #
#######################
    def twoe(self, X_train, X_test, y_train, y_test, cv, X, y):
        mtry_list = [1.0, 3, 5]
        accuracy_list = []
        #for mtry in mtry_list:
        #    rf_classifier = RandomForestClassifier(min_samples_split=mtry)
        #    rf_classifier.fit(X_train, y_train)
        #    tree_pred = rf_classifier.predict(X_test)
        #    accuracy_list.append(statistics.mean(cross_val_score(rf_classifier, X_train, y_train, cv=cv)))
        #plt.plot(mtry_list, accuracy_list)
        #plt.show()
        #plt.clf()
        # Highest accuracy when mtry=2
        rf_classifier = RandomForestClassifier(min_samples_split=2)
        rf_classifier.fit(X_train, y_train)
        tree_pred = rf_classifier.predict(X_test)
        print("Generalized Accuracy score: ", statistics.mean(cross_val_score(rf_classifier, X_train, y_train, cv=cv)))
        print("Holdout Final Accuracy:", metrics.accuracy_score(y_test, tree_pred))
        # Plot variable importance
        importances = pd.DataFrame(
            {'feature': X_train.columns.levels[0],
             'importance': rf_classifier.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        plt.bar(importances.feature, importances.importance)
        plt.show()
        print("AUC: ", roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1]))
        #Generalized Accuracy score:  0.7234683325166382
        #Holdout Final Accuracy: 0.713172702351753
        # AUC: 0.58
        # Important features are PercentClose, followed by Spent, and closest

######################################################################################################################
# Question 2f         #
#######################
    def twof(self, X_train, X_test, y_train, y_test, cv, X, y):
        gradient_booster = GradientBoostingClassifier()
        parameters = {
            "n_estimators": [5, 50, 250, 500],
            "max_depth": [1, 10, 50, 100],
            "learning_rate": [0.1, 1, 10, 100]
        }
        #gcv = GridSearchCV(gradient_booster,parameters,cv=5)
        #gcv.fit(X_train,y_train)
        #print(f'Best parameters are: {gcv.best_params_}')
        # Best parameters are: {'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 5}

        gradient_booster = GradientBoostingClassifier(learning_rate=0.1, max_depth=1, n_estimators=5)
        gradient_booster.fit(X_train,y_train)
        y_pred = gradient_booster.predict(X_test)
        print("Final Accuracy:", metrics.accuracy_score(y_test, y_pred))
        # Plot variable importance
        importances = pd.DataFrame(
            {'feature': X_train.columns.levels[0],
             'importance': gradient_booster.feature_importances_
             })
        importances = importances.sort_values('importance', ascending=False)
        plt.bar(importances.feature, importances.importance)
        plt.show()

        print("Generalized Accuracy: ", statistics.mean(cross_val_score(gradient_booster, X_train, y_train, cv=cv)))
        print("Holdout Final Accuracy:", metrics.accuracy_score(y_test, y_pred))

        le = LabelEncoder()
        y_encoder = y_train
        print(type(y_train))
        y_encoder = le.fit_transform(y_train)
        model = XGBClassifier(learning_rate=0.1, max_depth=1, n_estimators=5)
        model.fit(X_train, y_encoder)
        plot_tree(model)
        plt.show()
        # Generalized Accuracy score:  0.755
        # Holdout Final Accuracy: 0.748

######################################################################################################################
# Question 2g         #
#######################
    def twog(self, X_train, X_test, y_train, y_test, cv, X, y):
        weights, params = [], []
        accuracy_scores = []
        '''
        for c in np.arange(-5, 2, dtype=float):
            lsvc = SVC(C=10 ** c, kernel='rbf', gamma='auto')
            lsvc.fit(X_train, y_train)
            lr_pred = lsvc.predict(X_test)
            score = metrics.accuracy_score(y_test, lr_pred)
            print("Accuracy score: ", score)
            print("c = ", c)
            accuracy_scores.append(score)
            params.append(10 ** c)
        weights = np.array(weights)
        print(params)
        '''

        # Decision region drawing
        plt.plot(params, accuracy_scores, color='blue', marker='x', label='Accuracy')
        plt.ylabel('accuracy score')
        plt.xlabel('C')
        plt.legend(loc='right')
        plt.xscale('log')
        plt.show()

        clf = SVC(kernel='rbf', C=0.00001, gamma='auto')
        clf.fit(X_train, y_train)
        clf_pred = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, clf_pred)
        print("Accuracy score: ", score)

        #  Holdout Model accuracy: 0.748
        # Feature importance is not possible in SVC using the rbf kernel

######################################################################################################################
# Question 2h         #
#######################
    def twoh(self, X_train, X_test, y_train, y_test, cv, X, y):
        le = LabelEncoder()
        y_encoder = y_train
        y_encoder = le.fit_transform(y_train)
        y_test_encoder = y_test
        y_test_encoder = le.fit_transform(y_test)
        accuracy_scores = []

        for i in np.arange(1,7):
            model = Sequential()
            model.add(Dense(12, input_dim=5, activation='relu'))
            for j in np.arange(1,i+1):
                model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_encoder, epochs=5, batch_size=10)
            # evaluate the keras model
            _, accuracy = model.evaluate(X_test, y_test_encoder)
            accuracy_scores.append(accuracy)
            print('Accuracy: %.2f' % (accuracy * 100))
        plt.plot(np.arange(1,7), accuracy_scores)
        plt.show()
        plt.clf()
        # Maximum accuracy achieved with 2 nodes, holdout accuracy 74.79%

        model = Sequential()
        model.add(Dense(12, input_dim=5, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_encoder, epochs=5, batch_size=10)

        neural_network = KerasClassifier(model=model,
                                         epochs=5,
                                         batch_size=10)

        # evaluate the keras model
        y_encoder = le.fit_transform(y)
        _, accuracy = model.evaluate(X, y_encoder)
        print("Generalized Accuracy: ", statistics.mean(cross_val_score(neural_network, X, y, cv=cv)))
        # Generalized accuracy:
        # Holdout accuracy

######################################################################################################################
# Question 2i         #
#######################
#
# Out of all of the tested model, the one with the highest generalized accuracy was a tie between the tree models and
# the logistical regression model with accuracies of 75.5%. Considering that each of these models were equivalent
# in accuracy, the simplest model should be used for interpretability. Personally, I would choose the decision tree
# classifier, as its interpretability is very high. These models produce great visuals demonstrating the model's
# workings and are easy to explain to laypeople
#

######################################################################################################################
# Preliminary Analysis  3      #
###############################
    def prelim3(self):
        krogerdf = pd.read_csv("KROGER.csv")
        originaldf = krogerdf.copy()
        krogerdf[krogerdf > 0].dropna()
        if (krogerdf <= 0).values.any():
            print("negative values")
        transformeddf = krogerdf.copy()
        lambda_list = []
        print(krogerdf.isnull().any())
        for column in transformeddf:
            print(column)
            transformeddf[column] += 0.001
            # Boxcox transform training data & save lambda value
            column_data = transformeddf[column]
            fitted_data, fitted_lambda = stats.boxcox(column_data)
            fitted_data = pd.DataFrame(fitted_data, columns=[column])
            # Standardize data around zero
            scaler = StandardScaler()
            X = scaler.fit_transform(fitted_data)
            scaled_data = pd.DataFrame(X, columns=[column])
            transformeddf.update(scaled_data)
            lambda_list.append(fitted_lambda)
        krogerdf = transformeddf
        print(krogerdf.head())
        return krogerdf, originaldf

######################################################################################################################
# Question 3a         #
#######################
    def threea(self, krogerdf):
        distortions = []
        K = range(1, 16)

        for k in K:
            kmeanModel = KMeans(n_clusters=k, max_iter=25, n_init=30)
            kmeanModel.fit(krogerdf)
            distortions.append(kmeanModel.inertia_)
        plt.figure(figsize=(16, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

        kmeanModel = KMeans(n_clusters=3, max_iter=25, n_init=30)
        kmeanModel.fit(krogerdf)
        labels = list(kmeanModel.labels_)
        print("Clients in each cluster\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}".format(labels.count(0),
                                                                                            labels.count(1),
                                                                                            labels.count(2)))
        centers = list(kmeanModel.cluster_centers_)
        for i in centers:
            k = 0
            for j in i:
                i[k] = round(i[k], 2)
                k += 1
        print(centers)
        print("Cluster centers:\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}".format(centers[0], centers[1],
                                                                                     centers[2]))
        # Three clusters exist. One where the center of the cluster of predictors is around 0. The second occurs where
        # the predictor values are all near their maximum. Finally, the last cluster exists when predictor values are
        # in the lowest quartile fo their range

######################################################################################################################
# Question 3b         #
#######################
    def threeb(self, krogerdf, originaldf):
        fraction = originaldf.copy()
        print(originaldf.head())
        for index in fraction.index:
            total = 0
            row = list(fraction.iloc[index])
            total = sum(row)
            i = 0
            for value in fraction.iloc[index]:
                fraction.iloc[index][i] = value / total
                i += 1
            row = list(fraction.iloc[index])
            total = sum(row)
        fraction.drop(['OTHER'], axis = 'columns', inplace=True)
        # Scale dataframe
        for column in fraction:
            fraction[column] += 0.001
            # Boxcox transform training data & save lambda value
            column_data = fraction[column]
            fitted_data, fitted_lambda = stats.boxcox(column_data)
            fitted_data = pd.DataFrame(fitted_data, columns=[column])
            # Standardize data around zero
            scaler = StandardScaler()
            X = scaler.fit_transform(fitted_data)
            scaled_data = pd.DataFrame(X, columns=[column])
            fraction.update(scaled_data)
        kroger_scaled = fraction

        plt.figure(figsize=(10, 7))
        plt.title("Wholesale Dendograms")
        dend = shc.dendrogram(shc.linkage(kroger_scaled, method='ward'))
        #plt.show()

        acmodel = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        acmodel.fit_predict(kroger_scaled)
        labels = list(acmodel.labels_)
        print("Clients in each cluster\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}".format(
            labels.count(0), labels.count(1), labels.count(2)))

        adf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        bdf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        cdf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        i = 0
        for client in labels:
            if client == 0:
                adf = adf.append(kroger_scaled.loc[i])
            if client == 1:
                bdf = bdf.append(kroger_scaled.loc[i])
            if client == 2:
                cdf = cdf.append(kroger_scaled.loc[i])
            i += 1
        centroids = []
        cluster_list = [adf, bdf, cdf]
        for cluster in cluster_list:
            alcohol_center = statistics.mean(list(cluster.loc[:, "ALCOHOL"]))
            baby_center = statistics.mean(list(cluster.loc[:, "BABY"]))
            cooking_center = statistics.mean(list(cluster.loc[:, "COOKING"]))
            fruit_center = statistics.mean(list(cluster.loc[:, "FRUITVEG"]))
            grain_center = statistics.mean(list(cluster.loc[:, "GRAIN"]))
            health_center = statistics.mean(list(cluster.loc[:, "HEALTH"]))
            house_center = statistics.mean(list(cluster.loc[:, "HOUSEHOLD"]))
            meat_center = statistics.mean(list(cluster.loc[:, "MEAT"]))
            pet_center = statistics.mean(list(cluster.loc[:, "PET"]))
            prepared_center = statistics.mean(list(cluster.loc[:, "PREPARED"]))
            snacks_center = statistics.mean(list(cluster.loc[:, "SNACKS"]))
            current_list = [alcohol_center, baby_center, cooking_center, fruit_center, grain_center,
                            health_center, house_center, meat_center, pet_center, prepared_center, snacks_center]
            centroids.append(current_list)

        print("Cluster centers:\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}".format(centroids[0],
                                                                                                    centroids[1],
                                                                                                    centroids[2]))

        # Cluster 1: wine moms
        # Cluster 2: Junk foodies
        # Cluster 3: Home cooks



        acmodel = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
        acmodel.fit_predict(kroger_scaled)
        labels = list(acmodel.labels_)

        adf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        bdf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        cdf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        ddf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        i = 0
        for client in labels:
            if client == 0:
                adf = adf.append(kroger_scaled.loc[i])
            if client == 1:
                bdf = bdf.append(kroger_scaled.loc[i])
            if client == 2:
                cdf = cdf.append(kroger_scaled.loc[i])
            if client == 3:
                ddf = ddf.append(kroger_scaled.loc[i])
            i += 1
        centroids = []
        cluster_list = [adf, bdf, cdf, ddf]
        for cluster in cluster_list:
            alcohol_center = statistics.mean(list(cluster.loc[:, "ALCOHOL"]))
            baby_center = statistics.mean(list(cluster.loc[:, "BABY"]))
            cooking_center = statistics.mean(list(cluster.loc[:, "COOKING"]))
            fruit_center = statistics.mean(list(cluster.loc[:, "FRUITVEG"]))
            grain_center = statistics.mean(list(cluster.loc[:, "GRAIN"]))
            health_center = statistics.mean(list(cluster.loc[:, "HEALTH"]))
            house_center = statistics.mean(list(cluster.loc[:, "HOUSEHOLD"]))
            meat_center = statistics.mean(list(cluster.loc[:, "MEAT"]))
            pet_center = statistics.mean(list(cluster.loc[:, "PET"]))
            prepared_center = statistics.mean(list(cluster.loc[:, "PREPARED"]))
            snacks_center = statistics.mean(list(cluster.loc[:, "SNACKS"]))
            current_list = [alcohol_center, baby_center, cooking_center, fruit_center, grain_center,
                            health_center, house_center, meat_center, pet_center, prepared_center, snacks_center]
            centroids.append(current_list)

        print("Clients in each cluster\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}\nCluster 4: {}".format(
            labels.count(0), labels.count(1), labels.count(2), labels.count(3)))
        print("Cluster centers:\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}\n Cluster 4: {}".format(centroids[0],
                                                                                                     centroids[1],
                                                                                                     centroids[2],
                                                                                                     centroids[3]))

        # Cluster 1: Junk foodies
        # Cluster 2: Drunk cooks
        # Cluster 3: Grocery shoppers
        # Cluster 4: Parents

        acmodel = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
        acmodel.fit_predict(kroger_scaled)
        labels = list(acmodel.labels_)

        adf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        bdf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        cdf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        ddf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        edf = pd.DataFrame(columns=['ALCOHOL', 'BABY', 'COOKING', 'FRUITVEG', 'GRAIN', 'HEALTH', 'HOUSEHOLD',
                                    'MEAT', 'PET', 'PREPARED', 'SNACKS'])
        i = 0
        for client in labels:
            if client == 0:
                adf = adf.append(kroger_scaled.loc[i])
            if client == 1:
                bdf = bdf.append(kroger_scaled.loc[i])
            if client == 2:
                cdf = cdf.append(kroger_scaled.loc[i])
            if client == 3:
                ddf = ddf.append(kroger_scaled.loc[i])
            if client == 4:
                edf = edf.append(kroger_scaled.loc[i])
            i += 1
        centroids = []
        cluster_list = [adf, bdf, cdf, ddf, edf]
        for cluster in cluster_list:
            alcohol_center = statistics.mean(list(cluster.loc[:, "ALCOHOL"]))
            baby_center = statistics.mean(list(cluster.loc[:, "BABY"]))
            cooking_center = statistics.mean(list(cluster.loc[:, "COOKING"]))
            fruit_center = statistics.mean(list(cluster.loc[:, "FRUITVEG"]))
            grain_center = statistics.mean(list(cluster.loc[:, "GRAIN"]))
            health_center = statistics.mean(list(cluster.loc[:, "HEALTH"]))
            house_center = statistics.mean(list(cluster.loc[:, "HOUSEHOLD"]))
            meat_center = statistics.mean(list(cluster.loc[:, "MEAT"]))
            pet_center = statistics.mean(list(cluster.loc[:, "PET"]))
            prepared_center = statistics.mean(list(cluster.loc[:, "PREPARED"]))
            snacks_center = statistics.mean(list(cluster.loc[:, "SNACKS"]))
            current_list = [alcohol_center, baby_center, cooking_center, fruit_center, grain_center,
                            health_center, house_center, meat_center, pet_center, prepared_center, snacks_center]
            centroids.append(current_list)

        print(
            "Clients in each cluster\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}\nCluster 4: {}, \nCluster 5: {}".format(
                labels.count(0), labels.count(1), labels.count(2), labels.count(3), labels.count(4)))
        print("Cluster centers:\nCluster 1: {}\nCluster 2: {}\nCluster 3: {}\n Cluster 4: {}\n Cluster 5: {}".format(
            centroids[0],
            centroids[1],
            centroids[2],
            centroids[3],
            centroids[4]))

        # Cluster 1: Grab-and-go
        # Cluster 2: Booze and beef
        # Cluster 3: Grocery shoppers
        # Cluster 4: Parents
        # Cluster 5: "just need one thing"


if __name__ == '__main__':
    final = final()
    final.main()
