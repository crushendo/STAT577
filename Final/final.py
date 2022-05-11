import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


class final():
    def main(self):
        X_train, X_test, y_train, y_test = final.prelim()
        print(X_train.head())
        #final.onea(X_train, X_test, y_train, y_test)
        #final.oneb(X_train, X_test, y_train, y_test)
        #final.onec(X_train, X_test, y_train, y_test)
        #final.oned(X_train, X_test, y_train, y_test)
        #final.onee(X_train, X_test, y_train, y_test)
        final.onef(X_train, X_test, y_train, y_test)
        #final.oneg(X_train, X_test, y_train, y_test)
        #X_train, X_test, y_train, y_test = final.prelim()

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
        return X_train, X_test, y_train, y_test

######################################################################################################################
# Question 1a         #
#######################
    def onea(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_train)
        print('Generalized Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
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

        # Generalized RMSE: 26.36; Holdout RMSE: 26.17
        # By far the most significant predictors are mean (positive), heatinghouyrs (positive), and coolinghours (neg)

######################################################################################################################
# Question 1b         #
#######################
    def oneb(self, X_train, X_test, y_train, y_test):
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
        rr_pred = rr.predict(X_train)
        print('Generalized Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, rr_pred)))
        rr_coef = rr.coef_
        rr_pred = rr.predict(X_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rr_pred)))
        predictor_list = X_train.columns
        predictor_importance = list(zip(rr.coef_, predictor_list))
        # plot feature importance
        plt.bar([x[1] for x in predictor_importance], [x[0] for x in predictor_importance])
        plt.show()
        plt.clf()
        # Generalized RMSE: 26.36; Holdout RMSE: 26.17
        # By far the most significant predictors are mean (positive), heatinghours (positive), and coolinghours (neg)

######################################################################################################################
# Question 1c         #
#######################
    def onec(self, X_train, X_test, y_train, y_test):
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
        knn_pred = knn.predict(X_train)
        print('Generalized Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, knn_pred)))
        knn_pred = knn.predict(X_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, knn_pred)))

######################################################################################################################
# Question 1d         #
#######################
    def oned(self, X_train, X_test, y_train, y_test):
        tree = DecisionTreeRegressor()
        tree.fit(X_train, y_train)
        tree_pred = tree.predict(X_train)
        print('Generalized Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, tree_pred)))
        tree_pred = tree.predict(X_test)
        print('Generalized Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, tree_pred)))
        predictor_importance = list(tree.feature_importances_)
        predictor_list = X_train.columns
        # plot feature importance
        plt.bar(predictor_list, predictor_importance)
        plt.show()
        plt.clf()
        # By far the most significant predictor was mean, followed by heatinghours

######################################################################################################################
# Question 1e         #
#######################
    def onee(self, X_train, X_test, y_train, y_test):
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
        tree_pred = regressor.predict(X_train)
        print('Generalized Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, tree_pred)))
        tree_pred = regressor.predict(X_test)
        print('Generalized Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, tree_pred)))
        predictor_importance = list(regressor.feature_importances_)
        predictor_list = X_train.columns
        # plot feature importance
        plt.bar(predictor_list, predictor_importance)
        plt.show()
        plt.clf()
        # The most important predictors were heatinghours and mean

######################################################################################################################
# Question 1f         #
#######################
    def onef(self, X_train, X_test, y_train, y_test):
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
        boost_pred = gradient_booster.predict(X_train)
        print('Generalized Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, boost_pred)))
        boost_pred = gradient_booster.predict(X_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, boost_pred)))
        # Generalised RMSE: 26.1; Holdout RMSE: 26.1

######################################################################################################################
# Question 1g         #
#######################
    def oneg(self, X_train, X_test, y_train, y_test):
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
        history = model.fit(X_train, y_train)
        print('Generalized Root Mean Squared Error:', np.sqrt(history.history['mse']))
        history = model.fit(X_test, y_test)
        print('Holdout Root Mean Squared Error:', np.sqrt(history.history['mse']))
        # Generalised RMSE: 38.8; Holdout RMSE: 31.7

######################################################################################################################
# Question 1h         #
#######################


######################################################################################################################
# Preliminary Analysis 2      #
###############################
    def prelim2(self):
        purchasedf = pd.read_csv("PURCHASE.csv")
        X = purchasedf.drop(['Purchase'], axis = 'columns')
        y = purchasedf.Purchase
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        X = scaled
        print(X.head())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=577)
        return X_train, X_test, y_train, y_test

######################################################################################################################
# Question 2a         #
#######################

if __name__ == '__main__':
    final = final()
    final.main()
