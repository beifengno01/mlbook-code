"""
http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
download this file then use pandas to load. sklearn.datasets.california_housing seems weird
http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import math
import numpy as np

housing = pd.read_csv('data/CaliforniaHousing.csv')
#":type : pd.core.frame.DataFrame"

#print(housing.describe())
X, y = housing.drop('medianHouseValue', axis=1), housing['medianHouseValue']

# y = np.log(y)

# split 80/20 train-test
# N = len(X)
# X_train = X[0:int(.8*N), :]
# y_train = y[0:int(.8*N)]
# X_test = X[int(.8*N):N, :]
# y_test = y[int(.8*N):N]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(X_train)
# print(y_train)

def boost():
    gbrt=GradientBoostingRegressor(n_estimators=100)
    gbrt.fit(X_train, y_train)
    y_pred=gbrt.predict(X_test)
    print("Feature Importances")
    print(gbrt.feature_importances_)

    #Let's print(the R-squared value for train/test. This explains how much of the variance in the data our model is
    #able to decipher.
    print("R-squared for Train: %.2f" %gbrt.score(X_train, y_train))
    print("R-squared for Test: %.2f" %gbrt.score(X_test, y_test))

    print((y_pred - y_test)/y_test)

def rf(max_leaf_nodes):
    # regr = RandomForestRegressor(n_estimators=100, max_leaf_nodes=max_leaf_nodes, oob_score=True)
    regr = RandomForestRegressor(oob_score=True)
    regr.fit(X_train, y_train)
    print(regr)

    print(regr.feature_importances_)
    print(regr.oob_score_)
    #
    # y_pred = regr.predict(X_test)
    # # print(y_pred - y_test)
    print("R-squared for Train: %.2f" % regr.score(X_train, y_train))
    print("R-squared for Test: %.2f" % regr.score(X_test, y_test))

rf(2)
# for i in range(1500,2000,30):
#     print("####", i)
#     rf(i)