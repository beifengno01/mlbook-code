"""
http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
download this file then use pandas to load. sklearn.datasets.california_housing seems weird
http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math

housing = pd.read_csv('../data/CaliforniaHousing.csv')
#":type : pd.core.frame.DataFrame"

print(housing.columns)
X, y = housing.drop('medianHouseValue', axis=1), housing['medianHouseValue']

# split 80/20 train-test
# N = len(X)
# X_train = X[0:int(.8*N), :]
# y_train = y[0:int(.8*N)]
# X_test = X[int(.8*N):N, :]
# y_test = y[int(.8*N):N]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(X_train)
# print(y_train)

regr = RandomForestRegressor(n_estimators=100, max_features="sqrt", max_leaf_nodes=3, oob_score=True)
regr.fit(X_train, y_train)

print(regr.feature_importances_)
print(regr.oob_score_)

prediction = regr.predict(X_test)
print(prediction - y_test)