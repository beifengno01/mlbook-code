"""
http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets.california_housing import fetch_california_housing
housing = fetch_california_housing()

print(housing.feature_names)
X, y = housing.data, housing.target

# split 80/20 train-test
N = len(X)
X_train = X[0:int(.8*N), :]
y_train = y[0:int(.8*N)]
X_test = X[int(.8*N):N, :]
y_test = y[int(.8*N):N]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print(y_train)

