"""
for stuff on techniques used for mnist

http://yann.lecun.com/exdb/mnist/
"""
from sklearn.datasets import load_digits
from sklearn.datasets.california_housing import fetch_california_housing
# write data into ./mldata/mnist-original.mat
mnist = load_digits(return_X_y=True)

housing = fetch_california_housing(data_home='.')
