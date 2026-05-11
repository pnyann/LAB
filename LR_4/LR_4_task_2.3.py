from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm
from sklearn import linear_model
import numpy as np

input_file = 'data_multivar_regr.txt'

data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

print("Linear R2 =", round(sm.r2_score(y_test, y_pred), 2))

polynomial = PolynomialFeatures(degree=10)
X_train_poly = polynomial.fit_transform(X_train)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

datapoint = [[7.75, 6.35, 5.56]]
poly_point = polynomial.fit_transform(datapoint)

print("Linear:", linear_regressor.predict(datapoint))
print("Polynomial:", poly_model.predict(poly_point))
