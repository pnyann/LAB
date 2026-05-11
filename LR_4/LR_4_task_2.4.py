import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("R2 =", r2_score(ytest, ypred))
print("MAE =", mean_absolute_error(ytest, ypred))
print("MSE =", mean_squared_error(ytest, ypred))

plt.scatter(ytest, ypred)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("Виміряно")
plt.ylabel("Передбачено")
plt.title("Регресія для набору diabetes")
plt.show()
