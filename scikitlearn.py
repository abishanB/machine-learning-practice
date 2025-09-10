from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import numpy as np


Xy = np.loadtxt("./CaliforniaHousing/cal_housing.data", delimiter=",")
X = Xy[:, :-1]
y = Xy[:, -1]
print(X.shape, y.shape)
mod = KNeighborsRegressor().fit(X, y)

pred = mod.predict(X)
plt.scatter(pred, y)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.title("KNN Predictions vs Actual")
plt.show()
