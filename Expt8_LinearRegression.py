import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
lr_model = LinearRegression()
lr_model.fit(X, y)
y_pred = lr_model.predict(X)
plt.scatter(X, y, label="Data")
plt.plot(X, y_pred, color='red', label="Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
intercept = lr_model.intercept_
slope = lr_model.coef_
print(f"Intercept (theta_0): {intercept[0]}")
print(f"Slope (theta_1): {slope[0]}")
