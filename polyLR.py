import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

predict_price = model.predict([[13]])
print(f"predicted value: {predict_price[0]:.2f}")

y_pred = model.predict(X_poly)

plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(X, y_pred, color='red', label='Polynomial Regression Curve')
plt.title("Polynomial Regression Example (y = x^2)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
