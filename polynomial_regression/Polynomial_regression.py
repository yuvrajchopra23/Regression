import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

#select features and target variable
x = df[["QUANTITYORDERED", "PRICEEACH", "MSRP"]].values #shape: (m,3)
y = df["SALES"].values.reshape(-1,1) #shape: (m,1)

#normalize y
y_mean = y.mean()
y_std = y.std()
y_norm = (y- y_mean)/y_std

#POLYNOMIAL EXPANSION using square roots and interactions
x_poly = []
for row in x:
    q, p, m = row
    x_poly.append([
        q, p, m, #original features
        np.sqrt(q), np.sqrt(p), np.sqrt(m), #square roots
        q*p #interactions
    ])

x_poly = np.array(x_poly) #shape: (m,7)

#normalize
x_poly_mean = x_poly.mean(axis=0)
x_poly_std = x_poly.std(axis=0)
x_poly = (x_poly -x_poly_mean) / x_poly_std
m, n = x_poly.shape #m = rows, n= no. of features

#initialize parameters
w = np.zeros((n,1)) #one weight per feature
b = 0
alpha = 0.01
iterations = 1000
losses = []

#GRADIENT DESCENT
for i in range(iterations):
    #predictions
    y_pred_norm = np.dot(x_poly, w) + b
    error = y_pred_norm - y_norm

    #gradients
    dw = (1/m)*np.dot(x_poly.T, error)
    db = (1/m)*np.sum(error)

    #update parameters
    w = w - alpha * dw
    b = b - alpha * db

    #cost function
    J = (1/(2*m)) * np.sum(error**2)
    losses.append(J)

#PREDICTION FUNCTION
def predict(quantity, price, msrp):
    x_input = np.array([
        quantity, price, msrp,
        np.sqrt(quantity), np.sqrt(price), np.sqrt(msrp),
        quantity*price
    ])
    x_input_norm = (x_input - x_poly_mean) / x_poly_std
    y_pred_norm = np.dot(x_input_norm, w) + b
    return float((y_pred_norm*y_std + y_mean).squeeze()) #rescale back to original sales unit

print(f"Predicted sales (qty=20, price=80, msrp=100): {predict(20, 80, 100):.2f}")
print(f"Predicted sales (qty=50, price=95, msrp=150): {predict(50, 95, 150):.2f}")


# Predictions
y_pred_norm = np.dot(x_poly, w) + b
y_pred = y_pred_norm*y_std +y_mean
#evaluations
mse = np.mean((y - y_pred)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred))
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)

print("final weights(one per feature): ")
print(f"Final weights: {w.flatten()}")
print(f"Final bias: {b:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Visualization
plt.figure(figsize=(15,8))

# Gradient Descent Convergence
plt.subplot(2,3,1)
plt.plot(range(iterations), losses, color="blue")
plt.xlabel("Iterations")
plt.ylabel("COST J(w,b)")
plt.title("Gradient Descent Convergence")

# Predicted vs Actual
plt.subplot(2,3,2)
plt.scatter(y, y_pred, color="blue", alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Predicted vs Actual Sales")

#Quantity vs Sales
plt.subplot(2,3,3)
plt.scatter(df["QUANTITYORDERED"], df["SALES"], alpha=0.5, color="blue")
plt.xlabel("Quantity Ordered")
plt.ylabel("Sales ($)")
plt.title("Quantity vs Sales")

# Price vs Sales
plt.subplot(2,3,4)
plt.scatter(df["PRICEEACH"], df["SALES"], alpha=0.5, color="green")
plt.xlabel("Price Each ($)")
plt.ylabel("Sales ($)")
plt.title("Price vs Sales")

# MSRP vs Sales
plt.subplot(2,3,5)
plt.scatter(df["MSRP"], df["SALES"], alpha=0.5, color="orange")
plt.xlabel("MSRP ($)")
plt.ylabel("Sales ($)")
plt.title("MSRP vs Sales")

plt.tight_layout()
plt.savefig("polynomial_regression_res.png")
plt.show()