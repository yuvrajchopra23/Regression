import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load data
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

#now here we are using 3 features instead of 1
x = df[["QUANTITYORDERED", "PRICEEACH", "MSRP"]].values #shape: (m,3)
y = df["SALES"].values.reshape(-1,1) #shape: (m,1)

print("X shape: ", x.shape)
print("Y shape: ", y.shape)

#NORMALIZE EACH FEATURE

#we must normalize each column seperately

x_mean = x.mean(axis=0) #axis = 0 -> mean of each column(one mean per feature)
x_std = x.std(axis=0) #one std per feature

x_norm = (x - x_mean)/x_std

#INITIAL THE PARAMETERS
m, n = x_norm.shape #m = rows, n= no. of features(3)

w = np. zeros((n,1)) #one weight per feature
b = 0
alpha = 0.01
iterations = 1000
losses = []

# GRADIENT DESCENT

for i in range(iterations):
    #prediction: we'll do now matrix multiplication instead of simple multiplication
    y_pred = np.dot(x_norm , w) + b
    error = y_pred - y

    #gradients
    dw = (1/m) * np.dot(x_norm.T, error) #x transposed @ error -> gradient per feature
    db = (1/m) * np.sum(error)

    w = w - alpha * dw
    b = b - alpha * db

    J = (1/(2*m)) * np.sum(error**2)
    losses.append(J)

print("final weights(one per feature): ")
print(f"w_quantity= {w[0][0]:.4f}")
print(f"w_price= {w[1][0]:.4f}")
print(f"w_msrp= {w[2][0]:.4f}")
print(f"final bias= {b:.4f}")\

#PREDICTION FUNCTION
def predict(quantity, price, msrp):
    x_input = np.array([quantity,price, msrp])
    x_input_norm = (x_input - x_mean) / x_std
    return np.dot(x_input_norm, w)[0] + b

print(f"Predicted sales (qty=20, price=80, msrp=100): {predict(20, 80, 100):.2f}")
print(f"Predicted sales (qty=50, price=95, msrp=150): {predict(50, 95, 150):.2f}")

#EVALUATE 
y_pred_final = np.dot(x_norm , w) + b

mse = np.mean((y - y_pred_final)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred_final))

ss_res = np.sum((y - y_pred_final)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res/ss_tot)

print("\n--- Model Evaluation (Multiple Regression) ---")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R2:   {r2:.4f}")

#compare with linear regressions r2
print(f"\nPrevious single-feature R² (Quantity only): ~0.30")
print(f"New multiple-feature R² (Quantity + Price + MSRP): {r2:.4f}")
print("If this is higher, it confirms: more relevant features = better predictions")

#PLOT

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(iterations), losses)
plt.xlabel("Iterations")
plt.ylabel("COST J(w,b)")
plt.title("Gradient Descent Convergence")

plt.subplot(1,2,2)
plt.scatter(y, y_pred_final, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(),y.max()], color="red", linestyle="--")#perfect prediction line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("predicted vs actual sales")
plt.tight_layout()

plt.show()