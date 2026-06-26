import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

#define x and y
x = df[["QUANTITYORDERED","PRICEEACH","MSRP"]].values #shape (m,3)

#target (y): Classify deal size as Medium/Large (1) vs small (0)
#this converts small to 0 and large/medium to 1
y = (df["DEALSIZE"] != "Small").astype(int).values.reshape(-1,1) #shape(m,1)

#NORMALIZE X
x_mean = x.mean(axis=0)
x_std = x.std(axis=0)
x_norm = (x - x_mean)/x_std

#INITIALIZE THE PARAM
m, n = x_norm.shape #m=rows, n= features (3)

w = np.zeros((n,1))#1 weight per features
b = 0
alpha = 0.1 #learning rate a bit larger for classification
iterations = 1000
losses = []

#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

#gradient descent
for i in range(iterations):
    #prediction using the sigmoid function
    z = np.dot(x_norm, w) + b
    y_pred = sigmoid(z)

    #errors
    error = y_pred - y

    #gradients
    dw = (1/m) * np.dot(x_norm.T, error)
    db = (1/m) * np.sum(error)

    #update the params
    w = w - alpha * dw
    b = b - alpha * db

    #cost function J(w,b) for classification 
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15) #to avoid log(0) error
    J = -(1/m) *np.sum(y * np.log(y_pred_clipped) + (1 - y)*np.log(1 - y_pred_clipped))

    losses.append(J)

print("final weights(one per feature): ")
print(f"w_quantity = {w[0][0]:.4f}")
print(f"w_price = {w[1][0]:.4f}")
print(f"w_msrp = {w[2][0]:.4f}")
print(f"final bias = {b:.4f}")

#prediction function
def predict(quantity, price, msrp):
    x_input = np.array([quantity, price, msrp])
    x_input_norm = (x_input - x_mean)/x_std
    #calculate the PROBABILITY
    z_input = np.dot(x_input_norm, w)[0] + b
    prob = sigmoid(z_input)    
    #if probability >= 0.5, we classify it as medium/large (1), else small(0)
    pred_class = "Medium/Large" if prob >= 0.5 else "Small"
    return prob, pred_class

print(f"Predicted sales category (qty=20, price=80, msrp=100): {predict(20, 80, 100)}")
print(f"Predicted sales category (qty=50, price=95, msrp=150): {predict(50, 95, 150)}")

#EVALUATE
#get predictions for all samples(probabilities)
z_all = np.dot(x_norm, w) + b
y_pred_prob = sigmoid(z_all)

#convert prob to binary class predictions (0 or 1)
y_pred_final = (y_pred_prob >= 0.5).astype(int)

#Calculate TP, TN, FP, FN
tp = np.sum((y == 1) & (y_pred_final == 1))
tn = np.sum((y == 0) & (y_pred_final == 0))
fp = np.sum((y == 0) & (y_pred_final == 1))
fn = np.sum((y == 1) & (y_pred_final == 0))

#classification metrics
accuracy = (tp + tn) / m
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp/(tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision * recall) > 0 else 0

print("\n--- Model Evaluation (Logistic Regression) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

#PLOT
plt.figure(figsize=(12,5))

#subplot1
plt.subplot(1,2,1)
plt.plot(range(iterations), losses, color="purple")
plt.xlabel("iterations")
plt.ylabel("cost J(w,b)")
plt.title("Gradient descent convergence")

#subplot2
plt.subplot(1,2,2)
plt.scatter(range(len(y)), y_pred_prob, c=y, cmap="bwr",alpha= 0.3)
plt.axhline(0.5, color="black", linestyle="--", label="threshold (0.5)")
plt.xlabel("Sample index")
plt.ylabel("predicted probability")
plt.title("Predicted probabilities (blue=small, red=medium/large)")
plt.legend()

plt.tight_layout()
plt.savefig("logistic_regression_res.png")
plt.show()