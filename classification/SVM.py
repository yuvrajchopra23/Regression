import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Load data
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

x = df[["QUANTITYORDERED", "PRICEEACH", "MSRP"]].values
y = (df["DEALSIZE"] != "Small").astype(int).values

#train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#Scaling to be done for SVM's due the same reason as knn that it works on the distance.
#SVM cares about the distance/margins between points
#unscaled features would distort the margin calculation

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Train the model
model = SVC(kernel="linear", C=1.0)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy (linear kernel, C=1.0): {accuracy:.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Small", "Medium/Large"]))

cm = confusion_matrix(y_test, y_pred)
print(f"Number of Support vectors: {model.n_support_}")
print("(these are the only points that actually define the boundary)")

#Compare different kernels: linear vs rbf(curved boundary)
print("\nComparing kernels")
kernels = ["linear", "rbf", "poly"]
kernel_accuracies = []

for k in kernels:
    model_k = SVC(kernel=k, C=1.0)
    model_k.fit(x_train_scaled, y_train)
    acc_k = model_k.score(x_test_scaled, y_test)
    kernel_accuracies.append(acc_k)
    print(f"kernel: {k:8s} -> accuracy: {acc_k:.4f}")

#Compare different C values
best_kernel = kernels[np.argmax(kernel_accuracies)]
print("Comparing C values")
C_values = [0.01, 0.1, 1, 10, 100]
c_accuracies = []

for c in C_values:
    model_c = SVC(kernel=best_kernel, C = c)
    model_c.fit(x_train_scaled, y_train)
    acc_c = model_c.score(x_test_scaled, y_test)
    c_accuracies.append(acc_c)
    print(f"C: {c:6} -> accuracy: {acc_c:.4f}")

print("\nFor comparison:")
print("Logistic Regression (sklearn, unseen):  0.9381")
print("KNN (best k=30, unseen):                0.9628")
print("Decision Tree (max_depth=6, unseen):    0.9681")
print(f"SVM (linear, C=1.0, unseen):           {accuracy:.4f}")

#VISUALIZE:

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.bar(kernels, kernel_accuracies, color=["#3c3489", "#085041", "#BA7517"])
plt.ylabel("Accuracy")
plt.title("SVM Accuracy by Kernel Type")
plt.ylim(0.8, 1.0)

plt.subplot(1,2,2)
plt.plot([str(c) for c in C_values], c_accuracies, marker="o", color="#378add", linewidth=2)
plt.xlabel(f"C (kernel={best_kernel})")
plt.ylabel("Accuracy")
plt.title("Accuracy vs C (margin Strictness)")
plt.grid(alpha=0.3)

# Classification graph using QUANTITYORDERED vs PRICEEACH
# Best parameters: kernel=rbf, C=100

X_vis = x[:, :2]   # take first two features for visualization
y_vis = y

# scale
scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

# train SVM with best parameters
model_vis = SVC(kernel="rbf", C=100)
model_vis.fit(X_vis_scaled, y_vis)

# create mesh grid
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# predict over grid
Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# plot decision boundary
plt.figure(figsize=(7,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=y_vis,
            edgecolors="k", cmap=plt.cm.coolwarm, alpha=0.8)
plt.xlabel("QUANTITYORDERED (scaled)")
plt.ylabel("PRICEEACH (scaled)")
plt.title("SVM Classification Boundary (RBF Kernel, C=100)")

plt.tight_layout()
plt.savefig("svm_results1.png", dpi=150)
plt.show()