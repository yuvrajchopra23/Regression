import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


#LOAD DATA
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

x = df[["QUANTITYORDERED", "PRICEEACH", "MSRP"]].values
y = (df["DEALSIZE"] != "Small").astype(int).values

#train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

#SCALE the Features: important for KNN specifically
    #KNN measures distance between points, if one feature ranges 0-100000(like msrp)
    #and another ranges 0-100(like quantity ordered), the big range feature will
    #completely dominate the distance calculation, even if it's not more important.
    #Scaling puts every feature on the same footing

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)#learn mean and std from training data only
x_test_scaled = scaler.transform(x_test) #apply the same mean/std to test data
#here fit_transform() fit represents that whatever the function is applied on the training data

#CREATE AND TRAIN THE KNN MODEL
#n_neighbors = 5 means: to classify a new point, look at its 5 closest
#neighbors in the training data, and take a majority vote among them.

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_scaled, y_train)

#PREDICT AND EVALUATE
y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN accuracy (k=5): {accuracy:.4f}")
print("full classification report: ")
print(classification_report(y_test, y_pred, target_names=["Small", "Medium/Large"]))
cm = confusion_matrix(y_test, y_pred)

#try few different values of K: see how it affects??

k_values = [1,3,5,7,9,13,15,18,21,25,27,30]
accuracies = []
for k in k_values:
    model_k = KNeighborsClassifier(n_neighbors=k)
    model_k.fit(x_train_scaled, y_train)
    acc_k = model_k.score(x_test_scaled, y_test)
    accuracies.append(acc_k)
    print(f"k ={k} -> Accuracy: {acc_k:.4f}")


best_k = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"best accuracy: {best_acc} for k: {best_k}")

print("\nFor comparison:")
print("Your from-scratch Logistic Regression accuracy: 0.9408")
print(f"KNN (k=5, unseen test set): {accuracies[2]:.4f}")


#VISUALIZE
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(k_values, accuracies, marker="o", color="#3c3489", linewidth=2)
plt.axvline(best_k, color="red", linestyle="--", alpha=0.5, label=f"Best k:{best_k}")
plt.xlabel("k_Values")
plt.ylabel("Accuracy")
plt.title("KNN accuracy vs k_values")
plt.legend()
plt.grid(alpha=0.3)

#confusion matrix as graph
plt.subplot(1,2,2)
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0,1], ["Small","Medium/Large"])
plt.yticks([0,1],["Small", "Medium/Large"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (k=5, Accuracy={accuracy:.2%})")

#annotate each cell with its count
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(cm[i,j]), ha="center", va="center",
                 color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=14)
        
plt.tight_layout()
plt.savefig("knn_result.png", dpi=150)
plt.show()