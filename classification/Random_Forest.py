import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#LOAD DATASET
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

#features and targets
x = df[["QUANTITYORDERED", "PRICEEACH", "MSRP"]].values
y = (df["DEALSIZE"] != "Small").astype(int).values #1=large/medium, 0="small"

#train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, # no. of trees
    max_depth=6, #depth of each tree, also we defined the depth to avoid overfitting
    random_state=42
)
rf_model.fit(x_train, y_train)

#PREDICTIONS
y_pred = rf_model.predict(x_test)

#accuracy and report
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest accuracy:{accuracy:.4f} ")
print("Classification Report: ")
print(classification_report(y_test, y_pred, target_names=["Small", "Medium/Large"]))

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix: ")
print(cm)

#feature importance
features = ["QUANTITYORDERED", "PRICEEACH", "MSRP"]
print("feature importance: ")
for feat, importance in zip(features, rf_model.feature_importances_):
    print(f"{feat}: {importance:.4f}")

plt.figure(figsize=(16,6))

#plot1
plt.subplot(1,3,1)
tree_counts = [10,50,100,200,300]
accuracies = []
for n in tree_counts:
    rf = RandomForestClassifier(n_estimators=n, max_depth=6, random_state=42)
    rf.fit(x_train, y_train)
    acc = rf.score(x_test, y_test)
    accuracies.append(acc)

plt.plot(tree_counts, accuracies, marker="o", color="green", linewidth=2)
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Trees")
plt.grid(alpha=0.3)

#plto2: feature importance
plt.subplot(1,3,2)
plt.barh(features, rf_model.feature_importances_, color="orange")
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

#plot3: Confusion matrix
plt.subplot(1,3,3)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Small", "Medium/Large"])
plt.yticks([0,1], ["Small", "Medium/Large"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="Black")

plt.tight_layout()
plt.savefig("random_forest.png")
plt.show()

