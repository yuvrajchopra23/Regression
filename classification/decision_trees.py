import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load data
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

x = df[["QUANTITYORDERED", "PRICEEACH", "MSRP"]].values
y = (df["DEALSIZE"] != "Small").astype(int).values

#train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#no SCALING needed as this is different from knn
#decision trees split the data based on simple thresholds (eg. "is priceeach > 80?")
#they dont measure distance between points, so feature scaling doesn't matter here.
#this is an important differ from knn

#TRAIN the model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Decision tree accuracy (max_depth=4), {accuracy:.4f}")
print("Full classification report: ")
print(classification_report(y_test, y_pred, target_names=["Small", "Medium/Large"]))

cm = confusion_matrix(y_test, y_pred)

#Which feature mattered the most??? trees have this feature which tells us that which feature was the most important and which was the least important.
print("Feature important??:")
features = ["QUANTITYORDERED", "PRICEEACH", "MSRP"]
for feat, importance in zip(features, model.feature_importances_):
    print(f"{feat:20s}: {importance:.4f}")

#Compare different tree depths: same as testing different values of k in KNN
print("comparing different depths:")
depths = [2,3,4,5,6,8,10,None]
accuracies = []

for d in depths:
    model_d = DecisionTreeClassifier(max_depth=d, random_state=42)
    model_d.fit(x_train, y_train)
    acc_d = model_d.score(x_test, y_test)
    accuracies.append(acc_d)
    label = d if d is not None else "None(unlimited)"
    print(f"max_depth={label!s:18s} -> Accuracy:{acc_d:.4f}")

print("For comparison: ")
print("Logistic Regression (sklearn, unseen):  0.9381")
print("KNN (k=5, unseen):                       0.9593")
print("KNN (best k=30, unseen):                 0.9628")
print(f"Decision Tree (max_depth=4, unseen):     {accuracy:.4f}")

#VISUALIZE:

plt.figure(figsize=(16,6))

#plot1: actual tree structure
plt.subplot(1,3,1)
plot_tree(model, feature_names=features, class_names=["Small", "Medium/Large"], filled=True, fontsize=7, max_depth=2)#showing only top 2 levels for readability
plt.title("Decision tree structure(top 2 levels shown)")

#plot2: accuracy vs depth
plt.subplot(1,3,2)
depth_label = [str(d) if d is not None else "None" for d in depths]
plt.plot(range(len(depths)), accuracies, marker="o", color="#BA7517", linewidth=2)
plt.xticks(range(len(depths)), depth_label)
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Tree depth")
plt.grid(alpha=0.3)

#plot3:feature importance bar chart
plt.subplot(1,3,3)
plt.barh(features, model.feature_importances_, color="#3C3489")
plt.xlabel("Importance")
plt.title("Which feature Mattered the most?")

plt.tight_layout()
plt.savefig("Decision_tree_result.png", dpi=150)
plt.show()