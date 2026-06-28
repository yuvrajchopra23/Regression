import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#load data
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

x = df[["QUANTITYORDERED", "PRICEEACH", "MSRP"]].values
y = df["SALES"].values

#train test split: it is used to train and test the model on the same data set
#train on the 80% data of the training set and use other 20% of the data to test the model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training Samples: ", x_train.shape[0])
print("Testing Samples: ", x_test.shape[0])

#CREATE AND TRAIN THE MODEL: this replaces our entire gradient descent loop(normalization, init weights, 1000 interactions)

model = LinearRegression()
model.fit(x_train, y_train) #this will train the model according to the data as model is already created as an object which is linearregression()

#PREDICT
y_pred = model.predict(x_test)

#EVALUATE: same metrics, but sklearn computes for us
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R2: {r2:.4f}")

print(f"\nFor comparison — your from-scratch R² (trained AND tested on same data): 0.7959")
print(f"sklearn R² (trained on 80%, tested on UNSEEN 20%): {r2:.4f}")
print("\nIf this number is similar, it confirms your from-scratch model was genuinely correct,")
print("not just memorizing the training data.")

#inspect the learned weights: compare to your own w, b
print(f"sklearn weights: {model.coef_}")
print(f"sklearn bias: {model.intercept_:.4f}")
print("Note: these won't match your weights exactly, because sklearn trains on")
print("raw unnormalized data while we normalized - but the Predictions should be same")