import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# first we will load the dataset
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

# define X and Y
x = df["QUANTITYORDERED"].values #array
y = df["SALES"].values #array

#change the array into a matrix
x = x.reshape(-1,1) # made the array of 1d into 2D
y = y.reshape(-1,1)

#NORMALIZE X (this is completely optional it makes the data smaller for ex if one value is 10000 then it will decrease it to some points like 0.56893)
x_mean = x.mean()
x_std = x.std()
x = (x - x.mean()) / x.std()

#parameters
m = len(y) #no. of rows
w = 0 #slope
b = 0 #intercept/bias
alpha = 0.01 #learning rate
iterations = 1000 #no. of steps to be taken for gradient descent

losses = []

#Gradient Descent
for i in range(iterations):
    #prediction f(x) = wx + b
    y_pred = w * x + b

    #errors
    error = y_pred - y

    #Gradients
    dw = (1/m)*np.sum(error * x)
    db = (1/m)*np.sum(error)

    #update the parameters
    w = w - alpha * dw
    b = b - alpha * db

    #cost function J(w,b)
    J = (1/(2*m)) * np.sum(error**2)
    losses.append(J)

print("final w(slope) :",w)
print("final b(intercet): ",b)

# subplot1: PLOT THE REGRESSION LINE
plt.figure(figsize=(12,5))
plt.subplot(1,2,1) #1 row, 2 columns, first plot
plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, w*x + b, color="red", label="regression Line")
plt.xlabel("Units Sold (Normalized)")
plt.ylabel("Sales")
plt.legend()
plt.title("Linear Regression Fit")

#Subplor2: PLOT the COST function convergence
plt.subplot(1,2,2)
plt.plot(range(iterations), losses)
plt.xlabel("Iterations")
plt.ylabel("COST J(w,b)")
plt.title("Gradient Descent Convergence")
plt.tight_layout()

plt.show()

print("len of dataset: ",m)

def predict(Quantity_ordered, x_mean, x_std):
    x_norm = (Quantity_ordered - x_mean)/x_std
    return w * x_norm + b

print("predicted Sales for 20 units: ", predict(20,x_mean,x_std))
print("predicted sales for 50 units: ", predict(50,x_mean,x_std))


#NOW to check the Accuracy of the model

#predictions
y_pred = w * x + b

#MEAN SQUARED ERROR
mse = np.mean((y - y_pred)**2)

#ROOT MEAN SQUARED ERROR
rmse = np.sqrt(mse)

#MEAN ABSOLUTE ERROR
mae = np.mean(np.abs(y - y_pred))

#R square (Coefficient of Determination)
ss_res = np.sum((y - y_pred)**2) #residua sum of squares
ss_tot = np.sum((y - np.mean(y))**2) #total sum of squares
r2 = 1 - (ss_res/ss_tot)

print("MSE: ", mse)
print("RMSE", rmse)
print("MAE: ", mae)
print("R2: ", r2)

#RIGHT NOW: the accuracy will be too low because the sales aren't determined by quantity alone.
#now this model is too simple as it is just a linear regression model
#to increase its accuracy the we will need to transform this into a multivariable regression model
