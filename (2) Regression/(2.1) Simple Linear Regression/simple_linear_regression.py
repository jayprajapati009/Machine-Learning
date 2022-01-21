# Data Preprocessing Template

# Importing the libraries
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv(
    r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\Machine-Learning\(2) Regression\(2.1) Simple Linear Regression\Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Training the Simple Linear Regression on the Training Set
_regressor = LinearRegression()
_regressor.fit(x_train, y_train)

# Predicting the Test Set
y_pred = _regressor.predict(x_test)

# Predicting salary for custom input of years of experience
print("Predicting salary for custom input of years of experience = 12 Years")
print('salary =', float(_regressor.predict([[12]])))

# Getting the equation for the Regression model
print("Getting the equation for the Linear Regression model")
print('Salary', '=', _regressor.intercept_, '+',
      '(', float(_regressor.coef_), '*', 'Years of Experience', ')')

# Visualizing the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, _regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualizing the test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, _regressor.predict(x_test), color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
