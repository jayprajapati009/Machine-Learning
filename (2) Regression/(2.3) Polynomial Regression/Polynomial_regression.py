# Importing the Libraries

from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Reading the data set

data = pd.read_csv(
    r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\Machine-Learning\(2) Regression\(2.3) Polynomial Regression\Position_Salaries.csv")

x = data.iloc[:, 1:2].values
y = data.iloc[:, -1].values

# Building a Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Generating feature matrix for Polynomial Regression
n = 10
pol_feat = PolynomialFeatures(degree = n)
xPoly = pol_feat.fit_transform(x)

# Building the Polynomial Regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(xPoly, y)

""" # Visualizing the Linear Regression Results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
 """
# Visualizing the Polynomial Regression Results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(xPoly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()





