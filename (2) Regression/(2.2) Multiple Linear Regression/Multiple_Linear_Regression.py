# Data Preprocessing Template

# Importing the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset
dataset = pd.read_csv(
    r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\Machine-Learning\(2) Regression\(2.2) Multiple Linear Regression\50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data (Independent Variable) 

_ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(_ct.fit_transform(x))

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
_regressor = LinearRegression()
_regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = _regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)),1))
