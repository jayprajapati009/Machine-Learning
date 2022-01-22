# Data Preprocessing Template

# Importing the libraries
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

print(x_train)
print(x_test)
print(y_train)
print(y_test)
