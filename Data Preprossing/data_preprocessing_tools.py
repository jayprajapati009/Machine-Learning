# Data Preprocessing Tools

# Importing Library
import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import transformers

# Importing the Data set

r"""
This Line Gives Error
dataSet = pd.read_csv("C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\My Prac\Data Preprossing\Data.csv")

Follwing two lines won't give error

(1) here 'r' converts the string into raw string
dataSet = pd.read_csv(r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\My Prac\Data Preprossing\Data.csv")
(2) here the back slash (/) worked to solve the error 
dataSet = pd.read_csv("C:/Users/jaypr/Desktop/VSCodes/Machine Learning/Udemy Course/My Prac/Data Preprossing/Data.csv") 
"""

dataSet = pd.read_csv(
    r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\Machine-Learning\Data Preprossing\Data.csv")

x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

# Taking Care of Missing Data

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding Categorical Data

_ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(_ct.fit_transform(x))

print(x)
print(y)
