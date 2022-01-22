# Data Preprocessing Tools

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Importing the Data set

r"""
This Line Gives Error
dataSet = pd.read_csv("C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\Machine-Learning\Data Preprossing\Data.csv")

Follwing two lines won't give error

(1) here 'r' converts the string into raw string
dataSet = pd.read_csv(r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\Machine-Learning\Data Preprossing\Data.csv")
(2) here the back slash (/) worked to solve the error 
dataSet = pd.read_csv("C:/Users/jaypr/Desktop/VSCodes/Machine Learning/Udemy Course/Machine-Learning/Data Preprossing/Data.csv") 
"""

dataSet = pd.read_csv(
    r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\Machine-Learning\(1) Data Preprossing\Data.csv")

x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

# Taking Care of Missing Data

_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
_imputer.fit(x[:, 1:3])
x[:, 1:3] = _imputer.transform(x[:, 1:3])

# Encoding Categorical Data (Independent Variable)

_ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(_ct.fit_transform(x))

# Encoding Categorical Data (Dependent Variable)

_le = LabelEncoder()
y = _le.fit_transform(y)

print(x)
print(y)

# Splitting the dataset into Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

# Feature Scaling

_sc = StandardScaler()
x_train[:, 3:] = _sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = _sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
print(y_train)
print(y_test)
