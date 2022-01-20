# Data Preprocessing Tools

# Importing Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Data set

# This Line Gives Error
# dataSet = pd.read_csv("C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\My Prac\Data Preprossing\Data.csv")

# Follwing two lines won't give error

# (1) here 'r' converts the string into raw string
# dataSet = pd.read_csv(r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\My Prac\Data Preprossing\Data.csv")
# (2) here the back slash (/) workd to solve the error 
# dataSet = pd.read_csv("C:/Users/jaypr/Desktop/VSCodes/Machine Learning/Udemy Course/My Prac/Data Preprossing/Data.csv")


dataSet = pd.read_csv("C:/Users/jaypr/Desktop/VSCodes/Machine Learning/Udemy Course/My Prac/Data Preprossing/Data.csv")

x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

print(x)
print(y)

