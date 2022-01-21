# Data Preprocessing Template

# Importing the libraries
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    r"C:\Users\jaypr\Desktop\VSCodes\Machine Learning\Udemy Course\Machine-Learning\Data Preprossing\Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
