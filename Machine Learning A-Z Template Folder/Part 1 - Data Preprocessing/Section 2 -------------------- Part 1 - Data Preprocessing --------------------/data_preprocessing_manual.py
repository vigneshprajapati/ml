# Data Preprocessing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data, import Imputer class
from sklearn.preprocessing import Imputer
# Create an Imputer object (use command i to see more)
# so imputer inherits from class Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0, )
# Objetct imputer applies indecies of where the missing data is
imputer = imputer.fit(X[:, 1:3])
# actually applies the calculation, X is now changed (Data.csv has not):
X[:, 1:3] = imputer.transform(X[:, 1:3])
X










