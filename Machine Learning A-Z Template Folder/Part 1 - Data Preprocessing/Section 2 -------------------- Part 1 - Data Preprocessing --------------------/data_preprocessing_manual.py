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

# Encoding categorical data, LabelEncoder Class, including a mask
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Encode the Country column
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Nice :)
X[:, 0]
X
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Even better :)
X
# For y, the dependent we need only a labelencoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# Great, y sorted too
y

# Splitting dataset into Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Feature Scaling not necessary here as it is 0,1











