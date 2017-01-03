# Data Preprocessing template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting SLR to the training set
from sklearn.linear_model import LinearRegression
# so this just makes an object (regressor) of the class, regressor is the 'machine'
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

# Visualising the ***Training*** set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_pred_train, color='blue')
plt.title('Salary vs Experience (TRAINING set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the ***Test*** set results
plt.scatter(X_test, y_test, color='red')
# no need to change these as the lin regression line is already done, so..
plt.plot(X_train, y_pred_train, color='green')
plt.title('Salary vs Experience (TEST set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

