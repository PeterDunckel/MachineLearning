# Data Preprocessing

# Importing the Libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing the Dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Taking care of missing data
#Purpose: provide clearer data set for machine to parse
from sklearn.preprocessing import Imputer
#Use the mean of other rows data to fill in empty data
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:, 1:3]) 

#Encoding categorical data
#Purpose: break data down so it can be understood by the machine
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#break down first column to numerical values representing categories
x[:,0] = labelencoder_X.fit_transform(x[:,0])
#split first column into columns representing each piece of categorical data
#in the first column with 1 representing that this category is selected
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
#break down labels into numerical values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into the Training set and Test set
#Purpose: train the machine learning algorithm and 
#then test the preprocessing code on other data set to confirm
#that it can dynamically preprocess data passed its way
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)