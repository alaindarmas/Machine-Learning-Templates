import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Relative path to the config file
config_file_relative_path = 'Documentation/ConfigFile.xlsx'

# Get the absolute path to the config file
config_file_path = os.path.abspath(config_file_relative_path)

# Read the config file
config = pd.read_excel(config_file_path)
config_dict = pd.Series(config.Value.values, index=config.Key).to_dict()

# Convert paths to relative
base_path = os.path.dirname(config_file_path)
dataset_path = os.path.abspath(os.path.join(base_path, config_dict['dataset_path']))
output_path = os.path.abspath(os.path.join(base_path, config_dict['output_path']))

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)