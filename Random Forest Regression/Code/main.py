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

dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)