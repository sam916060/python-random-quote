# %%
pip install numpy matplotlib pandas scikit-learn


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset

file_path = "/Users/sampathkumar/Desktop/Visual studio/LC-11.11.csv"




# %%

dataset = pd.read_csv(file_path)


# %%

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset


# %%

## Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 16)



# %%
## Training the Random Forest Regression model on the whole dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


# %%
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
max_features.append(None)
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)



# %%

regressor = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)

rf_random.best_params_

best_grid = rf_random.best_estimator_



# %%
y_pred = rf_random.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

y_pred1 = rf_random.predict([[10,6,1000,6,1000,15,20,3000,6900000,150,741,750,30]])
y_pred1


