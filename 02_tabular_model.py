#%% 0. Imports and config
#update system path
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


from joblib import dump, load
wd = os.path.dirname(__file__) 
os.chdir(wd)
if wd in sys.path:
    sys.path.insert(0, wd)

#imports. Variables have been imported R style rather than with the config parser(less verbose)
from config import *
from functions.tabular_model_functions import *

#%% - Load Data
df = pd.read_parquet(final_data_noTA_path)

#%% - One Hot Encode
df = one_hot_encode(df, OHE_list)

#%% - Make X and Y data
X = df.drop(['Date', 'Ticker', 'y'], axis=1)
y = df['y']

#%% - Split Data
# time split
X_train, X_test = timesplit(X)
y_train, y_test = timesplit(y)

# Random Split
# X_train, X_test, y_train, y_test = randomsplit(X, Y, test_frac = 0.2, seed = 7)

#%% - Scale Data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# dump(ss, scaler_model_path) # Save scaler model
#load(scaler_model_path) # Load a previously saved scaler model

#%% - Optimise Models and return stats

