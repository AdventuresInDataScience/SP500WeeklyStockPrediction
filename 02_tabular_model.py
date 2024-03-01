#%% 0. Imports and config
#update system path
import os
import sys
wd = os.path.dirname(__file__) 
os.chdir(wd)
if wd in sys.path:
    sys.path.insert(0, wd)

#imports. Variables have been imported R style rather than with the config parser(less verbose)
from config import *
from functions.data_functions import *

#%% - Load Data
df = pd.read_parquet(final_data_noTA_path)

#%% - One Hot Encode

#%% - Make X and Y data
X = df.drop(['Date', 'ticker', 'y'], axis=1)
y = df['y']

#%% - Split Data
# time split
X_train, X_test = timesplit(X)
y_train, y_test = timesplit(y)

# Random Split
# X_train, X_test, y_train, y_test = randomsplit(X, Y, test_frac = 0.2, seed = 7)

#%% - Scale Data
    X = df.drop(['Date', 'ticker', 'y'], axis=1)
    y = df['y']

