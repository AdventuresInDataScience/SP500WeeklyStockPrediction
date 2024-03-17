#%% 0. Imports and config
#update system path
import os
import sys
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA

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
X_train, X_test = timesplit(X, test_frac = 0.2)
y_train, y_test = timesplit(y, test_frac = 0.2)

# Random Split
# X_train, X_test, y_train, y_test = randomsplit(X, Y, test_frac = 0.2, seed = 7)

#%% - Scale Data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# dump(ss, scaler_model_path) # Save scaler model
#load(scaler_model_path) # Load a previously saved scaler model

#%% - PCA as an Alternative to Factor Analysis, which was not helpful
#The below was used to work out the ideal n_components number.
# pca = PCA()
# pca.fit(X_train)
# evr = pd.Series(pca.explained_variance_ratio_)
# evrsum = evr.cumsum()

#2. Rebuild using the right number of vars only
#95% = 243
#99% = 321
pca = PCA(n_components=243)
pca.fit(X_train) #fit model

#apply transforms
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
#dump(pca, 'pca.joblib') #save model

gc.collect()
#%% - Scratchpad
t0 = time.time()
xgb = model_param_list[3][0]
xgb.fit(X_train, y_train)
t1 = time.time()
print("XGBoost, took", (t1 - t0)/60, "minutes to fit")
#XGBoost takes 1 minute
y_pred = xgb.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)**0.5
print('xgboost rmse:',rmse)
data = pd.DataFrame(data = [y_test.values, y_pred]).T
data.columns = ['test','pred']


t0 = time.time()
lgb = model_param_list[4][0]
lgb.fit(X_train, y_train)
t1 = time.time()
print("LightGBM, took", (t1 - t0)/60, "minutes to fit")
#Light GBM takes 40 seconds, slightly better result
y_pred = lgb.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)**0.5
print('lgbm rmse:', rmse)
data = pd.DataFrame(data = [y_test.values, y_pred]).T
data.columns = ['test','pred']

#random forest is really slow
#%% - Optimise Models and return stats
final_models = optimise_tabuler_model(X_train, y_train, X_test, y_test, model_param_list)
optimisations = pd.DataFrame(final_models, columns=['model', 'best_params', 'rmse'])
