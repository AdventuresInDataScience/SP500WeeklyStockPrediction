"""
Need to fix the 'remove y values of 1' block.
The close and open are missing, so I need to filter the changes which are zero

"""

# %% 0. Imports and config
# update system path
import os
import sys
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    BaggingClassifier,
)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from gpboost import GPBoostClassifier
import KTBoost.KTBoost as KTBoost


from joblib import dump, load

wd = os.path.dirname(__file__)
os.chdir(wd)
if wd in sys.path:
    sys.path.insert(0, wd)

# imports. Variables have been imported R style rather than with the config parser(less verbose)
from config import *
from functions.tabular_model_functions import *

# %% - Load Data
df = pd.read_parquet(final_data_noTA_path)

# %% - One Hot Encode
df = one_hot_encode(df, OHE_list)

# %% - Make target variable
df["y"] = df["y"] >= 0.04

# %% - Split Data
# time split X. Keep 2 version, one with the data and ticker(for later), and one without (for model fit)
# original X data
df_train, df_test = timesplit(df, test_frac=0.1)

# Random Split
# X_train, X_test, y_train, y_test = randomsplit(X, Y, test_frac = 0.2, seed = 7)

# %% - remove y values of 1 from the Train data only
df_train = df_train[df_train["change"] != 1]
df_train = df_train.dropna()

# %% - Make X and Y data
X_train = df_train.drop(["Date", "Ticker", "y"], axis=1)
X_test = df_test.drop(["Date", "Ticker", "y"], axis=1)

y_train = df_train["y"]
y_test = df_test["y"]


# %% - Scale Data
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# dump(ss, scaler_model_path) # Save scaler model
# load(scaler_model_path) # Load a previously saved scaler model

# %% - PCA as an Alternative to Factor Analysis, which was not helpful
# The below was used to work out the ideal n_components number.
# pca = PCA()
# pca.fit(X_train)
# evr = pd.Series(pca.explained_variance_ratio_)
# evrsum = evr.cumsum()

# 2. Rebuild using the right number of vars only
# 95% = 221
# 99% = 308
pca = PCA(n_components=308)
pca.fit(X_train)  # fit model

# apply transforms
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
# dump(pca, pca_model_path) #save model
##load(pca_model_path) # Load a previously saved pca model

gc.collect()
# %% - Scratchpad

# 1. Optimise weighting penalty to minimise False Discovery Rate
# ie FDR = FP/(TP+FP)
weight_list = []
tp_list = []
fp_list = []
probs_list = []


for n in [4, 5, 6, 7, 8, 9, 10]:
    t0 = time.time()
    model = LGBMClassifier(class_weight={0: 1, 1: n})
    model.fit(X_train, y_train)
    t1 = time.time()
    print("Model", n, "took", (t1 - t0) / 60, "minutes to fit")
    for prob in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        y_pred = (model.predict_proba(X_test)[:, 1] >= prob).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tp_list.append(tp)
        fp_list.append(fp)
        weight_list.append(n)
        probs_list.append(prob)

results = pd.DataFrame(
    {"weight": weight_list, "tp": tp_list, "fp": fp_list, "prob": probs_list}
)
results["FDR"] = results["fp"] / (results["tp"] + results["fp"])
results.to_csv("results.csv", index=False)
# 2. Optimise Probability threshold


t0 = time.time()
xgb = model_param_list[3][0]
xgb.fit(X_train, y_train)
t1 = time.time()
print("XGBoost, took", (t1 - t0) / 60, "minutes to fit")
# XGBoost takes 1 minute
y_pred = xgb.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("xgboost rmse:", rmse)
data = pd.DataFrame(data=[y_test.values, y_pred]).T
data.columns = ["test", "pred"]

# Light GBM Model
t0 = time.time()
lgb = model_param_list[4][0]
lgb.fit(X_train, y_train)
t1 = time.time()
print("LightGBM, took", (t1 - t0) / 60, "minutes to fit")
# Light GBM takes 40 seconds, slightly better result
y_pred = lgb.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("lgbm rmse:", rmse)
data = pd.DataFrame(data=[y_test.values, y_pred]).T
data.columns = ["test", "pred"]
dump(lgb, "C:/Users/malha/Documents/model.joblib")
# random forest is really  slow in comparison

# Compare predictions, Year by Year
# 1. get results into a dataframe
results = df_test[["Date", "Ticker"]].reset_index(drop=True)
results["y"] = data["test"]
results["pred"] = data["pred"]
# 2.groupby date, taking the mean of the top 10 results
results = results.sort_values(by=["Date", "pred"], ascending=[True, False])
summary = (
    results[["Date", "y", "pred"]].groupby("Date").apply(lambda x: x.head(10).mean())
)
summary["return"] = summary["y"] + 1 - 0.0005  # covers daily funded bet and spreads
summary["cum_return"] = summary["return"].cumprod()

yearly = summary.copy()
yearly["YM"] = pd.to_datetime(
    yearly["Date"].dt.year.astype(str) + yearly["Date"].dt.month.astype(str),
    format="%Y%m",
)
yearly = yearly.reset_index(drop=True)
yearly = (
    yearly.groupby("YM")["return"]
    .apply(lambda x: x.cumprod())
    .reset_index()
    .drop("level_1", axis=1)
)
yearly = yearly.groupby("YM").apply(lambda x: x.tail(1)).reset_index(drop=True)
plt.plot(yearly["YM"], yearly["return"].cumprod())


feature_imp = pd.DataFrame(
    {
        "Value": lgb.feature_importances_,
        "Feature": df_test.drop(["Date", "Ticker", "y"], axis=1).columns,
    }
)

# %% - Optimise Models and return stats
final_models = optimise_tabuler_model(
    X_train, y_train, X_test, y_test, model_param_list
)
optimisations = pd.DataFrame(final_models, columns=["model", "best_params", "rmse"])
