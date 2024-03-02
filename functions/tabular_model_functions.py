#Imports and config
import pandas as pd
import numpy as np
from config import *
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix

#functions



def randomsplit(X, Y, test_frac = 0.2, seed = 7):
    X_train, X_test, y_train, y_test = train_test_split(X = X, y = y, test_size=test_frac, random_state=seed)
    return X_train, X_test, y_train, y_test
    

#Split into train and test groups. For convenience, the train df is called df, test df is called test_df
def timesplit(df, test_frac = 0.2):
    length = df.shape[0]
    cutoff = int(length * (1 - test_frac))
    return (df.iloc[0:cutoff], df.iloc[cutoff:length])


#Onehot encoding is slightly different. We have to make a one-hot array, then append it to the dataframe, then drop the original value. This is easier with pd.get_dummies
def one_hot_encode(df, column_list):
    for column in column_list:
        tempdf = pd.get_dummies(df[column], prefix=column)
        df = pd.merge(
            left=df,
            right=tempdf,
            left_index=True,
            right_index=True,
        )
        df = df.drop(columns=column)
    return df

