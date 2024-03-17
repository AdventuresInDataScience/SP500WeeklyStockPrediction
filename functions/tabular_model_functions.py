#Imports and config
import pandas as pd
import numpy as np
from config import *
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
import time
from joblib import dump, load


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

def optimise_tabuler_model(X_train, y_train, X_test, y_test, model_param_list):
    output = []
    for item in model_param_list:
        try:
            if item[0] == lgbml:
                t0 = time.time()
                model = item[0] #the model
                param_dict = item[1] #the param dictionary unique to linear lightgbm
                
                # Dataset for linear trees
                train_data_linear = lgb.Dataset(X_train, label=y_train, params={'linear_tree': True})
                #fit model using params saved in item tuple
                LGBML = lgb.train(param_dict, train_data_linear)
                # get error measure
                y_pred = LGBML.predict(X_test, num_iteration=LGBML.best_iteration)
                rmse = mean_squared_error(y_test, y_pred)**0.5
                results = (item[0], item[1], rmse)
                output.append(results)
                #save, in case run time gets disrupted
                dump(output, interim_optimisations_path)
                t1 = time.time()
                print("model",model, "took", (t1-t0)/60, "minutes")

            else:
                t0 = time.time()
                model = item[0] #the model
                param_dict = item[1] #the param dictionary unique to each model
                X = X_train
                y = y_train
                
                # make a search object
                grid = HalvingGridSearchCV(model, param_dict) 
                grid.fit(X, y)
                # get error measure
                y_pred = grid.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred)**0.5
                # save best params
                results = (model, grid.best_params_, rmse) 
                output.append(results)
                #save, in case run time gets disrupted
                dump(output, interim_optimisations_path)
                t1 = time.time()
                print("model",model, "took", (t1-t0)/60, "minutes")
            
        except:
            print("error encountered with model:", model)
            continue

    return output

'''
TO THE ABOVE FUNCTION, WE NEED TO ADD A VALIDATION SCORE MEASURE AND SAVE THIS
y_pred = grid.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)**0.5

search = HyperbandSearchCV(model, param_dist, 
                           resource_param='n_estimators',
                           scoring='roc_auc')
'''

