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
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

import tensorflow as tf
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

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

#%% - Run Model
def build_model(input_size = 757, dropout_rate = 0.2):
    
        # Create a Sequential model
    model = tf.keras.Sequential()

    # Add layers to the model
    model.add(tf.keras.layers.Dense(units=512, activation='mish', input_shape=(input_size,)))  # Input layer
    model.add(Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=256, activation='mish', kernel_constraint=MaxNorm(5)))  # Hidden layer 1
    model.add(Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=128, activation='mish', kernel_constraint=MaxNorm(5)))  # Hidden layer 2
    model.add(Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=64, activation='mish', kernel_constraint=MaxNorm(5)))  # Hidden layer 3
    model.add(Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=32, activation='mish', kernel_constraint=MaxNorm(5)))  # Hidden layer 4
    model.add(Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(units=16, activation='mish', kernel_constraint=MaxNorm(5)))  # Hidden layer 5
    model.add(tf.keras.layers.Dense(units=1))  # Output layer (1 regression output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
    
def fit_NN(X_train, X_test, y_train, y_test, model):
    #Callbacks
    es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=20,
                              verbose=0, 
                              mode='auto')
    mc = ModelCheckpoint('best_model.h5', 
                         monitor='val_accuracy', 
                         mode='max', 
                         verbose=1, 
                         save_best_only=True)
    # Train the model
    model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_test, y_test), verbose=0, callbacks=[es, mc], batch_size=32)
    # load the saved model
    best_model = load_model('best_model.h5')
    test_loss, test_acc = best_model.evaluate(X_test,  y_test, verbose=0)
    return best_model


#build model
nn = build_model(input_size = 757, dropout_rate = 0.2)
#fit model to data
nn = fit_NN(X_train, X_test, y_train, y_test, nn)

#consider tf.keras.callbacks.BackupAndRestore