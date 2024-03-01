#%% 0. Imports and config
#update system path
import os
import sys
wd = os.path.dirname(__file__) 
os.chdir(wd)
if wd in sys.path:
    sys.path.insert(0, wd)

#imports. Variables have been imported R style rather than with the config parser(less verbose)
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import ta
from config import *
from functions.data_functions import *

#%% 1. Download all SP500 data and save

#Get object to connect to Fred API
fred = Fred(api_key=fred_key)

#make list of constituents
ticker_list = make_ticker_list()

#get weekly stock data
stocks = get_yahoo_data(interval = "1wk")

#SAVE/LOAD CHECKPOINT
#stocks.to_csv(stocks_path, index = False)
stocks = pd.read_csv(stocks_path)

#%% Basic Data Cleaning
stocks = clean_stocks(stocks)

#SAVE/LOAD CHECKPOINT
#stocks.to_parquet(stocks_path_parquet, index = False, compression='gzip')
stocks = pd.read_parquet(stocks_path_parquet)

#%% 2a. Download Macro Data and engineer features
#First we make a master date list,which is all the dates in the stocks df
dates_list = stocks['Date'].unique()

macro_df = get_macro_df(dates_list, stocks)

#SAVE/LOAD CHECKPOINT
#macro_df.to_csv(macro_path, index = False)
macro_df = pd.read_csv(macro_path)

#%% Indexes linked to stocks
#inflation = pd.read_csv("C:\Users\malha\Documents\Projects\All SP500 stocks\us_inflation.csv")

etf_df = make_etf_data(interval = "1wk")

#SAVE/LOAD CHECKPOINT
#etf_df.to_csv(etf_path, index = False)
etf_df = pd.read_csv(etf_path)

#%% Add Basic features
stocks = engineer_basic_features(stocks)

# I'm leaving All the technical Indicators for now
# This can be a seperate thing if needed

#%% FINALLY THE TARGET VARIABLE ie % move for next week's stock
stocks = add_target(stocks)

#remove OHLC columns, as these are not stationary
stocks = stocks.drop(['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)


#SAVE/LOAD CHECKPOINT
#stocks.to_parquet(stocks_path_parquet, index = False, compression='gzip')
#stocks = pd.read_parquet(stocks_path_parquet)

#%% Join all data together and drop NAs
df = join_files(stocks, etf_df, macro_df)
df = df.dropna().reset_index(drop = True)

#SAVE/LOAD CHECKPOINT
#df.to_parquet(final_data_noTA_path, index = False, compression='gzip')
#df = pd.read_parquet(final_data_noTA_path)
