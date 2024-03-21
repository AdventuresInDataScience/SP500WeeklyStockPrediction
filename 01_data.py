'''
I need to:
    1. Macro and etf are saved. Stocks is saved from panel 1 only ie before any cleaning etc 
    2. Still need to check all the feature engineering is correct
'''


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
ticker_list, constituents = make_ticker_list()

#get weekly stock data
stocks = get_yahoo_data(ticker_list, constituents, interval = "1wk")

#SAVE/LOAD CHECKPOINT
#stocks.to_csv(stocks_path, index = False)
#stocks = pd.read_csv(stocks_path)

#%% Basic Data Cleaning
stocks = clean_stocks(stocks, remove_1s = False)

#SAVE/LOAD CHECKPOINT
#stocks.to_parquet(stocks_path_parquet, index = False, compression='gzip')
#stocks = pd.read_parquet(stocks_path_parquet)

#%% 2a. Download Macro Data and engineer features
# The function for this doesn't work. So for now, I'm just pasting in the loop
dates_list = stocks['Date'].unique()
#skeleton df with all the stock dates. macros are joined to this one by one
macro_df = pd.DataFrame({'Date':dates_list})
macro_df['Date'] = pd.to_datetime(macro_df['Date'])

#get data for all columns, engineer features, synch up dates and join together
for macro in fred_list:
    #get data from api
    data = fred.get_series(macro)
    #Test stationarity
    stationary = adfuller(data.dropna().values)[1] < 0.05
        #convert to pandas df
    data = pd.DataFrame({'Date':data.index, macro:data.values})
    
    #Engineer those vars that are common for stationary and non-stationary series
    # frac difference
    data[f'{macro}_fd'] = data[macro]/data[macro].shift(1)
    
    # lags of the diffs
    data[f'{macro}_fd_1'] = data[f'{macro}_fd'].shift(1)
    data[f'{macro}_fd_2'] = data[f'{macro}_fd'].shift(2)
    data[f'{macro}_fd_3'] = data[f'{macro}_fd'].shift(3)
    data[f'{macro}_fd_4'] = data[f'{macro}_fd'].shift(4)
    
    # lags of the levels
    data[f'{macro}_1'] = data[f'{macro}'].shift(1)
    data[f'{macro}_2'] = data[f'{macro}'].shift(2)
    data[f'{macro}_3'] = data[f'{macro}'].shift(3)
    data[f'{macro}_4'] = data[f'{macro}'].shift(4)
    
    # change over x rows
    data[f'{macro}_ch_1'] = data[f'{macro}_1']/data[f'{macro}_2']
    data[f'{macro}_ch_2'] = data[f'{macro}_1']/data[f'{macro}_3']
    data[f'{macro}_ch_3'] = data[f'{macro}_1']/data[f'{macro}_4']
    data[f'{macro}_ch_4'] = data[f'{macro}_1']/data[f'{macro}'].shift(5)
    
    #Different engineering for if data is stationary or not
    if stationary:
        # Delete the current level and frac diff, as there is approx a 1 period delay in getting updated data on FRED
        # We want to eliminate any possible look forward bias
        data = data.drop([macro, f'{macro}_fd'], axis = 1)
        
    elif stationary == False:
        # Delete ALL the levels as they are not stationary. Also delete the current frac diff, as there is approx a 1 period delay in getting updated data on FRED
        data = data.drop([macro, f'{macro}_fd', f'{macro}_1', f'{macro}_2', f'{macro}_3', f'{macro}_4'], axis = 1)
        
    #merge data into the skeleton, to build a master macro table
    macro_df = macro_df.merge(data, how='outer', on='Date')
    #forward fill NAs, and impute the rest with 0s
macro_df = macro_df.ffill()
macro_df = macro_df.fillna(0)











#1  - Feature Engineering - Lags






# #First we make a master date list,which is all the dates in the stocks df
# dates_list = stocks['Date'].drop_duplicates()

# macro_df = get_macro_df(fred, dates_list, stocks, fred_list)

# #SAVE/LOAD CHECKPOINT
# #macro_df.to_csv(macro_path, index = False)
# #macro_df = pd.read_csv(macro_path)

#%% 2b Indexes linked to stocks
#inflation = pd.read_csv("C:\Users\malha\Documents\Projects\All SP500 stocks\us_inflation.csv")

etf_df = make_etf_data(stocks, interval = "1wk")
etf_df = etf_df.rename(columns={"Date_": "Date"})

#SAVE/LOAD CHECKPOINT
#etf_df.to_csv(etf_path, index = False)
#etf_df = pd.read_csv(etf_path)

#%% Add Basic features
stocks = engineer_basic_features(stocks)
stocks['change'] = stocks['Close']/stocks['Open']
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
etf_df = etf_df.rename(columns={"Date_": "Date"})
df = join_files(stocks, etf_df, macro_df)
df = df.dropna().reset_index(drop = True)

#SAVE/LOAD CHECKPOINT
df.to_parquet(final_data_noTA_path, index = False, compression='gzip')
#df = pd.read_parquet(final_data_noTA_path)
