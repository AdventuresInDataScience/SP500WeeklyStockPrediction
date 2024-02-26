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
import pandas_ta as ta
import dfply
import configparser
import sys
from config import *
from functions.data_functions import *



#%% 1. Download all SP500 data and save

#Get object to connect to Fred API
fred = Fred(api_key=fred_key)

#make list of constituents
ticker_list() = make_ticker_list()


#get weekly stock data
stocks = get_yahoo_data(interval = "1wk")

#SAVE/LOAD CHECKPOINT
#stocks.to_csv(stocks_path, index = False)
stocks = pd.read_csv(stocks_path)

#%% Basic Data Cleaning
stocks = clean_stocks(stocks)

#SAVE/LOAD CHECKPOINT
#stocks.to_csv(stocks_path, index = False)
stocks = pd.read_csv(stocks_path)

#%% 2a. Download Macro Data and engineer features
#First we make a master date list,which is all the dates in the stocks df
dates_list = stocks['Date'].unique()

macro_df = get_macro_df(dates_list, stocks)

#SAVE/LOAD CHECKPOINT
#macro_df.to_csv(macro_path, index = False)
macro_df = pd.read_csv(macro_path)

#%% Indexes linked to stocks
#inflation = pd.read_csv("C:\Users\malha\Documents\Projects\All SP500 stocks\us_inflation.csv")

# First we download the various etfs and index histories
etf_list = ['^GSPC', 'XLE', 'XLI', 'XLB', 'XLY', 'XLP', 'XLV', 'XLF','XLU']
etf_df = yf.download(etf_list, period="max", interval = "1wk", threads = 'True')
# reshape data and add a column for the change that week
etf_df = etf_df.stack().reset_index()
etf_df['change'] = etf_df['Close']/etf_df['Open']
#filter to only dates in our stocks data
dates_list = stocks['Date'].drop_duplicates()
dates_list = pd.to_datetime(dates_list) #master date list of all the dates in the stocks df, as before
etf_df = etf_df.merge(dates_list.rename('Date'), how = 'left', on = 'Date')

#above works, but has outliers such as infs, which need replacing with 0s,
#and nans which also need replacing with 0s
etf_df = etf_df.fillna(0)
etf_df = etf_df.replace([np.inf, -np.inf], 0)

#SAVE/LOAD CHECKPOINT
#etf_df.to_csv(etf_path, index = False)
stocks = pd.read_csv(etf_path)
#%% Add Indicators to Stock data
#1  - Feature Engineering - Lags
for n in range(1,40):
    stocks[f'Close.lag{n}'] = stocks.groupby('Ticker')['Close'].shift(n)
stocks = stocks.copy()
#2- Feature Engineer 2 - Changes(normalised)
for n in range(1,40):
    stocks[f'Close.change{n}'] = stocks['Close']/stocks[f'Close.lag{n}']
stocks = stocks.copy()
#3 - Feature Engineer 3 - Range (normalised)
for n in range(1,40):
    a =  stocks.groupby('Ticker')['High'].rolling(n).max().reset_index()['High']
    b = stocks.groupby('Ticker')['Low'].rolling(n).min().reset_index()['Low']
    stocks[f'Close.range{n}'] = (a-b)/stocks['Close']
del a, b
stocks = stocks.copy()
#4 - Feature Engineer 4 - Distance from Low(normalised)
for n in range(1,40):
    stocks[f'Low.tolow{n}']= stocks.groupby('Ticker').apply(lambda x: x['Low']/x['Low'].shift(n), include_groups=False).reset_index()['Low']
stocks = stocks.copy()
#5 - Feature Engineer 5 - Distance from High (normalised)
for n in range(1,40):
    stocks[f'High.toHigh{n}']= stocks.groupby('Ticker').apply(lambda x: x['High']/x['High'].shift(n), include_groups=False).reset_index()['High']
stocks = stocks.copy()
# #6 - Feature Engineer 6 - Distance from SMAs Bollinger Bands(normalised)
'''
This is VERY slow. Needs some re-thinking
'''
# for ma in range(1,40):
#     for s in [-3,-2.5,-2,-1.5,-1,-0.5, 0, 0.5,1,1.5,2,2.5,3]:
#         df[f'Close.sma{n}.sd{s}'] = df['Close']/((df['Close'].rolling(ma).mean()) + s * (df['Close'].rolling(ma).std()))
#         a = stocks.groupby('Ticker').apply(lambda df: df['Close']/((df['Close'].rolling(ma).mean()) + s * (df['Close'].rolling(ma).std())).reset_index())
#         a = stocks.groupby('Ticker').apply(lambda df: df['Close']/((df['Close'].rolling(40).mean()) + 2 * (df['Close'].rolling(ma).std())).reset_index())
# stocks = stocks.copy()

#7 - Feature Engineer 7 - Distance from Highest High
for n in range(1,40):
    a = stocks.groupby('Ticker')['High'].rolling(n).max().reset_index()['High']
    stocks[f'Close.hh{n}'] = stocks['Close']/a
stocks = stocks.copy()
del a
#8 - Feature Engineer 8 - Distance from Lowest Low
for n in range(1,40):
    a = stocks.groupby('Ticker')['Low'].rolling(n).max().reset_index()['Low']
    stocks[f'Close.ll{n}'] = stocks['Close']/a
stocks = stocks.copy()
del a




'''
Resume from here

'''
#9 - Feature Engineer 9 - OHLC attributes over n lags
for n in range(1,201):
    df[f'Close.tolow{n}'] = df['Close']/df['Low'].rolling(n).min()
    df[f'Close.toHigh{n}'] = df['Close']/df['High'].rolling(n).max()
    df[f'Close.toOpen{n}'] = df['Close']/df['Open'].shift(n)
df = df.copy()
#10 - Feature Engineer 10 - Standard Deviation. This is the one causing probs. std doesnt work with NAs
for n in range(2,201):
    df[f'Close.sd{n}'] = df['Close'].rolling(n).std()
    #df[f'Close.sd{n}'] = df['Close'].rolling(n).apply(lambda x: x.std(skipna=True))/df['Close']
df = df.copy()
#11 - Feature Engineer 11 - Gaps
for n in range(1, 201):
    df[f'Gap.lag{n}'] = df['Close'].shift(n+1)-df['Open'].shift(n)/df['Close']
df = df.copy()

del ma, n, s

df['DayofWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month



#stocks['sma'] = df.groupby('Ticker')['Close'].apply(lambda x: ta.sma(x))
## A) Technical Indicators
#1. Lags of OHLC - Levels

#2. Lags of OHLC - Normalised (percentage change)

#3. Mins

#4. Maxs

#5. SDs

#6. Autocorrelation over x days

#7. Peaks/troughs

#8. Change over x days

## B) Technical Indicators
#for this, we need to make the datetime the index
#1. MAs
n = 5

stocks[f'sme{n}'] = stocks.groupby('Ticker')['Close'].apply(lambda x: ta.sma(x, n, append = True, offset = None))

#2. Bollinger Bands

#3. Stochastic

#4. MACD

#5. RSI

#6. Ichimoku cloud

#7. Average Directional Index (ADX)

#8. Parabolic SAR (PSAR)

#9. ATR

#10. Donchian Channel

#%% Save stocks to parquet file
stocks.to_parquet(f"{upper_path}stocks.parquet.gzip",
              compression='gzip')

#%% Join all data together


#%% - Remove unwanted columns and split
#Copy to keep a backup of the original, then add a column for 'return'
data = df.copy()
data['return'] = data['Close']

#reset index
df = df.dropna().reset_index(drop = True)

#Split into train and test groups. For convenience, the train df is called df, test df is called test_df
def timesplit(df, test_frac = 0.2):
    length = df.shape[0]
    cutoff = int(length * (1 - test_frac))
    return (df.iloc[0:cutoff], df.iloc[cutoff:length])

df, df_test = timesplit(df)