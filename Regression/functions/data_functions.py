from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import pandas_ta as ta
from config import *


def make_ticker_list():
    constituents = pd.read_csv(constituents_path)
    constituents = "".join(constituents["tickers"])
    constituents = constituents.split(",")
    constituents = set(constituents)

    # turn list into a string for yfinance to download
    ticker_list = ""
    for x in constituents:
        ticker_list = ticker_list + x + " "
    del x
    return ticker_list, constituents


def get_yahoo_data(ticker_list, constituents, interval="1wk"):
    df2 = yf.download(ticker_list, period="max", interval=interval, threads="True")
    df = df2.stack()
    df = df.reset_index()

    # get ticker sector info
    sec_list = []
    ind_list = []
    tick_list = []
    for x in constituents:
        try:
            data = yf.Ticker(x)
            sector = data.info["sector"]
            industry = data.info["industry"]
            sec_list.append(sector)
            ind_list.append(industry)
            tick_list.append(x)
        except:
            continue
    # merge data together
    df2 = pd.DataFrame({"Ticker": tick_list, "sector": sec_list, "industry": ind_list})
    df3 = df.merge(df2, how="left", on="Ticker")
    return df3


def clean_stocks(stocks, remove_1s):
    # remove those stocks where the open is 0, this is clearly wrong
    stocks = stocks[stocks["Open"] != 0]
    # trim outliers below the 0.4% percentile, and above 99.6%
    stocks = stocks[
        stocks["Close"] / stocks["Open"]
        <= np.percentile(stocks["Close"] / stocks["Open"], 99.6)
    ]
    stocks = stocks[
        stocks["Close"] / stocks["Open"]
        >= np.percentile(stocks["Close"] / stocks["Open"], 0.4)
    ]
    # There's a wierd number of values where open and close are teh same ie change is 0.
    if remove_1s == True:
        # We also remove this, at its probably an error
        stocks = stocks[stocks["Close"] / stocks["Open"] != 1]
    return stocks


def get_macro_df(fred, dates_list, stocks, fred_list):
    # skeleton df with all the stock dates. macros are joined to this one by one
    macro_df = pd.DataFrame({"Date": dates_list})
    macro_df["Date"] = pd.to_datetime(macro_df["Date"])

    # get data for all columns, engineer features, synch up dates and join together
    for macro in fred_list:
        # get data from api
        data = fred.get_series(macro)
        # Test stationarity
        stationary = adfuller(data.dropna().values)[1] < 0.05
        # convert to pandas df
        data = pd.DataFrame({"Date": data.index, macro: data.values})

        # Engineer those vars that are common for stationary and non-stationary series
        # frac difference
        data[f"{macro}_fd"] = data[macro] / data[macro].shift(1)

        # lags of the diffs
        data[f"{macro}_fd_1"] = data[f"{macro}_fd"].shift(1)
        data[f"{macro}_fd_2"] = data[f"{macro}_fd"].shift(2)
        data[f"{macro}_fd_3"] = data[f"{macro}_fd"].shift(3)
        data[f"{macro}_fd_4"] = data[f"{macro}_fd"].shift(4)

        # lags of the levels
        data[f"{macro}_1"] = data[f"{macro}"].shift(1)
        data[f"{macro}_2"] = data[f"{macro}"].shift(2)
        data[f"{macro}_3"] = data[f"{macro}"].shift(3)
        data[f"{macro}_4"] = data[f"{macro}"].shift(4)

        # change over x rows
        data[f"{macro}_ch_1"] = data[f"{macro}_1"] / data[f"{macro}_2"]
        data[f"{macro}_ch_2"] = data[f"{macro}_1"] / data[f"{macro}_3"]
        data[f"{macro}_ch_3"] = data[f"{macro}_1"] / data[f"{macro}_4"]
        data[f"{macro}_ch_4"] = data[f"{macro}_1"] / data[f"{macro}"].shift(5)

        # Different engineering for if data is stationary or not
        if stationary:
            # Delete the current level and frac diff, as there is approx a 1 period delay in getting updated data on FRED
            # We want to eliminate any possible look forward bias
            data = data.drop([macro, f"{macro}_fd"], axis=1)

        elif stationary == False:
            # Delete ALL the levels as they are not stationary. Also delete the current frac diff, as there is approx a 1 period delay in getting updated data on FRED
            data = data.drop(
                [
                    macro,
                    f"{macro}_fd",
                    f"{macro}_1",
                    f"{macro}_2",
                    f"{macro}_3",
                    f"{macro}_4",
                ],
                axis=1,
            )

        # merge data into the skeleton, to build a master macro table
        macro_df = macro_df.merge(data, how="outer", on="Date")
        # forward fill NAs, and impute the rest with 0s
    macro_df = macro_df.ffill()
    macro_df = macro_df.fillna(0)
    return macro_df

    """
    #%% Comments on below, for context
    #fred.search('cpiau').T # this one's not working atm
    #fred.search('vix').T # we'll handle volatility via the GSPC Standard deviation
    #fred.search('silicon').T # these next ones through yahoo or dukoscopy
    #fred.search('sugar').T Most commodities are 2017
    #fred.search('gold').T #from 2003 in dukoscopy
    #fred.search('silver').T #from 2003 in dukoscopy
    #fred.search('copper').T #from 2012
    #fred.search('platinum').T #2021 :-(
    #And do we have oil a bit further back? FRED only starts in 1986. Nope its 2010 for dukoscopy
    #So maybe gold, silver,and the main sector spydrs?
    # Energy: XLE 1998
    # Materials: XLB 1998
    # Industrials: XLI 1998
    # Consumer Discretionary: XLY 1998
    # Consumer Staples: XLP 1998
    # Healthcare: XLV 1998
    # Financials: XLF 1998
    # Information Technology: SMH 2011
    # Communication Services: XTL 2011
    # Utilities: XLU 1998
    # Real Estate: IYR 2000

    #So it seems XLE, XLI, XLB, XLY, XLP, XLV, XLF and XLU might be the most useful
    """


def make_etf_data(stocks, interval="1wk"):
    # First we download the various etfs and index histories
    etf_df = yf.download(etf_list, period="max", interval=interval, threads="True")
    # reshape data and add a column for the change that week
    etf_df = etf_df.stack().reset_index()
    etf_df["change"] = etf_df["Close"] / etf_df["Open"]
    # filter to only dates in our stocks data
    dates_list = stocks["Date"].drop_duplicates()
    dates_list = pd.to_datetime(
        dates_list
    )  # master date list of all the dates in the stocks df, as before
    etf_df = etf_df.merge(dates_list.rename("Date"), how="left", on="Date")
    etf_df["Ticker"] = etf_df["Ticker"].str.replace("^", "")
    # add features
    # a. changes
    etf_df["change1"] = etf_df.groupby(["Ticker"])["Close"].shift(1)
    etf_df["change1"] = etf_df["Close"] / etf_df["change1"]
    etf_df["change2"] = etf_df.groupby(["Ticker"])["Close"].shift(2)
    etf_df["change2"] = etf_df["Close"] / etf_df["change2"]
    etf_df["change3"] = etf_df.groupby(["Ticker"])["Close"].shift(3)
    etf_df["change3"] = etf_df["Close"] / etf_df["change3"]
    etf_df["change4"] = etf_df.groupby(["Ticker"])["Close"].shift(4)
    etf_df["change4"] = etf_df["Close"] / etf_df["change4"]
    # b.
    etf_df["h"] = (
        etf_df.groupby(["Ticker"])["High"].rolling(1).max().reset_index(0, drop=True)
    )
    etf_df["l"] = (
        etf_df.groupby(["Ticker"])["Low"].rolling(1).min().reset_index(0, drop=True)
    )
    etf_df["range1"] = (etf_df["h"] - etf_df["l"]) / etf_df["Close"]
    etf_df["h"] = (
        etf_df.groupby(["Ticker"])["High"].rolling(2).max().reset_index(0, drop=True)
    )
    etf_df["l"] = (
        etf_df.groupby(["Ticker"])["Low"].rolling(2).min().reset_index(0, drop=True)
    )
    etf_df["range2"] = (etf_df["h"] - etf_df["l"]) / etf_df["Close"]
    etf_df["h"] = (
        etf_df.groupby(["Ticker"])["High"].rolling(3).max().reset_index(0, drop=True)
    )
    etf_df["l"] = (
        etf_df.groupby(["Ticker"])["Low"].rolling(3).min().reset_index(0, drop=True)
    )
    etf_df["range3"] = (etf_df["h"] - etf_df["l"]) / etf_df["Close"]
    etf_df["h"] = (
        etf_df.groupby(["Ticker"])["High"].rolling(4).max().reset_index(0, drop=True)
    )
    etf_df["l"] = (
        etf_df.groupby(["Ticker"])["Low"].rolling(4).min().reset_index(0, drop=True)
    )
    etf_df["range4"] = (etf_df["h"] - etf_df["l"]) / etf_df["Close"]

    # pivot wider, so each etf is its own set of columns
    etf_df = etf_df.pivot(
        index="Date",
        columns="Ticker",
        values=[
            "change",
            "change",
            "change",
            "change",
            "range1",
            "range2",
            "range3",
            "range4",
        ],
    ).reset_index(drop=False)
    etf_df.columns = ["_".join(col).strip() for col in etf_df.columns.values]
    # above works, but has outliers such as infs, which need replacing with 0s,
    # and nans which also need replacing with 0s
    etf_df = etf_df.fillna(0)
    etf_df = etf_df.replace([np.inf, -np.inf], 0)
    return etf_df


def engineer_basic_features(stocks):
    # 1  - Feature Engineering - Lags
    for n in range(1, 40):
        stocks[f"Close.lag{n}"] = stocks.groupby("Ticker")["Close"].shift(n)
    stocks = stocks.copy()
    # 2- Feature Engineer 2 - Changes(normalised)
    for n in range(1, 40):
        stocks[f"Close.change{n}"] = stocks["Close"] / stocks[f"Close.lag{n}"]
    stocks = stocks.copy()
    # 3 - Feature Engineer 3 - Range (normalised)
    for n in range(1, 40):
        a = stocks.groupby("Ticker")["High"].rolling(n).max().reset_index(0, drop=True)
        b = stocks.groupby("Ticker")["Low"].rolling(n).min().reset_index(0, drop=True)
        stocks[f"Close.range{n}"] = (a - b) / stocks["Close"]
    del a, b
    stocks = stocks.copy()
    # 4 - Feature Engineer 4 - Distance from Low(normalised)
    for n in range(1, 40):
        stocks[f"Low.tolow{n}"] = (
            stocks.groupby("Ticker")
            .apply(lambda x: x["Low"] / x["Low"].shift(n), include_groups=False)
            .reset_index(0, drop=True)
        )
    stocks = stocks.copy()
    # 5 - Feature Engineer 5 - Distance from High (normalised)
    for n in range(1, 40):
        stocks[f"High.toHigh{n}"] = (
            stocks.groupby("Ticker")
            .apply(lambda x: x["High"] / x["High"].shift(n), include_groups=False)
            .reset_index(0, drop=True)
        )
    stocks = stocks.copy()

    # 6 - Feature Engineer 7 - Distance from Highest High
    for n in range(1, 40):
        a = stocks.groupby("Ticker")["High"].rolling(n).max().reset_index(0, drop=True)
        stocks[f"Close.hh{n}"] = stocks["Close"] / a
    stocks = stocks.copy()
    del a
    # 7 - Feature Engineer 8 - Distance from Lowest Low
    for n in range(1, 40):
        a = stocks.groupby("Ticker")["Low"].rolling(n).min().reset_index(0, drop=True)
        stocks[f"Close.ll{n}"] = stocks["Close"] / a
    stocks = stocks.copy()
    del a

    # 8 - Feature Engineer 10 - Standard Deviation. This is the one causing probs. std doesnt work with NAs
    for n in range(2, 40):
        stocks[f"Close.sd{n}"] = (
            stocks.groupby("Ticker")["Close"].rolling(n).std().reset_index(0, drop=True)
        )
    stocks = stocks.copy()

    # 9 - Date information
    stocks["Date"] = pd.to_datetime(stocks["Date"])
    stocks["DayofWeek"] = stocks["Date"].dt.dayofweek
    stocks["Month"] = stocks["Date"].dt.month

    # 10 - Feature Engineer 10 - normalised value for previous Gap,
    a = stocks.groupby("Ticker")["Close"].shift(1).reset_index(0, drop=True)
    stocks["Last.Gap"] = (stocks["Open"] - a) / stocks["Open"]
    del a
    return stocks


def add_target(stocks):
    nxtopn = stocks.groupby("Ticker")["Open"].shift(
        -1
    )  # .reset_index().reset_index(0,drop=True)
    nxtcls = stocks.groupby("Ticker")["Close"].shift(
        -1
    )  # .reset_index().reset_index(0,drop=True)
    stocks["y"] = (nxtcls - nxtopn) / nxtcls
    del nxtcls
    del nxtopn
    return stocks


def join_files(stocks, etf_df, macro_df):
    # convert date columns to the dame dtype
    stocks["Date"] = pd.to_datetime(stocks["Date"])
    etf_df["Date"] = pd.to_datetime(etf_df["Date"])
    macro_df["Date"] = pd.to_datetime(macro_df["Date"])
    # merge
    df = stocks.merge(etf_df, on="Date", how="left")
    df = df.merge(macro_df, on="Date", how="left")
    return df
