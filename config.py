##############################################################################
# Data Config

#Keys
fred_key = '378bdfc0c483b3259c41cecde2ae2f0f'

#Paths
constituents_path = "C:/Users/malha/Documents/Data/S&P 500 Historical Components & Changes(08-01-2023).csv"
upper_path = "C:/Users/malha/Documents/Data/SP500WeeklyStockPrediction/"
stocks_path = "C:/Users/malha/Documents/Data/SP500WeeklyStockPrediction/stocks1w.csv"
stocks_path_parquet = "C:/Users/malha/Documents/Data/SP500WeeklyStockPrediction/stocks1w.parquet.gzip"
macro_path = "C:/Users/malha/Documents/Data/SP500WeeklyStockPrediction/macros1w.csv"
etf_path = "C:/Users/malha/Documents/Data/SP500WeeklyStockPrediction/etfs1w.csv"
final_data_noTA_path = "C:/Users/malha/Documents/Data/SP500WeeklyStockPrediction/final_data_noTA.parquet.gzip"

#Lists
fred_list = ['MCOILWTICO', 'WCOILBRENTEU', 'UNRATE', 'FF', 'PCE', 'FYFSGDA188S', 'GDP',
              'ADPWINDCONNERSA', 'CLDACBW027NBOG', 'MSPUS', 'WPRIME', 'RALACBW027NBOG', 'LOANS','OTHSECNSA', 
              'CEU6500000001', 'TB3MS', 'DNPVRC1A027NBEA', 'DNPVRC1A027NBEA', 'DGDSRC1', 'DNRGRC1M027SBEA', 'AHEMAN',
              'CEU3100000008','AHECONS', 'LNU03032229', 'LNU02032184', 'FEDFUNDS', 'CEU7000000008', 'CEU6000000008',
              'CEU4200000008', 'PI', 'REALLNNSA', 'MVPHGFD027MNFRBDAL']

etf_list = ['^GSPC', 'XLE', 'XLI', 'XLB', 'XLY', 'XLP', 'XLV', 'XLF','XLU']

##############################################################################
# Model Config
# Variables that need one hot encoding
OHE_list = ['sector', 'industry', 'DayofWeek', 'Month']
scaler_model_path = "C:/Users/malha/Documents/Data/SP500WeeklyStockPrediction/scaler_model.joblib"

model_dictionary = {}
##############################################################################


'''
random forest
ridge
XBnet

pip install mapie
https://github.com/scikit-learn-contrib/MAPIE

pip install --upgrade git+https://github.com/tusharsarkar3/XBNet.git
https://github.com/tusharsarkar3/XBNet/blob/master/XBNet/models.py
# Wheel build error - C++ 14

pip install --upgrade linear-tree
https://github.com/cerlymarco/linear-tree/blob/main/notebooks/README.md

pip install gpboost -U
https://gpboost.readthedocs.io/en/latest/pythonapi/gpboost.GPBoostRegressor.html#gpboost.GPBoostRegressor

pip install -U KTBoost
https://github.com/fabsig/KTBoost

#this is snapboost
pip install snapml
https://snapml.readthedocs.io/en/latest/boosting_machines.html#boosting-machine-regressor

pip install bartpy
https://github.com/JakeColtman/bartpy
#issue with deprecated sklearn vs scikit learn in requirements.txt

pip install SMAC3
pip install Cmake
pip install wheel setuptools --upgrade
# STILL NOT WORKING
https://github.com/automl/SMAC3
'''
