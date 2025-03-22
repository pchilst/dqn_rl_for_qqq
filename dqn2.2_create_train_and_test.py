'''
08/22/24

This script is to create the training and testing datasets which shall then be 
passed to the BT script. It does all the feature engineering, and fits and saves 
out the one hot encoder.
'''

import pandas as pd
import pandas_ta
import numpy
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from  sklearn.metrics  import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import category_encoders as ce
#import shap
import pickle
import talib
import json
import joblib
from stable_baselines3 import DQN, PPO, A2C

global SLIPPAGE_PCT
SLIPPAGE_PCT = .0004

base_path = r'C:\Users\Peter\Dropbox\Stocks\Back-Testing\Machine Learning\DQN v2.1'
os.chdir(base_path)







# New first-rate dataset (correct)
df = pd.read_csv('TQQQ_full_1hour_adjsplit_Aug20_no930adj.csv')
df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'QQQ_Close', 'VIX_Close', 'TNX_Close']]
df['Datetime'] = pd.to_datetime(df['Datetime'])



''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
''' Section 1: Feature Engineering '''
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

def ema(s, p):
    return pd.Series(s).ewm(span=p, adjust=False).mean()

def tema(s, p):
    ema_s = ema(s, p)
    ema_ema_s = ema(ema_s, p)
    tema_s = 3 * ema_s - 3 * ema_ema_s + ema(ema_ema_s, p)
    return tema_s

def ma(t, s, p):
    ema_1 = ema(s, p)
    rma_1 = s.rolling(window=p).mean()
    vwma_1 = (s * pd.Series(range(1, len(s) + 1))).rolling(window=p).sum() / pd.Series(range(1, len(s) + 1)).rolling(window=p).sum()
    wma_1 = s.rolling(window=p).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)))
    tema_1 = tema(s, p)
    sma_1 = s.rolling(window=p).mean()

    if t == "ema":
        return ema_1
    elif t == "rma":
        return rma_1
    elif t == "vwma":
        return vwma_1
    elif t == "wma":
        return wma_1
    elif t == "tema":
        return tema_1
    elif t == "sma":
        return sma_1

def off(s, o):
    return s.shift(-o) if o > 0 else s


df['ema10'] = ma("ema", off(df['Close'], 0), 20)
df['ema20'] = ma("ema", off(df['Close'], 0), 20)
df['ema50'] = ma("ema", off(df['Close'], 0), 50)
df['sma100'] = ma("sma", off(df['Close'], 0), 100)
df['sma200'] = ma("sma", off(df['Close'], 0), 200)




############################
# SMI
############################

pct_k = 10
pct_d = 6

def calculate_smi(df, a, b):
    ll = df['Low'].rolling(window=a).min()
    hh = df['High'].rolling(window=a).max()
    diff = hh - ll
    rdiff = df['Close'] - (hh + ll) / 2
    avgrel = pd.Series(rdiff).ewm(span=b).mean().ewm(span=b).mean()
    avgdiff = pd.Series(diff).ewm(span=b).mean().ewm(span=b).mean()
    SMI = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
    return hh, ll, rdiff, avgrel, avgdiff, SMI


df['hh'], df['ll'], df['rdiff'], df['avgrel'], df['avgdiff'], df['SMI'] = calculate_smi(df, pct_k, pct_d)






############################
# MACD
############################

def calculate_macd(df, close_column='Close', macd_params=None):

    results = df[[close_column]].copy()
    
    for fast, slow, signal in macd_params:
        # Calculate the fast and slow EMAs
        ema_fast = df[close_column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[close_column].ewm(span=slow, adjust=False).mean()
        
        # Calculate the MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate the signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate the MACD histogram
        macd_histogram = macd_line - signal_line
        
        # Add the results to the DataFrame
        results[f'MACD_{fast}_{slow}_{signal}'] = macd_line
        results[f'Signal_{fast}_{slow}_{signal}'] = signal_line
        results[f'Hist_{fast}_{slow}_{signal}'] = macd_histogram
    
    return results

# Define custom MACD parameters if needed
custom_params = [
    (12, 26, 9),
    (5, 35, 5),
    (24, 52, 18),
    (8, 21, 5),
    (17, 43, 9),
    (100, 200, 50),
    (150, 300, 75),
    (200, 400, 100)
]

# Calculate MACD
macd_results = calculate_macd(df, close_column='Close', macd_params=custom_params)
macd_results.drop('Close', axis=1, inplace=True)


df = df.merge(macd_results, how='inner', left_index=True, right_index=True)





############################
# Bollinger Bands
############################

def calculate_bollinger_bands(df, windows=[20, 50, 100], std_dev=2):

    for window in windows:
        # Calculate Moving Average
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Calculate Standard Deviation
        df[f'SD_{window}'] = df['Close'].rolling(window=window).std()
        
        # Calculate Upper and Lower Bollinger Bands
        df[f'Upper_BB_{window}'] = df[f'MA_{window}'] + (df[f'SD_{window}'] * std_dev)
        df[f'Lower_BB_{window}'] = df[f'MA_{window}'] - (df[f'SD_{window}'] * std_dev)
        
        # Create binary indicators
        df[f'Over_Upper_BB_{window}'] = np.where(df['Close'] > df[f'Upper_BB_{window}'], 1, 0)
        df[f'Under_Lower_BB_{window}'] = np.where(df['Close'] < df[f'Lower_BB_{window}'], 1, 0)
    
    return df

# Define the windows you want to use
windows = [20, 50, 100, 200]

# Apply the function to your DataFrame
df = calculate_bollinger_bands(df, windows=windows)






############################
# General
############################

df['Volume'] = df['Volume'].astype(float)
df['Log_Volume'] = np.log(df['Volume'])


# Calculate the percentage difference from ATH
df['Close_vs_ATH'] = ((df['QQQ_Close'] - df['QQQ_Close'].cummax()) / df['QQQ_Close'].cummax()) * 100

# Calculate Rolling Statistics
df['Rolling_Mean_10'] = df['Close'].rolling(window=10).mean()
df['Rolling_Std_10'] = df['Close'].rolling(window=10).std()
df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
df['Rolling_Std_20'] = df['Close'].rolling(window=20).std()
df['Rolling_Mean_50'] = df['Close'].rolling(window=50).mean()
df['Rolling_Std_50'] = df['Close'].rolling(window=50).std()


df['VIX_Rolling_Mean_10'] = df['VIX_Close'].rolling(window=10).mean()
df['VIX_Rolling_Std_10'] = df['VIX_Close'].rolling(window=10).std()
df['VIX_Rolling_Mean_20'] = df['VIX_Close'].rolling(window=20).mean()
df['VIX_Rolling_Std_20'] = df['VIX_Close'].rolling(window=20).std()
df['VIX_Rolling_Mean_50'] = df['VIX_Close'].rolling(window=50).mean()
df['VIX_Rolling_Std_50'] = df['VIX_Close'].rolling(window=50).std()





def calculate_multiple_rsi(df, close_column='Close', windows=None):

    results = df[[close_column]].copy()
    
    for window in windows:
        delta = df[close_column].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        results[f'RSI_{window}'] = rsi
    
    return results

custom_windows = [5, 14, 21, 30, 50, 100, 200]

# Calculate RSI for multiple windows
rsi_results = calculate_multiple_rsi(df, close_column='Close', windows=custom_windows)
rsi_results.drop('Close', axis=1, inplace=True)

df = df.merge(rsi_results, how='inner', left_index=True, right_index=True)






# Calculate Volume Change
df['Volume_Change'] = df['Log_Volume'].pct_change()


# VIX Rate of Change
df['VIX_ROC'] = df['VIX_Close'].pct_change(periods=1)


# VIX to price ratio
df['VIX_Price_Ratio'] = df['VIX_Close'] / df['QQQ_Close']


# VIX Percentile
window = 252  # Approximately one trading year
df['VIX_Percentile'] = df['VIX_Close'].rolling(window=window).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1])

# VIX Momentum
df['VIX_Momentum_5'] = df['VIX_Close'] - df['VIX_Close'].shift(5)
df['VIX_Momentum_10'] = df['VIX_Close'] - df['VIX_Close'].shift(10)
df['VIX_Momentum_20'] = df['VIX_Close'] - df['VIX_Close'].shift(20)
df['VIX_Momentum_50'] = df['VIX_Close'] - df['VIX_Close'].shift(50)


# VIX Historical Ratio
df['Historical_Vol'] = df['QQQ_Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
df['VIX_HV_Ratio'] = df['VIX_Close'] / df['Historical_Vol']


# VIX Bollinger Bands
def calculate_vix_bollinger_bands(df, windows=[20, 50, 100], std_dev=2):

    for window in windows:
        # Calculate Moving Average
        df[f'VIX_MA_{window}'] = df['VIX_Close'].rolling(window=window).mean()
        
        # Calculate Standard Deviation
        df[f'VIX_SD_{window}'] = df['VIX_Close'].rolling(window=window).std()
        
        # Calculate Upper and Lower Bollinger Bands
        df[f'VIX_Upper_BB_{window}'] = df[f'VIX_MA_{window}'] + (df[f'VIX_SD_{window}'] * std_dev)
        df[f'VIX_Lower_BB_{window}'] = df[f'VIX_MA_{window}'] - (df[f'VIX_SD_{window}'] * std_dev)
        
        # Create binary indicators
        df[f'VIX_Over_Upper_BB_{window}'] = np.where(df['VIX_Close'] > df[f'VIX_Upper_BB_{window}'], 1, 0)
        df[f'VIX_Under_Lower_BB_{window}'] = np.where(df['VIX_Close'] < df[f'VIX_Lower_BB_{window}'], 1, 0)
    
    return df

# Define the windows you want to use
windows = [20, 50, 100, 200]

# Apply the function to your DataFrame
df = calculate_vix_bollinger_bands(df, windows=windows)



''' ~~~~ TNX Related Features ~~~~ '''

def add_tnx_qqq_features(df, windows=[5, 10, 20, 50, 100]):
    # Calculate returns
    df['TNX_Return'] = df['TNX_Close'].pct_change()
    df['QQQ_Return'] = df['QQQ_Close'].pct_change()

    # Direction alignment (1 if same direction, -1 if opposite, 0 if one is flat)
    df['Direction_Alignment'] = np.sign(df['TNX_Return'] * df['QQQ_Return'])

    # Correlation over multiple windows
    for window in windows:
        df[f'TNX_QQQ_Corr_{window}'] = df['TNX_Return'].rolling(window).corr(df['QQQ_Return'])

    # Relative strength (ratio of cumulative returns)
    for window in windows:
        df[f'TNX_QQQ_RelStrength_{window}'] = (
            (1 + df['TNX_Return']).rolling(window).apply(np.prod) /
            (1 + df['QQQ_Return']).rolling(window).apply(np.prod)
        )

    # Divergence (difference in z-scores)
    for window in windows:
        tnx_zscore = (df['TNX_Close'] - df['TNX_Close'].rolling(window).mean()) / df['TNX_Close'].rolling(window).std()
        qqq_zscore = (df['QQQ_Close'] - df['QQQ_Close'].rolling(window).mean()) / df['QQQ_Close'].rolling(window).std()
        df[f'TNX_QQQ_Divergence_{window}'] = tnx_zscore - qqq_zscore

    # Volatility ratio
    for window in windows:
        df[f'TNX_QQQ_VolRatio_{window}'] = df['TNX_Return'].rolling(window).std() / df['QQQ_Return'].rolling(window).std()

    # Trend alignment (both trending up, both down, or mixed)
    for window in windows:
        df[f'TNX_Trend_{window}'] = np.where(df['TNX_Close'] > df['TNX_Close'].shift(window), 1, -1)
        df[f'QQQ_Trend_{window}'] = np.where(df['QQQ_Close'] > df['QQQ_Close'].shift(window), 1, -1)
        df[f'Trend_Alignment_{window}'] = df[f'TNX_Trend_{window}'] * df[f'QQQ_Trend_{window}']

    # Relative performance
    for window in windows:
        df[f'TNX_RelPerf_{window}'] = df['TNX_Close'] / df['TNX_Close'].shift(window) - 1
        df[f'QQQ_RelPerf_{window}'] = df['QQQ_Close'] / df['QQQ_Close'].shift(window) - 1
        df[f'RelPerf_Diff_{window}'] = df[f'TNX_RelPerf_{window}'] - df[f'QQQ_RelPerf_{window}']

    # Yield-Price Ratio and its moving average
    df['Yield_Price_Ratio'] = df['TNX_Close'] / df['QQQ_Close']
    for window in windows:
        df[f'Yield_Price_Ratio_MA_{window}'] = df['Yield_Price_Ratio'].rolling(window).mean()

    return df

# Usage:
df = add_tnx_qqq_features(df)



for period in [1, 5, 10, 20, 50]:
    df[f'TNX_ROC_{period}'] = df['TNX_Close'].pct_change(periods=period) * 100


ma_periods = [10, 20, 50, 100, 200]
for period in ma_periods:
    df[f'TNX_MA_{period}'] = df['TNX_Close'].rolling(window=period).mean()

for short, long in zip(ma_periods[:-1], ma_periods[1:]):
    df[f'TNX_MA_{short}_{long}_Cross'] = (df[f'TNX_MA_{short}'] > df[f'TNX_MA_{long}']).astype(int)
    
    
def calculate_rsi(series, periods):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

for period in [6, 14, 28]:
    df[f'TNX_RSI_{period}'] = calculate_rsi(df['TNX_Close'], period)
    
    
for period in [14, 28, 56]:
    df[f'TNX_Momentum_{period}'] = df['TNX_Close'] - df['TNX_Close'].shift(period)
    
    
for period in [50, 100, 200]:
    df[f'TNX_Percentile_{period}'] = df['TNX_Close'].rolling(window=period).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df[f'TNX_Distance_From_High_{period}'] = (df['TNX_Close'] - df['TNX_Close'].rolling(window=period).max()) / df['TNX_Close'].rolling(window=period).max()
    df[f'TNX_Distance_From_Low_{period}'] = (df['TNX_Close'] - df['TNX_Close'].rolling(window=period).min()) / df['TNX_Close'].rolling(window=period).min()




# Datetime features
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Day_of_Week'] = df['Datetime'].dt.strftime('%A')
df['Month_Name'] = df['Datetime'].dt.strftime('%B')
df['Hour_of_Day'] = df['Datetime'].dt.hour




############################
# MKR
############################
kernel_name = "silverman"
bandwidth = 11
consec = 2
diff_threshold = 0

def silverman(diff, bandwidth):
    return np.where(np.abs(diff / bandwidth) <= 0.5, 0.5 * np.exp(-(diff / bandwidth) / 2) * np.sin((diff / bandwidth) / 2 + np.pi / 4), 0.0)


kernels = {
    "silverman": silverman,
}

def kernel(diff, bandwidth, kernel_name):
    if kernel_name in kernels:
        return kernels[kernel_name](diff, bandwidth)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
        

# Convert 'Close' column to a numpy array for faster access
closes = df['Close'].values

# Initialize an empty numpy array for 'mean'
means = np.empty(len(closes))
means[:] = np.nan

# Iterate over the main loop with numpy-based inner loop calculations
for i in range(bandwidth, len(closes)):
    # Calculate weights using the inner loop logic
    diffs = (np.arange(bandwidth) - 1)**2 / bandwidth**2
    weights = kernel(diffs, 1, kernel_name)
    
    # Select the range of close values based on i and bandwidth
    selected_closes = closes[i - bandwidth + 1:i + 1]
    
    # Calculate the weighted sum and the sum of weights
    sum_ = np.dot(selected_closes, weights[::-1])  # Note: We reverse weights due to the nature of indexing in the original logic
    sumw = np.sum(weights)
    
    # Calculate the mean
    means[i] = sum_ / sumw if sumw != 0 else np.nan

# Update the DataFrame with results
df['mean'] = means
df['older_mean'] = np.roll(df['mean'], 1)
#df['Signal'] = np.where(df['mean'] > np.roll(df['mean'], 1), 'bullish', 'bearish')

df['Diff'] = df['mean'] - df['older_mean']

df['Order_Temp'] = ''
df.loc[df['Diff'] / df['mean'] >= diff_threshold, 'Order_Temp'] = 'long'
df.loc[df['Diff'] / df['mean'] <= -diff_threshold, 'Order_Temp'] = 'short'

# Shift the orders up by 1 to simulate live trading
df['Order_Temp'] = df['Order_Temp'].shift(1)


# Apply requirement of x number of consecutive orders    
if consec >= 2:
    # Create a mask for consecutive 'long' or 'short' values
    def consecutive_mask(ser):
        return ser.groupby((ser != ser.shift()).cumsum()).cumcount().add(1)
    
    df['Consecutive_Long'] = consecutive_mask(df['Order_Temp'] == 'long').where(df['Order_Temp'] == 'long')
    df['Consecutive_Short'] = consecutive_mask(df['Order_Temp'] == 'short').where(df['Order_Temp'] == 'short')
    
    # Create a new column based on the consecutive requirement
    df['MKR_Order'] = None
    df.loc[df['Consecutive_Long'] >= consec, 'MKR_Order'] = 'long'
    df.loc[df['Consecutive_Short'] >= consec, 'MKR_Order'] = 'short'
    
    # Drop helper columns
    df.drop(['Consecutive_Long', 'Consecutive_Short'], axis=1, inplace=True)
else:
    df['MKR_Order'] = df['Order_Temp']
    
df = df.drop(columns='Order_Temp',axis=1)
df = df.drop(columns='MKR_Order',axis=1)


############################
# Fear and Greed
############################
High_period = 5
Low_period = 50
Stdev_period = 50

# ohlc4
df['ohlc4'] = (df['High'] + df['Low'] + df['Close'] + df['Open']) / 4

# Calculate true range
df['tr'] = np.maximum.reduce([df['High'] - df['Low'],
                                      np.abs(df['High'] - df['Close'].shift(1)),
                                      np.abs(df['Low'] - df['Close'].shift(1))])

# Calculate FZ1
df['FZ1'] = (df['ohlc4'].rolling(window=High_period).max() - df['ohlc4']) / df['ohlc4'].rolling(window=High_period).max()
df['AVG1'] = df['FZ1'].rolling(Stdev_period).apply(lambda x: ((np.arange(Stdev_period)+1)*x).sum()/(np.arange(Stdev_period)+1).sum(), raw=True)

df['STDEV1'] = df['FZ1'].rolling(window=Stdev_period).std()
df['FZ1Limit'] = df['AVG1'] + df['STDEV1']

# Calculate FZ2
df['FZ2'] = df['ohlc4'].rolling(High_period).apply(lambda x: ((np.arange(High_period)+1)*x).sum()/(np.arange(High_period)+1).sum(), raw=True)
df['AVG2'] = df['FZ2'].rolling(Stdev_period).apply(lambda x: ((np.arange(Stdev_period)+1)*x).sum()/(np.arange(Stdev_period)+1).sum(), raw=True)
    
df['STDEV2'] = df['FZ2'].rolling(window=Stdev_period).std()
df['FZ2Limit'] = df['AVG2'] - df['STDEV2']

# FearZone
df['Fearzone_Con'] = (df['FZ1'] > df['FZ1Limit']) & (df['FZ2'] < df['FZ2Limit'])
df['FearZoneOpen'] = np.where(df['Fearzone_Con'], df['Low'] - df['tr'], np.nan)
df['FearZoneClose'] = np.where(df['Fearzone_Con'], df['Low'] - 2 * df['tr'], np.nan)

# Calculate GZ1
df['GZ1'] = (df['ohlc4'].rolling(window=Low_period).min() - df['ohlc4']) / df['ohlc4'].rolling(window=Low_period).min()
df['AVG1'] = df['GZ1'].rolling(Stdev_period).apply(lambda x: ((np.arange(Stdev_period)+1)*x).sum()/(np.arange(Stdev_period)+1).sum(), raw=True)

df['STDEV1'] = df['GZ1'].rolling(window=Stdev_period).std()
df['GZ1Limit'] = df['AVG1'] - df['STDEV1']

# Calculate GZ2
df['GZ2'] = df['ohlc4'].rolling(Low_period).apply(lambda x: ((np.arange(Low_period)+1)*x).sum()/(np.arange(Low_period)+1).sum(), raw=True)
df['AVG2'] = df['GZ2'].rolling(Stdev_period).apply(lambda x: ((np.arange(Stdev_period)+1)*x).sum()/(np.arange(Stdev_period)+1).sum(), raw=True) 
    
df['STDEV2'] = df['GZ2'].rolling(window=Stdev_period).std()
df['GZ2Limit'] = df['AVG2'] + df['STDEV2']

# GreedZone
df['Greedzone_Con'] = (df['GZ1'] < df['GZ1Limit']) & (df['GZ2'] > df['GZ2Limit'])
df['GreedZoneOpen'] = np.where(df['Greedzone_Con'], df['Low'] + df['tr'], np.nan)
df['GreedZoneClose'] = np.where(df['Greedzone_Con'], df['Low'] + 2 * df['tr'], np.nan)


# Fill NA with 0
df['FearZoneOpen'].fillna(0, inplace=True)
df['FearZoneClose'].fillna(0, inplace=True)
df['GreedZoneOpen'].fillna(0, inplace=True)
df['GreedZoneClose'].fillna(0, inplace=True)



############################
# HG - PTWB
############################
pivot_left = 20
pivot_right = 7

df = df.reset_index()

def checkhl(data_back, data_forward, hl):
    if hl == 'high' or hl == 'High':
        ref = data_back[len(data_back)-1]
        for i in range(len(data_back)-1):
            if ref < data_back[i]:
                return 0
        for i in range(len(data_forward)):
            if ref <= data_forward[i]:
                return 0
        return 1
    if hl == 'low' or hl == 'Low':
        ref = data_back[len(data_back)-1]
        for i in range(len(data_back)-1):
            if ref > data_back[i]:
                return 0
        for i in range(len(data_forward)):
            if ref >= data_forward[i]:
                return 0
        return 1


def pivot(osc, LBL, LBR, highlow):
    left = []
    right = []
    pivots = []
    for i in range(len(osc)):
        pivots.append(0.0)
        if i < LBL + 1:
            left.append(osc[i])
        if i > LBL:
            right.append(osc[i])
        if i > LBL + LBR:
            left.append(right[0])
            left.pop(0)
            right.pop(0)
            if checkhl(left, right, highlow):
                pivots[i - LBR] = osc[i - LBR]
    return pivots

pivots_low = pivot(df['Low'], pivot_left, pivot_right, 'low')
pivots_low = pd.DataFrame(pivots_low)
pivots_low.columns = ['pl']
pivots_low['pl'] = pivots_low['pl'].shift(pivot_right)

df = df.merge(pivots_low, how='inner', left_index=True, right_index=True)


pivots_high = pivot(df['High'], pivot_left, pivot_right, 'high')
pivots_high = pd.DataFrame(pivots_high)
pivots_high.columns = ['ph']
pivots_high['ph'] = pivots_high['ph'].shift(pivot_right)

df = df.merge(pivots_high, how='inner', left_index=True, right_index=True)


# Forward fill the zeros and derive previous pivot
df['pl'] = df['pl'].replace(0, pd.NaT).fillna(method='ffill')
df['pl_prev'] = np.where(df['pl'].ne(df['pl'].shift()), df['pl'].shift(), np.nan)
df['pl_prev'] = df['pl_prev'].ffill()

df['ph'] = df['ph'].replace(0, pd.NaT).fillna(method='ffill')
df['ph_prev'] = np.where(df['ph'].ne(df['ph'].shift()), df['ph'].shift(), np.nan)
df['ph_prev'] = df['ph_prev'].ffill()

# Create a new column 'pl_valuewhen' with forward-filled values from 'index'
df['pl_valuewhen_0'] = df['index'].where(df['pl'].diff().ne(0)).ffill()
df['ph_valuewhen_0'] = df['index'].where(df['ph'].diff().ne(0)).ffill()

# Calculate xAxis
df['pl_xAxis'] = df['pl_valuewhen_0'].diff().where(df['pl_valuewhen_0'].ne(df['pl_valuewhen_0'].shift())).ffill()
df['ph_xAxis'] = df['ph_valuewhen_0'].diff().where(df['ph_valuewhen_0'].ne(df['ph_valuewhen_0'].shift())).ffill()

# Evaluate pivot conditions
df['pl_pivotCond'] = np.where(
    df['pl'].diff().ne(0),
    np.where(df['pl'].astype(bool), df['pl'] > df['pl_prev'], df['pl'] < df['pl_prev']),
    False
)

df['ph_pivotCond'] = np.where(
    df['ph'].diff().ne(0),
    np.where(df['ph'].astype(bool), df['ph'] > df['ph_prev'], df['ph'] < df['ph_prev']),
    False
)



# Where pivot condition is true, the "trendline" (slope, y1, y2) resets.
df.loc[df['pl_pivotCond'], 'pl_slope'] = (df['pl'] - df['pl_prev']) / df['pl_xAxis']
df['pl_slope'] = df['pl_slope'].ffill()

df.loc[df['ph_pivotCond'], 'ph_slope'] = (df['ph'] - df['ph_prev']) / df['ph_xAxis']
df['ph_slope'] = df['ph_slope'].ffill()


# Calculate change in x (this appears to start from pivot_length and count by 1)
df['pl_changeInX'] = 0
count = 0
for index, row in df.iterrows():
    if row['pl_pivotCond']:
        count = pivot_right
    df.at[index, 'pl_changeInX'] = count
    count += 1
    
df['ph_changeInX'] = 0
count = 0
for index, row in df.iterrows():
    if row['ph_pivotCond']:
        count = pivot_right
    df.at[index, 'ph_changeInX'] = count
    count += 1



df['pl_gety2'] = df['pl'] + (df['pl_slope'] * df['pl_changeInX'])
df['ph_gety2'] = df['ph'] + (df['ph_slope'] * df['ph_changeInX'])

'''
df['Order'] = ''
df.loc[df['Close'] < df['pl_gety2'], 'Order'] = 'Short'
'''

# ~~~ The below code places the Short order such that it does not overlap. ~~~
df['pl_Order'] = ''
pivot_cond_mask = df['pl_pivotCond']

# Assign a unique group ID to each consecutive sequence of True values
group_id = pivot_cond_mask.cumsum()

# Filter the DataFrame where Close < pl_gety2
short_mask = df['Close'] < df['pl_gety2']
short_df = df[short_mask]

# Set 'Short' only for the first occurrence within each group
first_occurrence_mask = short_df.groupby(group_id).cumcount() == 0
df.loc[short_mask, 'pl_Order'] = first_occurrence_mask.map({True: 'Short', False: ''})

# Reset the index if needed
df.reset_index(drop=True, inplace=True)
# ~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~


# ~~~ The below code places the Long order such that it does not overlap. ~~~
df['ph_Order'] = ''
pivot_cond_mask = df['pl_pivotCond']

# Assign a unique group ID to each consecutive sequence of True values
group_id = pivot_cond_mask.cumsum()

# Filter the DataFrame where Close < pl_gety2
short_mask = df['Close'] > df['pl_gety2']
short_df = df[short_mask]

# Set 'Short' only for the first occurrence within each group
first_occurrence_mask = short_df.groupby(group_id).cumcount() == 0
df.loc[short_mask, 'ph_Order'] = first_occurrence_mask.map({True: 'Long', False: ''})

# Reset the index if needed
df.reset_index(drop=True, inplace=True)
# ~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~


# Calculate distances to orders
df['Group_ID'] = (df['pl_Order'] != df['pl_Order'].shift()).cumsum()
df['PL_Rows_In_Between'] = df.groupby('Group_ID').cumcount()
df['PL_Rows_In_Between'] = df['PL_Rows_In_Between'].astype(float)
df.drop('Group_ID', axis=1, inplace=True)

df['Group_ID'] = (df['ph_Order'] != df['ph_Order'].shift()).cumsum()
df['PH_Rows_In_Between'] = df.groupby('Group_ID').cumcount()
df['PH_Rows_In_Between'] = df['PH_Rows_In_Between'].astype(float)
df.drop('Group_ID', axis=1, inplace=True)

df.drop(columns=['pl_valuewhen_0','ph_valuewhen_0'],axis=1,inplace=True)




############################
# Parabolic SAR
############################


def calculate_multiple_sar(df, high_column='High', low_column='Low', sar_params=None):
    
    results = df[[high_column, low_column]].copy()
    
    for acceleration, maximum in sar_params:
        sar = talib.SAR(df[high_column], df[low_column], acceleration=acceleration, maximum=maximum)
        results[f'SAR_{acceleration}_{maximum}'] = sar
        
        # Add binary indicators for price above/below SAR
        results[f'Price_Above_SAR_{acceleration}_{maximum}'] = np.where(df['Close'] > sar, 1, 0)
        results[f'Price_Below_SAR_{acceleration}_{maximum}'] = np.where(df['Close'] < sar, 1, 0)
    
    return results


# Define custom SAR parameters if needed
custom_sar_params = [
    (0.02, 0.2),   # Traditional SAR
    (0.01, 0.1),   # Slower SAR
    (0.03, 0.3),   # Faster SAR
    (0.02, 0.4),   # Higher maximum SAR
    (0.05, 0.2),   # Higher acceleration SAR
    (0.015, 0.15), # Custom SAR 1
    (0.04, 0.25)   # Custom SAR 2
]

# Calculate SAR for multiple parameter pairs
sar_results = calculate_multiple_sar(df, high_column='High', low_column='Low', sar_params=custom_sar_params)

# Merge the SAR results back into the original DataFrame
df = pd.concat([df, sar_results.drop(columns=['High', 'Low'])], axis=1)






############################
# Elliott Wave Oscillator
############################

def calculate_multiple_ewo(df, close_column='Close', window_pairs=None):
    
    results = df[[close_column]].copy()
    
    for short_window, long_window in window_pairs:
        # Calculate short-term and long-term exponential moving averages
        short_ema = df[close_column].ewm(span=short_window, adjust=False).mean()
        long_ema = df[close_column].ewm(span=long_window, adjust=False).mean()
        
        # Calculate Elliott Wave Oscillator (EWO)
        ewo = short_ema - long_ema
        
        # Add the EWO to the results DataFrame
        results[f'EWO_{short_window}_{long_window}'] = ewo
    
    return results

custom_window_pairs = [
    (5, 35),    # Traditional EWO
    (3, 10),    # Short-term EWO
    (8, 34),    # Medium-term EWO
    (20, 100),  # Long-term EWO
    (50, 200),  # Very long-term EWO
    (13, 48),   # Custom EWO 1
    (34, 144)   # Custom EWO 2 (Fibonacci-inspired)
]

# Calculate EWO for multiple window pairs
ewo_results = calculate_multiple_ewo(df, close_column='Close', window_pairs=custom_window_pairs)
ewo_results.drop('Close', axis=1, inplace=True)

df = df.merge(ewo_results, how='inner', left_index=True, right_index=True)




############################
# OBV, VWAP, and Support/Res
############################

def calculate_obv(df):

    # Assign the initial OBV value (often 0)
    df['OBV'] = 0.0
    
    # Calculate the change in closing price
    df['Change'] = df['Close'] - df['Close'].shift(1)
    
    # Update OBV based on price change and volume
    df.loc[df['Change'] > 0, 'OBV'] = df['OBV'] + df['Log_Volume']
    df.loc[df['Change'] < 0, 'OBV'] = df['OBV'] - df['Log_Volume']
    
    # Drop the temporary 'Change' column
    df.drop('Change', axis=1, inplace=True)
    
    return df['OBV']

def calculate_vwap(df, window=20):
    """Calculate Volume-Weighted Average Price (VWAP)"""
    v = df['Log_Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * v).rolling(window=window).sum() / v.rolling(window=window).sum()


# Assuming 'df' is your pandas DataFrame with OHLCV data
df['OBV'] = calculate_obv(df)
df['VWAP'] = calculate_vwap(df)









############################
# Linear Regression Trendline
############################

cols_for_trendline = ['Close', 'Close_vs_ATH', 'Log_Volume', 'VWAP', 'VIX_Close']
look_backs = [10, 20, 50, 100]

def calculate_trend_slope(df, col_for_trendline, look_back):
    # Initialize a new column for trend slope
    df[col_for_trendline + '_' + str(look_back) + '_' + 'Trend_Slope'] = np.nan

    # Iterate over each row in the DataFrame
    for i in range(len(df)):
        # Determine the start and end indices for linear regression
        start_index = max(0, i - look_back + 1)
        end_index = i + 1
        
        # Extract the data for linear regression
        X = np.arange(start_index, end_index).reshape(-1, 1)
        y = df[col_for_trendline].fillna(0).iloc[start_index:end_index].values.reshape(-1, 1)

        # Initialize linear regression model
        model = LinearRegression()

        # Fit the model to the data
        model.fit(X, y)

        # Get the slope of the trend line
        slope = model.coef_[0][0]

        # Assign the slope to the corresponding row in the DataFrame
        df.at[i, col_for_trendline + '_' + str(look_back) + '_' + 'Trend_Slope'] = slope

# Call the function to calculate trend slope with a look back of 20 days
for col_for_trendline in cols_for_trendline:
    for look_back in look_backs:
        calculate_trend_slope(df, col_for_trendline, look_back)




############################
# $ Change in Time Vars
############################

''' Note: Unless I use a recurrent model (which the default DQN is not)
    the model will only see the snapshot current row of data, not what came before.
    The idea behind this is to derive some variables that show how things have 
    changed from the past 24 hrs, 48 hrs, and so on.
'''

# List of variables to process
variables = ['Close', 'Log_Volume', 'VIX_Close', 'VWAP']

# List of time periods for percent change calculation
periods = [5, 10, 20, 50, 100]

for var in variables:
    for period in periods:
        # Calculate percent change
        col_name = f'{var}_pct_change_{period}'
        df[col_name] = df[var].pct_change(periods=period)

        # Convert to percentage and round to 2 decimal places
        df[col_name] = (df[col_name] * 100).round(2)




############################
# ATR. THIS IS NOT USED AS A FEATURE, BUT JUST FOR EVALUATION OF TRADES LATER.
############################
# Add ATR for PPO learning agent
def smoothing(prices: pd.Series, smoothing_type, smoothing_period) -> pd.Series:
        if smoothing_type == 'sma':
            return talib.SMA(prices, smoothing_period)
        elif smoothing_type == 'ema':
            return talib.EMA(prices, smoothing_period)
        elif smoothing_type == 'rma':
            return pandas_ta.rma(prices, smoothing_period)
        elif smoothing_type is None:
            return prices

def run() -> pd.Series:
    atr = smoothing(talib.TRANGE(
        df['High'],
        df['Low'],
        df['Close'],
    ), "rma", 14)
    return atr    

atr_df = run()
atr_df = pd.DataFrame(atr_df).reset_index()
atr_df = atr_df[['RMA_14']]
atr_df.columns = ['ATR']

df = df.merge(atr_df, how='inner', left_index=True, right_index=True)
#df.dropna(inplace=True)







######################################################
# Further exclude the following cols 
######################################################
try:
    exc = ['index', 'Volume','QQQ_Close', 'Unnamed: 0']
    df = df.drop(exc, axis=1)
except:
    exc = ['index', 'Volume','QQQ_Close']
    df = df.drop(exc, axis=1)
data = df.copy()
 



# Replace nulls in numeric cols with 0
numeric_cols = data.select_dtypes(include=['float','int']).columns
data[numeric_cols] = data[numeric_cols].fillna(0)

# Replace nulls in object cols with 'Unknown'
object_cols = data.select_dtypes(include=['object']).columns
data[object_cols] = data[object_cols].fillna('Unknown')


# Drop any rows with remaining nulls
data = data.dropna()
data = data.reset_index(drop=True)



'''
Normalize all fields that track price
'''
var_list_normalize = ['ema10', 'ema20', 'ema50', 'sma100', 'sma200', 
                      'MA_20', 'Upper_BB_20', 'Lower_BB_20', 'MA_50', 'Upper_BB_50', 'Lower_BB_50', 
                      'MA_100','Upper_BB_100', 'Lower_BB_100', 'MA_200', 'Upper_BB_200', 'Lower_BB_200', 
                      'SD_20', 'SD_50', 'SD_100', 'SD_200',
                      'Rolling_Mean_10', 'Rolling_Mean_20', 'Rolling_Mean_50', 
                      'Rolling_Std_10', 'Rolling_Std_20', 'Rolling_Std_50',
                       'RSI_5', 'RSI_14', 'RSI_21', 'RSI_30', 'RSI_50', 'RSI_100', 'RSI_200',
                      'mean', 'older_mean', 'ohlc4', 'FZ2', 'AVG2', 'STDEV2', 'FZ2Limit', 'GZ2', 'GZ2Limit', 
                      'pl', 'ph', 'pl_prev', 'ph_prev', 'pl_gety2', 'ph_gety2', 
                      'SAR_0.02_0.2', 'SAR_0.01_0.1', 'SAR_0.03_0.3', 'SAR_0.02_0.4', 'SAR_0.05_0.2',
                      'SAR_0.015_0.15', 'SAR_0.04_0.25', ''
                      'hh', 'll', 'VWAP', 
                      'FearZoneOpen', 'FearZoneClose', 'GreedZoneOpen', 'GreedZoneClose',
                      ]

for var in var_list_normalize:
    data[var] = (data[var].astype(float) - data['Close']) / data['Close']





var_list_normalize_vix = [
                      'VIX_MA_20', 'VIX_Upper_BB_20', 'VIX_Lower_BB_20', 'VIX_MA_50', 'VIX_Upper_BB_50', 'VIX_Lower_BB_50', 
                      'VIX_MA_100','VIX_Upper_BB_100', 'VIX_Lower_BB_100', 'VIX_MA_200', 'VIX_Upper_BB_200', 'VIX_Lower_BB_200', 
                      'VIX_SD_20', 'VIX_SD_50', 'VIX_SD_100',
                      'VIX_Rolling_Mean_10', 'VIX_Rolling_Mean_20', 'VIX_Rolling_Mean_50', 
                      'VIX_Rolling_Std_10', 'VIX_Rolling_Std_20', 'VIX_Rolling_Std_50'
                         ]

for var in var_list_normalize_vix:
    data[var] = (data[var].astype(float) - data['VIX_Close']) / data['VIX_Close']




var_list_normalize_tnx = [
                      'TNX_MA_10', 'TNX_MA_20', 'TNX_MA_50', 'TNX_MA_100', 'TNX_MA_200'

                         ]

for var in var_list_normalize_tnx:
    data[var] = (data[var].astype(float) - data['TNX_Close']) / data['TNX_Close']






# Remove ETH so hour of day doesn't have useless values
# data = data[(data['Datetime'].dt.hour >= 9) & (data['Datetime'].dt.hour <= 16)]
# data = data.reset_index(drop=True)


# Apply encoders
var_list_ohe = ['Day_of_Week','Fearzone_Con','Greedzone_Con','pl_pivotCond','ph_pivotCond','pl_Order','ph_Order', 'Month_Name', 'Hour_of_Day']



# Initialize OneHotEncoder
encoder_ohe = OneHotEncoder(handle_unknown='ignore')
encoded_values = encoder_ohe.fit_transform(data[var_list_ohe])
encoded_df = pd.DataFrame(encoded_values.toarray(), columns=encoder_ohe.get_feature_names_out(var_list_ohe))

data = data.drop(columns=var_list_ohe).merge(encoded_df, how='inner', left_index=True, right_index=True)
#data = pd.concat([data.drop(columns=var_list_ohe), encoded_df], axis=1)


with open('one_hot_encoder.sav', 'wb') as f:
    pickle.dump(encoder_ohe, f)





# Optional: Remove highly correlated features
VARS_TO_EXCLUDE = ['Open','High','Low','Close', 'ATR', 'Datetime']

def remove_correlated_features(df, threshold=0.95):
    
    # Separate excluded variables
    df_excluded = df[VARS_TO_EXCLUDE]
    df_to_process = df.drop(columns=VARS_TO_EXCLUDE)
    
    # Calculate the correlation matrix
    corr_matrix = df_to_process.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Print info about dropped columns
    print(f"Dropped columns: {to_drop}")
    print(f"Number of columns dropped: {len(to_drop)}")
    print(f"Number of columns remaining: {len(df_to_process.columns) - len(to_drop)}")
    
    # Drop the highly correlated features
    df_filtered = df_to_process.drop(to_drop, axis=1)
    
    # Recombine with excluded variables
    df_final = pd.concat([df_excluded, df_filtered], axis=1)
    
    return df_final, to_drop

# Usage
# Assuming 'data' is your pandas DataFrame with features
filtered_data, dropped_features = remove_correlated_features(data, threshold=0.95)

data = filtered_data.copy()

# Split

data['Datetime'] = pd.to_datetime(data['Datetime'])

ppo_train = data[(data['Datetime'] < '2023-01-01') & (data['Datetime'] >= '2018-01-01')]
ppo_test = data[data['Datetime'] >= '2023-01-01']

ppo_train.to_csv('ppo_training_data_eth_noadj.csv', index=False)
ppo_test.to_csv('ppo_testing_data_eth_noadj.csv', index=False)




