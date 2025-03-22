'''
05/19/2024:
    Alt 1 - Add a hold order option to the env.
    
v1.3 - Attempting to remove randomness in PCA.

v1.4 - I thought this was behind us, but the model itself is giving inconsistent results.
It's consistent when you fit it, and it's consistent when you load it. But those 2 are 
consistently different from one another. SOmething gets lost in the saving process of the 
model.

Update: This problem was being caused by the process that applied the model to make predictions.


v1.5 - GPU only - try a CNN approach
'''

import requests
import json
import os
import pandas as pd
import numpy as np
from datetime import timedelta, date
import time
import datetime
import talib
import math
import pandas_ta
from polygon import RESTClient
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import itertools
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
import multiprocessing
import warnings
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.utils import set_random_seed
import gym
import torch
import tensorflow as tf
import random
import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import joblib
import shutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tenacity import retry, stop_after_attempt, wait_exponential
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=Warning)
pd.options.mode.chained_assignment = None
pd.set_option('mode.chained_assignment', None)
pd.set_option('future.no_silent_downcasting', True)  
base_path = r'C:\Users\Peter\Dropbox\Stocks\Back-Testing\Machine Learning\DQN v2.2 Separate Models - 2024 Validation Data'
os.chdir(base_path)



# Set a seed value for reproducibility 
SEED = 75427

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED) 
torch.manual_seed(SEED)
set_random_seed(SEED)



client = RESTClient('_CLSbRc2T1I2xXJwvJ83DXQX1aCIGfeH')

global SLIPPAGE_PCT
SLIPPAGE_PCT = .0004

'''
resp = client.get_aggs('TQQQ', multiplier=1, timespan='minute', from_='2023-08-31', to='2023-09-04', limit=50000)
tqqq2 = pd.DataFrame(resp)
tqqq2['Datetime'] = pd.to_datetime(tqqq2['timestamp'],unit='ms')

# Subset the datetime column to times between 9:30 AM and 4:00 PM
start_time = pd.to_datetime('09:30:00').time()
end_time = pd.to_datetime('16:00:00').time()
subset = (tqqq2['Datetime'].dt.time >= start_time) & (tqqq2['Datetime'].dt.time <= end_time)
df_subset = tqqq2[subset]
df_subset.drop(['transactions','otc','timestamp'], axis=1, inplace=True)
df_subset.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Datetime']

tqqq = df_subset.copy()
'''

# To Test a specific day:
# tqqq = pd.read_csv(r"C:\Users\Peter\Dropbox\Stocks\Back-Testing\Machine Learning\Classification\UCSSMI Classification\UCSSMI Classification Live Automation\Data Timestamps\timestamp_2024-05-01 15-59-58.csv")
# #tqqq = pd.read_csv('polygon_test_data_no930_adj.csv')
# tqqq.columns = ['index','Datetime','Open','High','Low','Close','Volume','QQQ_Close']
# tqqq.drop(columns='index',axis=1,inplace=True)





VARS_TO_EXCLUDE = ['Open','High','Low','Close', 'ATR', 'Datetime']



def preprocess_features(data, feature_columns):
    features = data[feature_columns].copy()
    
    def to_unix_timestamp(d):
        if isinstance(d, (date, pd.Timestamp)):
            return int(pd.Timestamp(d).timestamp())
        return d

    for col in features.columns:
        if pd.api.types.is_datetime64_any_dtype(features[col]):
            features[col] = features[col].apply(to_unix_timestamp)
    
    features = features.apply(pd.to_numeric, errors='coerce')
    return features

def reduce_dimensions(train_data, test_data, n_components, random_state=42):
    """
    Reduce the dimensionality of both train and test data using PCA, 
    ensuring identical features in both datasets.
    """
    feature_columns = [col for col in train_data.columns if col not in VARS_TO_EXCLUDE]
    
    # Ensure test data has all the columns present in train data
    for col in feature_columns:
        if col not in test_data.columns:
            raise ValueError(f"Column '{col}' present in train data but missing in test data")
    
    # Preprocess both train and test features
    train_features = preprocess_features(train_data, feature_columns)
    test_features = preprocess_features(test_data, feature_columns)
    
    # Double-check that columns are identical
    if not train_features.columns.equals(test_features.columns):
        raise ValueError("Train and test feature columns are not identical after preprocessing")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    train_imputed = imputer.fit_transform(train_features)
    test_imputed = imputer.transform(test_features)
    
    # Standardize the data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)
    
    # Apply PCA with fixed random state and full SVD
    pca = PCA(n_components=n_components, random_state=random_state, svd_solver='full')
    
    # Use higher precision
    train_scaled = train_scaled.astype(np.float64)
    test_scaled = test_scaled.astype(np.float64)
    
    train_reduced = pca.fit_transform(train_scaled)
    test_reduced = pca.transform(test_scaled)
    
    # Create DataFrames with reduced features
    train_df = pd.DataFrame(train_reduced, columns=[f'PC_{i+1}' for i in range(n_components)])
    test_df = pd.DataFrame(test_reduced, columns=[f'PC_{i+1}' for i in range(n_components)])
    
    # Add back the excluded variables
    for col in VARS_TO_EXCLUDE:
        train_df[col] = train_data[col].values
        test_df[col] = test_data[col].values
    
    # Print the explained variance ratio
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")
    
    return train_df, test_df, pca, scaler, imputer


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_joblib_with_retry(data, filename):
    joblib.dump(data, filename)

def save_pca_transform(train_data, test_data, n_components, filename='pca_transform.joblib', random_state=42):
    train_data_reduced, test_data_reduced, pca, scaler, imputer = reduce_dimensions(train_data, test_data, n_components, random_state)
    
    transform_info = {
        'feature_columns': [col for col in train_data.columns if col not in VARS_TO_EXCLUDE],
        'pca': pca,
        'scaler': scaler,
        'imputer': imputer,
        'n_components': n_components,
        'random_state': random_state
    }
    
    try:
        save_joblib_with_retry(transform_info, filename)
    except Exception as e:
        print(f"Failed to save PCA transform after all retries: {str(e)}")
        raise  # Re-raise the exception if you want to halt execution
    
    return train_data_reduced, test_data_reduced

def apply_saved_pca_transform(new_data, filename='pca_transform.joblib'):
    # Load the saved transform information
    transform_info = joblib.load(filename)
    
    # Extract the necessary objects and information
    feature_columns = transform_info['feature_columns']
    pca = transform_info['pca']
    scaler = transform_info['scaler']
    imputer = transform_info['imputer']
    n_components = transform_info['n_components']
    
    # Preprocess the new data
    new_features = preprocess_features(new_data, feature_columns)
    
    # Apply the saved transformations
    new_imputed = imputer.transform(new_features)
    new_scaled = scaler.transform(new_imputed)
    new_reduced = pca.transform(new_scaled)
    
    # Create DataFrame with reduced features
    new_df = pd.DataFrame(new_reduced, columns=[f'PC_{i+1}' for i in range(n_components)])
    
    # Add back the excluded variables
    for col in VARS_TO_EXCLUDE:
        new_df[col] = new_data[col].values
    
    return new_df
    

def create_history_matrix(data, history_length):
    """Create a matrix of historical observations."""
    num_features = data.shape[1]
    num_samples = len(data) - history_length + 1
    print('num features, num samples:')
    print(num_features, num_samples)
    history_matrix = np.zeros((num_samples, history_length * num_features))
    
    for i in range(num_samples):
        history_matrix[i] = data.iloc[i:i+history_length].values.flatten()
    
    return history_matrix

def make_predictions(model, data, vars_to_exclude, history_length):
    """Make predictions using the model."""
    data_use = data.drop(vars_to_exclude, axis=1)
    
    # Create history matrix
    history_matrix = create_history_matrix(data_use, history_length)
    print('history matrix features and history length:')
    print(history_matrix.shape[1], history_length)
    
    # Make predictions in a single batch
    predictions, _ = model.predict(history_matrix)
    
    # Pad the predictions
    padding = np.full(history_length - 1, np.nan)
    full_predictions = np.concatenate([padding, predictions])
    
    return full_predictions



    

def backtester(atr_tp, atr_sl, short_reward_mult, neural_net, exp_rate, no_of_steps, pca_components, history_length):

    train_data = pd.read_csv('ppo_training_skinny_eth_noadj_options_ratioOnly.csv')
    test_data = pd.read_csv('ppo_testing_skinny_eth_noadj_options_ratioOnly.csv')
    
    train_data['Datetime'] = pd.to_datetime(train_data['Datetime'])
    test_data['Datetime'] = pd.to_datetime(test_data['Datetime'])
    
    train_data = train_data[(train_data['Datetime'].dt.hour >= 9) & (train_data['Datetime'].dt.hour <= 16)]
    test_data = test_data[(test_data['Datetime'].dt.hour >= 9) & (test_data['Datetime'].dt.hour <= 16)]
    
    # Use the function
    pca_filename = 'GPU_' + str(SEED) + '_' + str(pca_components) + '_Features' + '.joblib'
    train_data_reduced, test_data_reduced = save_pca_transform(train_data, test_data, pca_components, filename=pca_filename, random_state=42)

    train_data = train_data_reduced.copy()
    test_data = test_data_reduced.copy()
    
    
    '''
    # Optional: Import a prior saved PCA
    os.chdir('./Saved Models from BT')
    joblib_name = 'DQN2_ALT1_TEST_69_v4_redo_1.0_3.0_1.1_[512, 512]_0.2_100000'
    #joblib_name = 'pca_transform'
    train_data = apply_saved_pca_transform(train_data, joblib_name + '.joblib')
    test_data = apply_saved_pca_transform(test_data, joblib_name + '.joblib')
    os.chdir(base_path)
    '''
    
    


    
    
    
    
    
    class StockEnv(gym.Env):
        """Improved Stock Trading Environment with historical data."""
        metadata = {'render.modes': []}
    
        def __init__(self, data, transaction_cost=0.003, atr_tp=atr_tp, atr_sl=atr_sl, seed=None, history_length=history_length):
            self.data = data
            self.history_length = history_length
            self.current_day = history_length - 1  # Start after having enough history
            self.action_space = gym.spaces.Discrete(2)  # Only Short, Hold
            
            # Modify observation space to include historical data
            feature_count = len(train_data.drop(VARS_TO_EXCLUDE, axis=1).columns)
            self.observation_space = gym.spaces.Box(
                low=0, high=1, 
                shape=(feature_count * history_length,)
            )
            
            self.entry_day = None
            self.entry_price = None
            self.entry_atr = None
            self.current_price = None
            self.transaction_cost = transaction_cost
            self.atr_tp = atr_tp
            self.atr_sl = atr_sl
            self.total_reward = 0.0
            self.seed(seed)
    
        def seed(self, seed=None):
            if seed is not None:
                self.random_state = np.random.RandomState(seed)
                print('SEEDING WITH: ' + str(seed))
            else:
                self.random_state = np.random.RandomState()
            return [self.random_state.get_state()[1][0]]
    
        def reset(self):
            self.current_day = self.history_length - 1
            self.entry_price = None
            self.entry_day = None
            self.entry_atr = None
            self.total_reward = 0.0
            self.random_state.seed(self.seed()) 
            print("Environment Reset")
            return self._get_obs()
    
        def step(self, action):
            if action not in range(self.action_space.n):
                raise ValueError("Invalid action")
           
            if self.current_day >= len(self.data) - 1:
                done = True
                reward = 0
                return self._get_obs(), reward, done, {}
                
            self.current_price = self.data["Close"].iloc[self.current_day]
            current_price = self.data["Close"].iloc[self.current_day]
            atr = self.data["ATR"].iloc[self.current_day]
        
            self.entry_day = self.current_day
            self.entry_price = current_price
            self.entry_atr = atr
            
            tp_hit = False
            sl_hit = False
            max_adverse_move = 0
            max_favorable_move = 0
            future_day = self.current_day + 1
        
            while future_day < len(self.data):
                next_price_low = self.data["Low"].iloc[future_day]
                next_price_high = self.data["High"].iloc[future_day]
                next_price_open = self.data["Open"].iloc[future_day]
                
                # Calculate max adverse and favorable moves
                adverse_move = max(0, next_price_high - self.entry_price)
                favorable_move = max(0, self.entry_price - next_price_low)
                max_adverse_move = max(max_adverse_move, adverse_move)
                max_favorable_move = max(max_favorable_move, favorable_move)
        
                if action == 0:  # Short position
                    if self.entry_price - self.atr_tp * self.entry_atr >= next_price_low or self.entry_price - self.atr_tp * self.entry_atr >= next_price_open:
                        tp_hit = True
                        break
                    elif self.entry_price + self.atr_sl * self.entry_atr <= next_price_high or self.entry_price + self.atr_sl * self.entry_atr <= next_price_open:
                        sl_hit = True
                        break
                elif action == 1: # Hold position
                    if self.entry_price + self.atr_sl * self.entry_atr <= next_price_high or self.entry_price + self.atr_sl * self.entry_atr <= next_price_open:
                        sl_hit = True
                        break
                    elif self.entry_price - self.atr_tp * self.entry_atr >= next_price_low or self.entry_price - self.atr_tp * self.entry_atr >= next_price_open:
                        tp_hit = True
                        break
                
                if future_day == len(self.data) - 1:
                    if action == 0 and self.entry_price < next_price_open:
                        tp_hit = True
                    elif action == 0 and self.entry_price >= next_price_open:
                        sl_hit = True
                    elif action == 1 and self.entry_price >= next_price_open:
                        sl_hit = True
                    elif action == 1 and self.entry_price < next_price_open:
                        tp_hit = True
                    break
                
                future_day += 1
            
            # Calculate reward based on TP or SL hit
            if action == 0:  # Short position
                if tp_hit:
                    reward = ((self.atr_tp * self.entry_atr) / self.entry_price) 
                    reward -= max_adverse_move / self.entry_price  # Deduct the max adverse move
                elif sl_hit:
                    reward = -((self.atr_sl * self.entry_atr) / self.entry_price) * short_reward_mult
                    reward += max_favorable_move / self.entry_price  # Add the max favorable move
                else:
                    reward = 0
            elif action == 1:  # Hold position
                if sl_hit:
                    reward = ((self.atr_sl * self.entry_atr) / self.entry_price) * short_reward_mult  # Positive reward for correct hold
                    reward -= max_adverse_move / self.entry_price  # Deduct the max adverse move
                elif tp_hit:
                    reward = -((self.atr_tp * self.entry_atr) / self.entry_price)  # Negative reward for incorrect hold
                    reward += max_favorable_move / self.entry_price  # Add the max favorable move
                else:
                    reward = 0
        
            self.total_reward += reward
            done = self.current_day >= len(self.data) - 1
            self.current_day += 1
            return self._get_obs(), reward, done, {}
    
        def _get_obs(self):
            if self.current_day >= len(self.data):
                return np.zeros(self.observation_space.shape[0])
            else:
                feature_list = list(train_data.drop(VARS_TO_EXCLUDE, axis=1).columns)
                obs = []
                for i in range(self.history_length):
                    day = self.current_day - self.history_length + 1 + i
                    obs.extend([self.data[feature].iloc[day] for feature in feature_list])
                return np.array(obs)
            
            
            
    # Custom CNN Feature Extractor
    class CNNFeatureExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
            super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0] // history_length
            self.cnn = nn.Sequential(
                nn.Conv1d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            
            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float().reshape(-1, n_input_channels, history_length)).shape[1]
            
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU()
            )
    
        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            n_input_channels = observations.shape[1] // history_length
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, n_input_channels, history_length)
            return self.linear(self.cnn(observations))
    
    
    # Create the environment
    env = DummyVecEnv([lambda: StockEnv(train_data.copy())])

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    
    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=neural_net[0]),
        net_arch=neural_net
    )
    
    model = DQN('CnnPolicy', env, verbose=1, 
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4, 
                buffer_size=10000, 
                exploration_fraction=exp_rate, 
                seed=SEED,
                device=device)
    
    model.learn(total_timesteps=no_of_steps)
        
    
    
    


    
    '''
    # Create the environment
    env = StockEnv(train_data.copy())
    
    # Create and train the model
    model = DQN('MlpPolicy', env, verbose=1, 
                policy_kwargs=dict(net_arch=neural_net), 
                learning_rate=1e-4, 
                buffer_size=10000, 
                exploration_fraction=exp_rate, 
                seed=SEED)
    model.set_random_seed(SEED)
    model.learn(total_timesteps=no_of_steps)
    '''
    


    

    
    predictions = make_predictions(model, test_data, VARS_TO_EXCLUDE, history_length)
    test_data['predictions'] = predictions
    
    

   
    predictions = make_predictions(model, train_data, VARS_TO_EXCLUDE, history_length)
    train_data['predictions'] = predictions

    
    
    
    # Derive dates
    train_data['Datetime_x'] = pd.to_datetime(train_data['Datetime'])
    train_data['Date'] = train_data['Datetime_x'].dt.date
    
    test_data['Datetime_x'] = pd.to_datetime(test_data['Datetime'])
    test_data['Date'] = test_data['Datetime_x'].dt.date
    
    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
    ''' Apply Space Definitions '''
    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''       
    
    test_data.loc[(test_data['predictions'] == 0), 'Order'] = 'short'
    
    train_data.loc[(train_data['predictions'] == 0), 'Order'] = 'short'


    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
    ''' Order Handling '''
    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''       
    
    finishing_vars = []
    
    for df in [train_data, test_data]:

        df['net'] = 0.0
        df['Close Order'] = 0.0
        df['Close Reason'] = ''
        df['working'] = 0
        start_time = pd.to_datetime('09:00:00').time()
        end_time = pd.to_datetime('15:00:00').time()
        
        # Close order logic
        for i in range(len(df)):
            price = df.at[i, 'Close']
            tp = df.at[i, 'ATR'] * atr_tp
            sl = df.at[i, 'ATR'] * atr_sl
            order = df.at[i, 'Order']
            working = df.at[i, 'working']
            
                        
                
            if order == 'long' and working == 0 and (df.at[i, 'Datetime_x'].time() >= start_time) and (df.at[i, 'Datetime_x'].time() <= end_time) and (not pd.isna(df.at[i, 'ATR'])):
                j = 1
                while (i + j) < len(df):
                    if (df.at[i+j, 'Datetime_x'].time() >= start_time) and (df.at[i+j, 'Datetime_x'].time() <= end_time):
                        if atr_sl != 'none' and price - sl >=  df.at[i + j, 'Low'] and df.at[i+j, 'Datetime_x'].time() == start_time:
                            if price - sl >=  df.at[i + j, 'Open']:
                                df.at[i + j, 'Close Order'] = df.at[i + j, 'Open']
                                df.at[i + j, 'Close Reason'] = 'SL - Start Time' 
                                df.at[i, 'net'] = df.at[i + j, 'Open'] - price
                            else:
                                df.at[i + j, 'Close Order'] = price - sl
                                df.at[i + j, 'Close Reason'] = 'SL - Start Time' 
                                df.at[i, 'net'] = -sl
                            break
                        elif atr_sl != 'none' and price - sl >=  df.at[i + j, 'Low'] and df.at[i+j, 'Datetime_x'].time() != start_time:
                            df.at[i + j, 'Close Order'] = price - sl
                            df.at[i + j, 'Close Reason'] = 'SL' 
                            df.at[i, 'net'] = -sl
                            break
                        # This case handles trades open at start time.
                        elif atr_tp != 'none' and price + tp <= df.at[i + j, 'High'] and df.at[i+j, 'Datetime_x'].time() == start_time:
                            if price + tp <= df.at[i + j, 'Open']:
                                df.at[i + j, 'Close Order'] = df.at[i + j, 'Open']
                                df.at[i + j, 'Close Reason'] = 'TP - Start Time' 
                                df.at[i, 'net'] = df.at[i + j, 'Open'] - price
                            else:
                                df.at[i + j, 'Close Order'] = price + tp
                                df.at[i + j, 'Close Reason'] = 'TP - Start Time' 
                                df.at[i, 'net'] = tp
                            break
                        elif atr_tp != 'none' and price + tp <= df.at[i + j, 'High'] and df.at[i+j, 'Datetime_x'].time() != start_time:
                            df.at[i + j, 'Close Order'] = price + tp
                            df.at[i + j, 'Close Reason'] = 'TP' 
                            df.at[i, 'net'] = tp
                            break
                    
                    df.at[i + j, 'working'] = 1
                    j += 1
            elif order == 'short' and working == 0 and (df.at[i, 'Datetime_x'].time() >= start_time) and (df.at[i, 'Datetime_x'].time() <= end_time) and (not pd.isna(df.at[i, 'ATR'])):
                j = 1
                while (i + j) < len(df):
                    if (df.at[i+j, 'Datetime_x'].time() >= start_time) and (df.at[i+j, 'Datetime_x'].time() <= end_time):
                        if atr_sl != 'none' and price + sl <= df.at[i + j, 'High'] and df.at[i+j, 'Datetime_x'].time() == start_time:
                            if price + sl <=  df.at[i + j, 'Open']:
                                df.at[i + j, 'Close Order'] = df.at[i + j, 'Open']
                                df.at[i + j, 'Close Reason'] = 'SL - Start Time' 
                                df.at[i, 'net'] = price - df.at[i + j, 'Open']
                            else:
                                df.at[i + j, 'Close Order'] = price + sl
                                df.at[i + j, 'Close Reason'] = 'SL - Start Time' 
                                df.at[i, 'net'] = -sl
                            break
                        elif atr_sl != 'none' and price + sl <=  df.at[i + j, 'High'] and df.at[i+j, 'Datetime_x'].time() != start_time:
                            df.at[i + j, 'Close Order'] = price + sl
                            df.at[i + j, 'Close Reason'] = 'SL' 
                            df.at[i, 'net'] = -sl
                            break
                        # This case handles trades open at start time.
                        elif atr_tp != 'none' and  price - tp >= df.at[i + j, 'Low'] and df.at[i+j, 'Datetime_x'].time() == start_time:
                            if price - tp >= df.at[i + j, 'Open']:
                                df.at[i + j, 'Close Order'] = df.at[i + j, 'Open']
                                df.at[i + j, 'Close Reason'] = 'TP - Start Time' 
                                df.at[i, 'net'] = price - df.at[i + j, 'Open']
                            else:
                                df.at[i + j, 'Close Order'] = price - tp
                                df.at[i + j, 'Close Reason'] = 'TP - Start Time' 
                                df.at[i, 'net'] = tp
                            break
                        elif atr_tp != 'none' and price - tp >= df.at[i + j, 'Low'] and df.at[i+j, 'Datetime_x'].time() != start_time:
                            df.at[i + j, 'Close Order'] = price - tp
                            df.at[i + j, 'Close Reason'] = 'TP' 
                            df.at[i, 'net'] = tp
                            break
                    
                    df.at[i + j, 'working'] = 1
                    j += 1
                    
        # ~~~ Final Aggregations ~~~ #
        #df_final = df[['Datetime_x','Date','Open','High','Low','Close','Close_vs_ATH','ATR','Order','Close Order','Close Reason','working','net']]
        df_final = df.copy()
        df_final['net_percent'] = df_final['net'] / df_final['Close']
        
        mask = df_final['net_percent'] != 0
        
        # Apply the logic using the boolean mask
        df_final.loc[mask, 'net_percent_slippage_fwd'] = df_final.loc[mask, 'net_percent'] - SLIPPAGE_PCT
        df_final.loc[mask, 'net_percent_slippage_rev'] = df_final.loc[mask, 'net_percent'] + SLIPPAGE_PCT
        df_final['net_percent_slippage_fwd'].fillna(0, inplace=True)
        df_final['net_percent_slippage_rev'].fillna(0, inplace=True)
    
        
        df_final['cum_pct'] = 1000 * (1 + df_final['net_percent_slippage_fwd']).cumprod()
        cum_pct = (df_final['cum_pct'][len(df)-1] - 1000) / 1000
        df_final['cum_pct_reverse'] = 1000 * (1 + -1*df_final['net_percent_slippage_rev']).cumprod()
        cum_pct_reverse = (df_final['cum_pct_reverse'][len(df)-1] - 1000) / 1000
    
        grouped_data = df_final.groupby('Date')['net_percent'].sum().reset_index()
    
        consecutive_positives = (grouped_data['net_percent'] > 0).astype(int).diff().ne(0).cumsum()
        best_streak = grouped_data[grouped_data['net_percent'] > 0].groupby(consecutive_positives)['net_percent'].transform('size').max()
    
        consecutive_negatives = (grouped_data['net_percent'] < 0).astype(int).diff().ne(0).cumsum()
        worst_streak = grouped_data[grouped_data['net_percent'] < 0].groupby(consecutive_positives)['net_percent'].transform('size').max()
        
        no_of_trades = len(df_final[df_final['net_percent'] != 0.0])
        try:
            avg_trade = df_final['net_percent'].sum() / no_of_trades
            win_percentage = len(df_final[df_final['net_percent'] > 0]) / no_of_trades
        except:
            print('Division by 0')
            avg_trade = 0
            win_percentage = 0
            
        finishing_vars.extend([cum_pct, win_percentage, avg_trade, no_of_trades, best_streak, worst_streak])
        
        
    # Due to randomness (makes it impossible to reproduce models), save successful models out.
    if finishing_vars[9] > 10 and finishing_vars[7] > .67 and finishing_vars[0] > 1000:
        os.chdir('./Saved Models from BT')
        filename = 'GPU_' + str(SEED) + '_short_v2_' + str(atr_tp) + '_' + str(atr_sl) + '_' + str(short_reward_mult) + '_' + str(neural_net) + '_' + str(exp_rate) + '_' + str(no_of_steps) + '_' + str(pca_components) + '_' + str(history_length)
        model.save(filename.replace('.',''))
        os.chdir(base_path)
        
        
        # Also save out the PCA transforms
        # current_dir = os.getcwd()
        # source_file = os.path.join(current_dir, 'pca_transform.joblib')
        # dest_dir = os.path.join(current_dir, 'Saved Models from BT')
        # dest_file = os.path.join(dest_dir, filename.replace('.','') + '.joblib')
        
        # Move and rename the file
        # shutil.move(source_file, dest_file)
        

            

    return finishing_vars + [atr_tp, atr_sl, short_reward_mult, neural_net, exp_rate, no_of_steps, pca_components, history_length]
    


# DF to be filled
final_results_columns = ['TrainCumPct', 'TrainWinPct', 'TrainAvgTrade', 'TrainNoTrades', 'TrainBestStreak', 'TrainWorstStreak',
                        'TestCumPct', 'TestWinPct', 'TestAvgTrade', 'TestNoTrades', 'TestBestStreak', 'TestWorstStreak',
                         'ATR_TP', 'ATR_SL', 'ShortRewardMult', 'NeuralNet', 'ExpRate', 'NoOfSteps', 'PCAComponents', 'HistoryLength'
                        ]

final_results = pd.DataFrame(columns=final_results_columns)
    


# Testing grid
atr_tp_list = [3.0]
atr_sl_list = [3.0]
short_reward_mult_list = [2.0, 2.5, 3.0]
neural_net_list = [[128, 128], [256, 256], [512, 512], [1024, 1024]]
exp_rate_list = [.05, .10, .15]
no_of_steps_list = [10000, 20000, 30000]
history_length_list = [5, 10, 15, 20]
pca_components_list = [5, 10, 20, 30]


'''
atr_tp = 3
atr_sl = 3
short_reward_mult = 3.0
neural_net = [256, 256]
exp_rate = .15
no_of_steps = 30000
pca_components = 5
history_length = 5
'''





# Send success email
def send_email(sender_email, sender_password, receiver_email, subject, body):
    # Set up the MIME
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    # Add body to email
    message.attach(MIMEText(body, 'plain'))

    # Create SMTP session
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()  # Enable security
        server.login(sender_email, sender_password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)

# Example usage
sender_email = "pchilstrm@gmail.com"
sender_password = "uulq miyt majb gqvt"  # Use an app password, not your regular password
receiver_email = "pchilstrm@gmail.com"







# Either loop through the list using itertools, OR import the top X resuts from a prior timeframe and only run on those.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_results_with_retry(df, filename):
    df.to_csv(filename, index=False)

print('Using IterTools')
    
CSV_NAME = 'GPU_' + str(SEED) + '_short_v2.csv'
final_results.to_csv(CSV_NAME)

import time 
time.sleep(2)

num_cores = os.cpu_count()

num_gpus = torch.cuda.device_count()
max_workers = num_gpus if num_gpus > 0 else os.cpu_count()

# Generate all combinations of parameters
combinations = list(itertools.product(atr_tp_list, atr_sl_list, short_reward_mult_list, neural_net_list, exp_rate_list, no_of_steps_list, pca_components_list, history_length_list))

# Split combinations into quarters
quarter_length = len(combinations) // 16
quarters = [combinations[i:i + quarter_length] for i in range(0, len(combinations), quarter_length)]

def process_combinations(combinations_subset):
    results_subset = []
    for combination in combinations_subset:
        atr_tp, atr_sl, short_reward_mult, neural_net, exp_rate, no_of_steps, pca_components, history_length = combination
        print(str(datetime.datetime.now()), atr_tp, atr_sl, short_reward_mult, neural_net, exp_rate, no_of_steps, pca_components, history_length)
        result = backtester(atr_tp, atr_sl, short_reward_mult, neural_net, exp_rate, no_of_steps, pca_components, history_length)
        
        results_subset.append(result)
    return results_subset



if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=16) as executor:
        # Use as_completed to collect results asynchronously
        futures = [executor.submit(process_combinations, quarter) for quarter in quarters]
        
        # Initialize an empty list to collect results
        results = []
        for future in as_completed(futures):
            # Append the results from each process
            results.append(future.result())
    
    # Flatten the results and concatenate them into a single DataFrame
    final_results = pd.DataFrame(chain.from_iterable(results))
    
    # Optionally, perform any additional processing on final_results
    
    # Save final_results to CSV
    final_results.columns = final_results_columns
    try:
        save_results_with_retry(final_results, CSV_NAME)
    except Exception as e:
        print(f"Failed to save after all retries: {str(e)}")
    
    send_email(sender_email, sender_password, receiver_email, CSV_NAME, CSV_NAME)


    









'''
import plotly.express as px
fig = px.line(df_final, x='Datetime_x', y='cum_pct', labels={'cum_pct': 'Cumulative Percentage'},
              title='Cumulative Percentage Over Time')
fig.write_html('performance.html')


# To extract data for the long term testing:
test = df_final[['Date', 'net', 'net_percent', 'cum_pct', 'cum_pct_reverse']]
test = test.drop_duplicates()

test['year'] = test['Date'].astype(str).str[:4]
result = test.groupby('year')['net_percent'].sum()
result = result.reset_index()
print(result)


# Long breakdown
long_test = df_final[(df_final['Order'] == 'long') & (df_final['net'] != 0)]
long_test['year'] = long_test['Date'].astype(str).str[:4]
result = long_test.groupby('year')['net_percent'].sum()
result = result.reset_index()
print('-- Only Longs --')
print(result)


# Short breakdown
short_test = df_final[(df_final['Order'] == 'short') & (df_final['net'] != 0)]
short_test['year'] = short_test['Date'].astype(str).str[:4]
result = short_test.groupby('year')['net_percent'].sum()
result = result.reset_index()
print('-- Only Shorts --')
print(result)


# By month
long_test['year_mo'] = long_test['Date'].astype(str).str[:7]
result_long = long_test.groupby('year_mo')['net_percent'].sum()
result_long = result_long.reset_index()

short_test['year_mo'] = short_test['Date'].astype(str).str[:7]
result_short = short_test.groupby('year_mo')['net_percent'].sum()
result_short = result_short.reset_index()

result_all = result_long.merge(result_short, how='outer', on='year_mo')
result_all.columns = ['year_mo', 'long_net', 'short_net']

print('--- Number of long large loss months: ' + str(len(result_all[result_all['long_net'] <= -.10])))
print('--- Number of short large loss months: ' + str(len(result_all[result_all['short_net'] <= -.10])))


import plotly.express as px

fig = px.bar(result_all, x='year_mo', y=['long_net', 'short_net'],
             title='Long Net and Short Net Bar Chart',
             labels={'value': 'Net Position'},
             height=400)

# Show the plot
fig.write_html('long_vs_short_SMAs.html')


'''