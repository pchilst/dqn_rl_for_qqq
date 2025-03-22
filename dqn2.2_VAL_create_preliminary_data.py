'''
09/15/24

'''


import pandas as pd
import os
from polygon import RESTClient
client = RESTClient('_')
from datetime import timedelta, date
import datetime


os.chdir(r'C:\Users\Peter\Dropbox\Stocks\Back-Testing\Machine Learning\DQN v2.2 Separate Models - Validation Data')



'''



# !! UPDATE: If no 9:30 adjustment needed: !!
    
df_filtered = pd.read_csv(r'TQQQ_full_1hour_adjsplit.txt')
df_filtered['Datetime'] = pd.to_datetime(df_filtered['Datetime'])


###########################################################################
# Find the QQQ hourly using Polygon API
###########################################################################
client = RESTClient('_CLSbRc2T1I2xXJwvJ83DXQX1aCIGfeH')

# Set the number of loops
num_loops = 50  # You can adjust this based on your requirement
    
polygon_current_time = int(datetime.datetime.now().timestamp() * 1000)

# Loop through the desired number of times
for i in range(num_loops):
    # Calculate timestamps for each iteration
    start_minutes = 100000 * (i + 1)
    end_minutes = 100000 * i
    
    #polygon_current_time = int(datetime.datetime.now().timestamp() * 1000)
    polygon_current_datetime = datetime.datetime.fromtimestamp(polygon_current_time / 1000)
    polygon_new_datetime = polygon_current_datetime - timedelta(minutes=start_minutes)
    polygon_new_timestamp = int(polygon_new_datetime.timestamp() * 1000)
    
    polygon_new_datetime_prev = polygon_current_datetime - timedelta(minutes=end_minutes)
    polygon_new_timestamp_prev = int(polygon_new_datetime_prev.timestamp() * 1000)
    
    # Make the API call for each iteration
    resp = client.get_aggs('QQQ', multiplier=1, timespan='hour', from_=polygon_new_timestamp, to=polygon_new_timestamp_prev, limit=50000)
    
    
    # Create a DataFrame from the response
    df = pd.DataFrame(resp)
    df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Datetime'] = pd.to_datetime(df['Datetime'] - pd.Timedelta(hours=4))
    
    
    # Append the results to the master DataFrame
    if i == 0:
        tmp1 = df.copy()
    else:
        tmp1 = pd.concat([df, tmp1])

# Reset the index of the final DataFrame
tmp1.reset_index(drop=True, inplace=True)

tmp1 = tmp1[['close','Datetime']]
tmp1.columns = ['QQQ_Close','Datetime']

#tmp1.to_csv('qqq_data_for_ath.csv')

# Optional: Merge to above df_filtered
tmp2 = df_filtered.merge(tmp1, how='left', on='Datetime')
#tmp2.to_csv('TQQQ_full_1hour_adjsplit_Aug20.csv', index=False)





###########################################################################
# VIX data
###########################################################################

vix = pd.read_csv('VIX_full_1hour.txt')
vix['Datetime'] = pd.to_datetime(vix['Datetime'])
vix = vix[['Datetime', 'Close']]
vix.columns = ['Datetime', 'VIX_Close']

df = tmp2.merge(vix, how='left', on='Datetime')
df['VIX_Close'] = df['VIX_Close'].ffill()
df['QQQ_Close'] = df['QQQ_Close'].ffill()
#df.to_csv('TQQQ_full_1hour_adjsplit_Aug20_no930adj.csv', index=False)



###########################################################################
# TNX data
###########################################################################

tnx = pd.read_csv('TNX_full_1hour.txt')
tnx['Datetime'] = pd.to_datetime(tnx['Datetime'])
tnx = tnx[['Datetime', 'Close']]
tnx.columns = ['Datetime', 'TNX_Close']

df = df.merge(tnx, how='left', on='Datetime')
df['TNX_Close'] = df['TNX_Close'].ffill()
df.to_csv('TQQQ_full_1hour_adjsplit_Nov4_no930adj.csv', index=False)



###########################################################################
# Optional: If avoiding a new pull of options, do this part.
###########################################################################


old_options = pd.read_csv('main_process_api_options_volume_history.csv')

new_options = pd.read_csv('LIVE_OPTIONS_DATA.CSV')
new_options = new_options.rename(columns={'Datetime': 'datetime'})


zero_rows_mask = old_options.drop('datetime', axis=1).eq(0).all(axis=1)

# Create mask for rows in new_options where any numeric column is non-zero
nonzero_rows_mask = new_options.drop('datetime', axis=1).ne(0).any(axis=1)

# Update old_options where conditions are met
old_options.loc[zero_rows_mask, :] = new_options.loc[nonzero_rows_mask, :]


old_options.to_csv('main_process_api_options_volume_history_updated.csv')


###########################################################################
# Pull all QQQ options historically
###########################################################################

expired_contracts = []
for option in client.list_options_contracts(underlying_ticker='QQQ',expiration_date_gte="2018-01-01",expired=True):
  expired_contracts.append(option)



def options_contracts_to_dataframe(contracts):
    # Extract the attributes from each OptionsContract object
    data = [{
        'additional_underlyings': contract.additional_underlyings,
        'cfi': contract.cfi,
        'contract_type': contract.contract_type,
        'correction': contract.correction,
        'exercise_style': contract.exercise_style,
        'expiration_date': contract.expiration_date,
        'primary_exchange': contract.primary_exchange,
        'shares_per_contract': contract.shares_per_contract,
        'strike_price': contract.strike_price,
        'ticker': contract.ticker,
        'underlying_ticker': contract.underlying_ticker
    } for contract in contracts]
    
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    
    return df

expired_df = options_contracts_to_dataframe(expired_contracts)





live_contracts = []
for option in client.list_options_contracts(underlying_ticker='QQQ',expiration_date_gte="2018-01-01",expired=False):
  live_contracts.append(option)


live_df = options_contracts_to_dataframe(live_contracts)


options_df = pd.concat([expired_df, live_df])

options_df.to_csv('QQQ_OPTIONS_SINCE_2018.csv')

'''














########################################################################################################
################## RUN AS BATCH ########################################################################
########################################################################################################

''' This portion pulls the actual volume of each above option and aggregates it appropriately '''

import os
os.chdir(r'C:\Users\Peter\Dropbox\Stocks\Back-Testing\Machine Learning\DQN v2.2 Separate Models - Validation Data')

import pandas as pd
from polygon import RESTClient
options_df = pd.read_csv('QQQ_OPTIONS_SINCE_2018.csv')
options_df_sub = options_df[:10000]
client = RESTClient("_CLSbRc2T1I2xXJwvJ83DXQX1aCIGfeH")
from datetime import datetime, timedelta
import concurrent.futures
from tqdm import tqdm

import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

def get_aggs_for_contract(client, contract):
    ticker = contract['ticker']
    start_date = contract['start_date']
    end_date = contract['expiration_date']
    aggs = []
    try:
        for a in client.list_aggs(
            ticker,
            1,
            "hour",
            start_date,
            end_date,
            limit=50000,
        ):
            aggs.append({
                'timestamp': a.timestamp,
                'volume': a.volume,
                'ticker': ticker
            })
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    return aggs

def process_date(args):
    date, options_df, aggs_df = args
    
    def get_next_expiring_options(date, options_df, option_type):
        next_options = options_df[(options_df['expiration_date'] > date) & 
                                  (options_df['contract_type'] == option_type)]
        if next_options.empty:
            return []
        next_expiration = next_options['expiration_date'].min()
        return next_options[next_options['expiration_date'] == next_expiration]['ticker'].tolist()

    def get_volume_for_date_range(date, options_df, aggs_df, option_type, min_days, max_days):
        relevant_options = options_df[
            (options_df['expiration_date'] > date + timedelta(days=min_days)) &
            (options_df['expiration_date'] <= date + timedelta(days=max_days)) &
            (options_df['contract_type'] == option_type)
        ]
        relevant_tickers = relevant_options['ticker'].tolist()
        return aggs_df[(aggs_df['datetime'].dt.floor('h') == date) & 
                       (aggs_df['ticker'].isin(relevant_tickers))]['volume'].sum()

    next_call_tickers = get_next_expiring_options(date, options_df, 'call')
    next_put_tickers = get_next_expiring_options(date, options_df, 'put')
    
    return {
        'datetime': date,
        'next_call_volume': aggs_df[(aggs_df['datetime'].dt.floor('h') == date) & 
                                    (aggs_df['ticker'].isin(next_call_tickers))]['volume'].sum(),
        'next_put_volume': aggs_df[(aggs_df['datetime'].dt.floor('h') == date) & 
                                   (aggs_df['ticker'].isin(next_put_tickers))]['volume'].sum(),
        'calls_7d': get_volume_for_date_range(date, options_df, aggs_df, 'call', 0, 7),
        'puts_7d': get_volume_for_date_range(date, options_df, aggs_df, 'put', 0, 7),
        'calls_8_30d': get_volume_for_date_range(date, options_df, aggs_df, 'call', 7, 30),
        'puts_8_30d': get_volume_for_date_range(date, options_df, aggs_df, 'put', 7, 30),
        'calls_31d_plus': get_volume_for_date_range(date, options_df, aggs_df, 'call', 30, 1000),
        'puts_31d_plus': get_volume_for_date_range(date, options_df, aggs_df, 'put', 30, 1000),
        'all_calls': get_volume_for_date_range(date, options_df, aggs_df, 'call', 0, 1000),
        'all_puts': get_volume_for_date_range(date, options_df, aggs_df, 'put', 0, 1000)
    }

def aggregate_options_volume(client, options_df, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()

    print(f"Using {max_workers} workers")

    options_df['expiration_date'] = pd.to_datetime(options_df['expiration_date'])
    options_df = options_df.sort_values('expiration_date')
    
    global_start_date = datetime(2018, 1, 1).date()
    options_df['start_date'] = global_start_date.strftime('%Y-%m-%d')
    
    # Fetch aggregates in the main process
    all_aggs = []
    for _, contract in tqdm(options_df.iterrows(), total=len(options_df), desc="Fetching data"):
        all_aggs.extend(get_aggs_for_contract(client, contract))
    
    aggs_df = pd.DataFrame(all_aggs)
    if aggs_df.empty:
        print("No aggregate data found for any contracts.")
        return pd.DataFrame()
    
    aggs_df['datetime'] = pd.to_datetime(aggs_df['timestamp'], unit='ms')
    
    end_date = options_df['expiration_date'].max()
    date_range = pd.date_range(start=global_start_date, end=end_date, freq='h')
    
    # Process dates using multiprocessing
    volume_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_date, (date, options_df, aggs_df)) for date in date_range]
        
        for future in tqdm(as_completed(futures), total=len(date_range), desc="Processing dates"):
            volume_data.append(future.result())
    
    volume_df = pd.DataFrame(volume_data)
    volume_df.set_index('datetime', inplace=True)
    volume_df.sort_index(inplace=True)
    
    return volume_df

if __name__ == "__main__":
    start_time = datetime.now()
    master_df = aggregate_options_volume(client, options_df, max_workers=50)
    end_time = datetime.now()
    print(f"Total execution time: {end_time - start_time}")

    if not master_df.empty:
        print(master_df.head())
        print(master_df.describe())
        
        for column in ['next_call_volume', 'next_put_volume']:
            non_zero = master_df[master_df[column] > 0]
            print(f"\n{column}:")
            print(f"Total entries: {len(master_df)}")
            print(f"Non-zero entries: {len(non_zero)} ({len(non_zero) / len(master_df) * 100:.2f}%)")
            print(f"Mean of non-zero values: {non_zero[column].mean():.2f}")
            print(f"Median of non-zero values: {non_zero[column].median():.2f}")
        
        master_df.to_csv('main_process_api_options_volume_history.csv')
    else:
        print("No data to save.")


########################################################################################################
################## END RUN AS BATCH ####################################################################
########################################################################################################



'''

###########################################################################
# Merge options volume history to other base data
###########################################################################

base_data = pd.read_csv('TQQQ_full_1hour_adjsplit_Nov4_no930adj.csv')
base_data['Datetime'] = pd.to_datetime(base_data['Datetime'])


options_data = pd.read_csv('main_process_api_options_volume_history.csv')


# For options data, do a time adjustment.

options_data['Datetime'] = pd.to_datetime(options_data['datetime'])


def adjust_time_for_dst(dt):
    # Function to determine if a date is in DST
    def is_dst(date):
        # DST starts on the second Sunday in March
        dst_start = pd.Timestamp(f"{date.year}-03-01") + pd.Timedelta(days=(6 - pd.Timestamp(f"{date.year}-03-01").dayofweek + 7) % 7 + 7)
        # DST ends on the first Sunday in November
        dst_end = pd.Timestamp(f"{date.year}-11-01") + pd.Timedelta(days=(6 - pd.Timestamp(f"{date.year}-11-01").dayofweek) % 7)
        
        return dst_start <= date < dst_end

    # Determine the offset based on DST
    offset = pd.Timedelta(hours=4) if is_dst(dt) else pd.Timedelta(hours=5)
    
    return dt - offset

# Apply the adjustment to your DataFrame
options_data['Datetime'] = options_data['Datetime'].apply(adjust_time_for_dst)


options_data.drop('datetime', axis=1, inplace=True)


base_data_w_options = base_data.merge(options_data, how='left', on = 'Datetime')

base_data_w_options.to_csv('TQQQ_full_1hour_adjsplit_Nov4_no930adj_options.csv', index=False)
'''