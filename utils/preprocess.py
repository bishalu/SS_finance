import numpy as np
import pandas as pd
from numba import njit

def clean_and_filter_data(df, remove_date_range=[]):
    """
    Cleans the input data by removing specified date ranges, dropping NaNs and duplicates, 
    sorting by datetime, and interpolating missing values.

    :param df: Input DataFrame containing OHLCV data with a 'datetime' column.
    :type df: pd.DataFrame
    :param remove_date_range: Optional list of two datetimes specifying a range of dates to remove.
    :type remove_date_range: list
    :return: Cleaned DataFrame.
    :rtype: pd.DataFrame
    """
    
    if len(remove_date_range) > 0:
        assert len(remove_date_range) == 2 
        df = pd.concat([df[df['datetime'] < remove_date_range[0]], df[df['datetime'] >= remove_date_range[1]]], axis=0)

    # Drop NaNs and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['datetime'], inplace=True)

    # Sort by datetime and interpolate missing values
    df.sort_values(by=['datetime'], inplace=True)
    df.set_index('datetime', inplace=True)

    return df.reset_index()


@njit
def _create_dollar_bars(price, volume, threshold):
    """
    Calculates dollar bars using Numba's Just-In-Time (JIT) compilation for improved performance.

    :param price: Array of close prices.
    :type price: np.array
    :param volume: Array of trading volumes.
    :type volume: np.array
    :param threshold: Dollar bar threshold.
    :type threshold: float
    :return: Arrays of open prices, high prices, low prices, close prices, bar volumes, and datetimes.
    :rtype: tuple of np.array
    """

    open_price = []
    high_price = []
    low_price = []
    close_price = []
    bar_volume = []
    datetimes = []
    
    cum_value = 0
    cum_volume = 0
    idx_open = 0
    
    for idx in range(price.shape[0]):
        cum_value += price[idx] * volume[idx]
        cum_volume += volume[idx]
        
        if cum_value >= threshold:
            open_price.append(price[idx_open])
            high_price.append(np.max(price[idx_open:idx+1]))
            low_price.append(np.min(price[idx_open:idx+1]))
            close_price.append(price[idx])
            bar_volume.append(cum_volume)
            datetimes.append(idx)
            
            cum_value = 0
            cum_volume = 0
            idx_open = idx + 1
    
    return np.array(open_price), np.array(high_price), np.array(low_price), np.array(close_price), np.array(bar_volume), np.array(datetimes)

def tick_to_dollar_bar(ohlcv_data, bars_per_day = 50):

    """
    Converts the cleaned OHLCV tick data into dollar bars.

    :param ohlcv_data: Input DataFrame containing cleaned OHLCV data.
    :type ohlcv_data: pd.DataFrame
    :param bars_per_day: Desired number of dollar bars per day.
    :type bars_per_day: int
    :return: DataFrame containing dollar bars.
    :rtype: pd.DataFrame
    """

    ohlcv_data = ohlcv_data.copy()

    # Convert OHLCV data to dollar bars
    total_dollar_value = ohlcv_data['close'].mul(ohlcv_data['volume']).sum()
    total_days = (ohlcv_data['datetime'].iloc[-1] - ohlcv_data['datetime'].iloc[0]).days
    threshold = total_dollar_value / (bars_per_day * total_days)

    open_price, high_price, low_price, close_price, bar_volume, datetimes = _create_dollar_bars(ohlcv_data['close'].to_numpy(), ohlcv_data['volume'].to_numpy(), threshold)

    # Create a DataFrame with dollar bars
    dollar_bars = pd.DataFrame({
        'datetime': ohlcv_data['datetime'].iloc[datetimes].values,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': bar_volume
    })

    return dollar_bars
