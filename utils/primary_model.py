
import pandas as pd
import numpy as np
from numba import njit
from concurrent.futures import ThreadPoolExecutor



def generate_trading_signals_ma(dollar_bars, fast_window, slow_window):
    # Calculate moving averages
    dollar_bars['fast_ma'] = dollar_bars['close'].rolling(fast_window).mean()
    dollar_bars['slow_ma'] = dollar_bars['close'].rolling(slow_window).mean()

    # Generate trading signals
    dollar_bars['side'] = np.where(dollar_bars['fast_ma'] > dollar_bars['slow_ma'], 1, -1)

    return dollar_bars.dropna()


@njit
def _cusum_filter(returns, threshold):
    num_returns = len(returns)
    events = np.zeros(num_returns, dtype=np.int64)
    pos_sum = 0.0
    neg_sum = 0.0

    for i in range(1, num_returns):
        pos_sum = max(0, pos_sum + returns[i])
        neg_sum = min(0, neg_sum + returns[i])

        if pos_sum > threshold:
            events[i] = 1
            pos_sum = 0.0
            neg_sum = 0.0
        elif neg_sum < -threshold:
            events[i] = 1
            pos_sum = 0.0
            neg_sum = 0.0

    return events

def generate_cusum_events(dollar_bars, threshold):

    dollar_bars.loc[:,'returns'] = dollar_bars['close'].pct_change()
    events = _cusum_filter(dollar_bars['returns'].values, threshold)
    trading_events = dollar_bars[events == 1][['datetime', 'side']].reset_index(drop=True)
    dollar_bars.drop(['returns'], axis=1, inplace=True)

    return trading_events

'''
def _triple_barrier_method_single(i, event_datetimes, event_sides, dollar_bar_datetimes, dollar_bar_closes, pt, sl, min_ret, num_days):
    event_time = event_datetimes[i]
    event_side = event_sides[i]
    event_idx = np.searchsorted(dollar_bar_datetimes, event_time)
    event_price = dollar_bar_closes[event_idx]
    time_limit = event_time + pd.Timedelta(days=num_days)

    touched = {
        'pt': False,
        'sl': False,
        'vb': False
    }

    for j in range(event_idx + 1, len(dollar_bar_datetimes)):
        row_time = dollar_bar_datetimes[j]

        if row_time > time_limit:
            touched['vb'] = True
            break

        if event_side == 1:
            price_return = (dollar_bar_closes[j] - event_price) / event_price
        else:
            price_return = (event_price - dollar_bar_closes[j]) / event_price

        if price_return >= pt and not touched['pt']:
            touched['pt'] = True
            return [event_time, 'pt', price_return, row_time, event_side]

        elif price_return <= -sl and not touched['sl']:
            touched['sl'] = True
            return [event_time, 'sl', price_return, row_time, event_side]

        if touched['pt'] and touched['sl']:
            break

    if touched['vb'] and not (touched['pt'] or touched['sl']):
        if event_side == 1:
            price_return = (dollar_bar_closes[j] - event_price) / event_price
        else:
            price_return = (event_price - dollar_bar_closes[j]) / event_price
        if abs(price_return) > min_ret:
            return [event_time, 'vb', price_return, row_time, event_side]

    return [None, None, None, None, None]


def triple_barrier_method(dollar_bars, events, pt, sl, min_ret, num_days):
    event_datetimes = events['datetime'].values
    event_sides = events['side'].values
    dollar_bar_datetimes = dollar_bars['datetime'].values
    dollar_bar_closes = dollar_bars['close'].values

    num_events = len(events)
    results = []

    with ThreadPoolExecutor() as executor:
        tasks = [executor.submit(_triple_barrier_method_single, i, event_datetimes, event_sides, dollar_bar_datetimes, dollar_bar_closes, pt, sl, min_ret, num_days)
                 for i in range(num_events)]
        results = [task.result() for task in tasks]

    labels_df = pd.DataFrame(results, columns=['datetime', 'type', 'return', 't1', 'side'])
    labels_df.dropna(inplace=True)
    labels_df.reset_index(drop=True, inplace=True)

    return labels_df
'''

def _triple_barrier_method_single(i, event_datetimes, event_sides, dollar_bar_datetimes, dollar_bar_closes, pt, sl, min_ret, num_days):
    event_time = event_datetimes[i]
    event_side = event_sides[i]
    event_idx = np.searchsorted(dollar_bar_datetimes, event_time)
    event_price = dollar_bar_closes[event_idx]
    time_limit = event_time + pd.Timedelta(days=num_days)

    touched = {
        'pt': False,
        'sl': False,
        'vb': False
    }

    for j in range(event_idx + 1, len(dollar_bar_datetimes)):
        row_time = dollar_bar_datetimes[j]

        if row_time > time_limit:
            touched['vb'] = True
            break

        if event_side == 1:
            price_return = (dollar_bar_closes[j] - event_price) / event_price
        else:
            price_return = (event_price - dollar_bar_closes[j]) / event_price

        if price_return >= pt and not touched['pt']:
            touched['pt'] = True
            return [event_time, 'pt', price_return, row_time, event_side, event_price, dollar_bar_closes[j]]

        elif price_return <= -sl and not touched['sl']:
            touched['sl'] = True
            return [event_time, 'sl', price_return, row_time, event_side, event_price, dollar_bar_closes[j]]

        if touched['pt'] and touched['sl']:
            break

    if touched['vb'] and not (touched['pt'] or touched['sl']):
        if event_side == 1:
            price_return = (dollar_bar_closes[j] - event_price) / event_price
        else:
            price_return = (event_price - dollar_bar_closes[j]) / event_price
        if abs(price_return) > min_ret:
            return [event_time, 'vb', price_return, row_time, event_side, event_price, dollar_bar_closes[j]]

    return [None, None, None, None, None, None, None]


def triple_barrier_method(dollar_bars, events, pt, sl, min_ret, num_days):
    event_datetimes = events['datetime'].values
    event_sides = events['side'].values
    dollar_bar_datetimes = dollar_bars['datetime'].values
    dollar_bar_closes = dollar_bars['close'].values

    num_events = len(events)
    results = []

    with ThreadPoolExecutor() as executor:
        tasks = [executor.submit(_triple_barrier_method_single, i, event_datetimes, event_sides, dollar_bar_datetimes, dollar_bar_closes, pt, sl, min_ret, num_days)
                 for i in range(num_events)]
        results = [task.result() for task in tasks]

    labels_df = pd.DataFrame(results, columns=['datetime', 'type', 'return', 't1', 'side', 'initial_price', 'final_price'])
    labels_df.dropna(inplace=True)
    labels_df.reset_index(drop=True, inplace=True)

    return labels_df
