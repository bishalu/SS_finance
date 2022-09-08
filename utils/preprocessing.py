import numpy as np
import pandas as pd
from multiprocess import mp_pandas_obj

def get_parksinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator

    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Parkinson volatility
    """

    ret = np.log(high / low)  # High/Low return
    estimator = 1 / (4 * np.log(2)) * (ret ** 2)
    return np.sqrt(estimator.rolling(window=window).mean())


def get_garman_class_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int = 20) -> pd.Series:
    """
    Garman-Class volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Garman-Class volatility
    """

    ret = np.log(high / low)  # High/Low return
    close_open_ret = np.log(close / open)  # Close/Open return
    estimator = 0.5 * ret ** 2 - (2 * np.log(2) - 1) * close_open_ret ** 2
    return np.sqrt(estimator.rolling(window=window).mean())


def get_yang_zhang_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                       window: int = 20) -> pd.Series:
    """
    Yang-Zhang volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Yang-Zhang volatility
    """

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    open_prev_close_ret = np.log(open / close.shift(1))
    close_prev_open_ret = np.log(close / open.shift(1))

    high_close_ret = np.log(high / close)
    high_open_ret = np.log(high / open)
    low_close_ret = np.log(low / close)
    low_open_ret = np.log(low / open)

    sigma_open_sq = 1 / (window - 1) * (open_prev_close_ret ** 2).rolling(window=window).sum()
    sigma_close_sq = 1 / (window - 1) * (close_prev_open_ret ** 2).rolling(window=window).sum()
    sigma_rs_sq = 1 / (window - 1) * (high_close_ret * high_open_ret + low_close_ret * low_open_ret).rolling(
        window=window).sum()

    return np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)




def cusum_filter(raw_time_series, threshold, time_stamps=True):
    """
    Advances in Financial Machine Learning, Snippet 2.4, page 39.

    The Symmetric Dynamic/Fixed CUSUM Filter.

    The CUSUM filter is a quality-control method, designed to detect a shift in the mean value of a measured quantity
    away from a target value. The filter is set up to identify a sequence of upside or downside divergences from any
    reset level zero. We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.

    One practical aspect that makes CUSUM filters appealing is that multiple events are not triggered by raw_time_series
    hovering around a threshold level, which is a flaw suffered by popular market signals such as Bollinger Bands.
    It will require a full run of length threshold for raw_time_series to trigger an event.

    Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine whether the occurrence
    of such events constitutes actionable intelligence. Below is an implementation of the Symmetric CUSUM filter.

    Note: As per the book this filter is applied to closing prices but we extended it to also work on other
    time series such as volatility.

    :param raw_time_series: (pd.Series) Close prices (or other time series, e.g. volatility).
    :param threshold: (float or pd.Series) When the abs(change) is larger than the threshold, the function captures
                      it as an event, can be dynamic if threshold is pd.Series
    :param time_stamps: (bool) Default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) Vector of datetimes when the events occurred. This is used later to sample.
    """

    t_events = []
    s_pos = 0
    s_neg = 0

    # log returns
    raw_time_series = pd.DataFrame(raw_time_series)  # Convert to DataFrame
    raw_time_series.columns = ['price']
    raw_time_series['log_ret'] = raw_time_series.price.apply(np.log).diff()
    if isinstance(threshold, (float, int)):
        raw_time_series['threshold'] = threshold
    elif isinstance(threshold, pd.Series):
        raw_time_series.loc[threshold.index, 'threshold'] = threshold
    else:
        raise ValueError('threshold is neither float nor pd.Series!')

    raw_time_series = raw_time_series.iloc[1:]  # Drop first na values

    # Get event time stamps for the entire series
    for tup in raw_time_series.itertuples():
        thresh = tup.threshold
        pos = float(s_pos + tup.log_ret)
        neg = float(s_neg + tup.log_ret)
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -thresh:
            s_neg = 0
            t_events.append(tup.Index)

        elif s_pos > thresh:
            s_pos = 0
            t_events.append(tup.Index)

    # Return DatetimeIndex or list
    if time_stamps:
        event_timestamps = pd.DatetimeIndex(t_events)
        return event_timestamps

    return t_events




# Snippet 3.4 page 49, Adding a Vertical Barrier
def add_vertical_barrier(t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0):
    """
    Advances in Financial Machine Learning, Snippet 3.4 page 49.

    Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    :param t_events: (pd.Series) Series of events (symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days: (int) Number of days to add for vertical barrier
    :param num_hours: (int) Number of hours to add for vertical barrier
    :param num_minutes: (int) Number of minutes to add for vertical barrier
    :param num_seconds: (int) Number of seconds to add for vertical barrier
    :return: (pd.Series) Timestamps of vertical barriers
    """
    timedelta = pd.Timedelta(
        '{} days, {} hours, {} minutes, {} seconds'.format(num_days, num_hours, num_minutes, num_seconds))
    # Find index to closest to vertical barrier
    nearest_index = close.index.searchsorted(t_events + timedelta)

    # Exclude indexes which are outside the range of close price index
    nearest_index = nearest_index[nearest_index < close.shape[0]]

    # Find price index closest to vertical barrier time stamp
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]

    vertical_barriers = pd.Series(data=nearest_timestamp, index=filtered_events)
    return vertical_barriers



def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False,
               side_prediction=None, verbose=True):
    """
    Advances in Financial Machine Learning, Snippet 3.6 page 50.

    Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (pd.Series) Close prices
    :param t_events: (pd.Series) of t_events. These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    :param pt_sl: (2 element array) Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (pd.Series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (pd.Series) A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (pd.Series) Side of the bet (long/short) as decided by the primary model
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
            -events['pt'] is profit taking multiple
            -events['sl']  is stop loss multiple
    """

    # 1) Get target
    target = target.reindex(t_events)
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events, dtype=t_events.dtype)

    # 3) Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side_prediction.reindex(target.index)  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
    events = events.dropna(subset=['trgt'])

    # Apply Triple Barrier
    first_touch_dates = mp_pandas_obj(func=apply_pt_sl_on_t1,
                                      pd_obj=('molecule', events.index),
                                      num_threads=num_threads,
                                      close=close,
                                      events=events,
                                      pt_sl=pt_sl_,
                                      verbose=verbose)

    for ind in events.index:
        events.at[ind, 't1'] = first_touch_dates.loc[ind, :].dropna().min()

    if side_prediction is None:
        events = events.drop('side', axis=1)

    # Add profit taking and stop loss multiples for vertical barrier calculations
    events['pt'] = pt_sl[0]
    events['sl'] = pt_sl[1]

    return events


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):  # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.

    Triple Barrier Labeling Method

    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.

    :param close: (pd.Series) Close prices
    :param events: (pd.Series) Indices that signify "events" (see cusum_filter function
    for more details)
    :param pt_sl: (np.array) Element 0, indicates the profit taking level; Element 1 is stop loss level
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """

    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index)  # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index)  # NaNs

    out['pt'] = pd.Series(dtype=events.index.dtype)
    out['sl'] = pd.Series(dtype=events.index.dtype)

    # Get events
    for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
        closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
        out.at[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # Earliest stop loss date
        out.at[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # Earliest profit taking date

    return out