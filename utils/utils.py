def create_features(dollar_bars):
    # Moving Averages
    for timeperiod in [5, 10, 20, 30, 50, 100]:
        dollar_bars[f'sma_{timeperiod}'] = talib.SMA(dollar_bars['close'], timeperiod=timeperiod)

    # Exponential Moving Averages
    for timeperiod in [5, 10, 20, 30, 50, 100]:
        dollar_bars[f'ema_{timeperiod}'] = talib.EMA(dollar_bars['close'], timeperiod=timeperiod)
    
    # Rate of Change
    for timeperiod in [5, 10, 20]:
        dollar_bars[f'roc_{timeperiod}'] = talib.ROC(dollar_bars['close'], timeperiod=timeperiod)
    
    # RSI
    for timeperiod in [7, 14, 21]:
        dollar_bars[f'rsi_{timeperiod}'] = talib.RSI(dollar_bars['close'], timeperiod=timeperiod)

    # Bollinger Bands
    for timeperiod in [10, 20, 50]:
        upper, middle, lower = talib.BBANDS(dollar_bars['close'], timeperiod=timeperiod)
        dollar_bars[f'upper_bb_{timeperiod}'] = upper
        dollar_bars[f'middle_bb_{timeperiod}'] = middle
        dollar_bars[f'lower_bb_{timeperiod}'] = lower

    # MACD
    for fastperiod, slowperiod, signalperiod in [(12, 26, 9), (5, 35, 5)]:
        macd, macdsignal, macdhist = talib.MACD(dollar_bars['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        prefix = f'macd_{fastperiod}_{slowperiod}_{signalperiod}'
        dollar_bars[f'{prefix}'] = macd
        dollar_bars[f'{prefix}_signal'] = macdsignal
        dollar_bars[f'{prefix}_hist'] = macdhist

    # Average True Range
    for timeperiod in [7, 14, 21]:
        dollar_bars[f'atr_{timeperiod}'] = talib.ATR(dollar_bars['high'], dollar_bars['low'], dollar_bars['close'], timeperiod=timeperiod)

    # Commodity Channel Index
    for timeperiod in [10, 20, 50]:
        dollar_bars[f'cci_{timeperiod}'] = talib.CCI(dollar_bars['high'], dollar_bars['low'], dollar_bars['close'], timeperiod=timeperiod)

    # Standard Deviation
    for timeperiod in [5, 10, 20]:
        dollar_bars[f'std_{timeperiod}'] = talib.STDDEV(dollar_bars['close'], timeperiod=timeperiod)

    # Momentum
    for timeperiod in [5, 10, 20]:
        dollar_bars[f'momentum_{timeperiod}'] = talib.MOM(dollar_bars['close'], timeperiod=timeperiod)


    # Percentage Price Oscillator
    for fastperiod, slowperiod, signalperiod in [(12, 26, 9), (5, 35, 5)]:
        ppo = talib.PPO(dollar_bars['close'], fastperiod=fastperiod, slowperiod=slowperiod)
        prefix = f'ppo_{fastperiod}_{slowperiod}'
        dollar_bars[f'{prefix}'] = ppo

    # On Balance Volume
    dollar_bars['obv'] = talib.OBV(dollar_bars['close'], dollar_bars['volume'])

    # Chaikin A/D Oscillator
    for fastperiod, slowperiod in [(3, 10), (5, 15)]:
        adosc = talib.ADOSC(dollar_bars['high'], dollar_bars['low'], dollar_bars['close'], dollar_bars['volume'], fastperiod=fastperiod, slowperiod=slowperiod)
        prefix = f'adosc_{fastperiod}_{slowperiod}'
        dollar_bars[f'{prefix}'] = adosc

    # Stochastic Oscillator
    for fastk_period, slowk_period, slowd_period in [(5, 3, 3), (14, 3, 3)]:
        slowk, slowd = talib.STOCH(dollar_bars['high'], dollar_bars['low'], dollar_bars['close'], fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=0, slowd_period=slowd_period, slowd_matype=0)
        prefix = f'stoch_{fastk_period}_{slowk_period}_{slowd_period}'
        dollar_bars[f'{prefix}_slowk'] = slowk
        dollar_bars[f'{prefix}_slowd'] = slowd

    return dollar_bars
