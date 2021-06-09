import json
import pandas as pd
import backtrader as bt
import btalib
import talib
from binance.client import Client
import numpy as np


def StochRSI(series, period=14, smoothK=3, smoothD=3):
    # Calculate RSI 
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI 
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    return stochrsi, stochrsi_K, stochrsi_D

with open('config.json') as f:
    data = json.load(f)

api_key = data["binance_api"]
api_secret = data["binance_secret"]
RSIperiod = data["period"]
smoothK = data["smoothK"]
smoothD = data["smoothD"]
lengthRSI = data["lengthRSI"]
lengthStoch = data["lengthStoch"]


client = Client(api_key, api_secret)
printJSON = json.dumps(client.get_account(), indent=2)
#print(client.get_asset_balance(asset='USDT'))

timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1d')
print(timestamp)
bars = client.get_historical_klines('BTCUSDT', '4h', timestamp, limit=1000)
# with open('btc_bars2.csv', 'w') as d:
#     for line in bars:
#         d.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}\n')
for line in bars:
    del line[5:]
btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
btc_df.set_index('date', inplace=True)

#print(btc_df.head())
btc_df.to_csv('BTCUSDT_bars.csv')
btc_df = pd.read_csv('BTCUSDT_bars.csv', index_col=0)
#btc_df.set_index('date', inplace=True)
btc_df.index = pd.to_datetime(btc_df.index, unit='ms')

# calculate 20 moving average using Pandas
btc_df['20sma'] = btc_df.close.rolling(20).mean()
#print(btc_df.tail(5))

sma = btalib.sma(btc_df.close)


RSI = talib.RSI(btc_df.close, RSIperiod)


# fastk, fastd = talib.STOCHRSI(btc_df.close, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
# print(pd.Series(fastk))
# print(pd.Series(fastd))

# k = sma(btalib.stoch(RSI, RSI, RSI, period=lengthStoch), smoothK)
# d = sma(k, smoothD)

#btc_df = btc_df.join([RSI.df])
btc_df.to_csv('stoch_result.csv')

