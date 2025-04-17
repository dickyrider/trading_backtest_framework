from binance.client import Client
import pandas as pd
import datetime


api_key = #api_key
api_secret = #secret_key


client = Client(api_key, api_secret)


symbol = "BTCUSDT"                      
interval = Client.KLINE_INTERVAL_1HOUR  # Hourly data
start_str = "1 Jan, 2022"                
end_str = "1 Apr, 2025"                  


klines = client.get_historical_klines(symbol, interval, start_str, end_str)

# Turn to dataframe
columns = [
    "Open Time", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Quote Asset Volume", "Number of Trades",
    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
]
df = pd.DataFrame(klines, columns=columns)


df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

# Turn to Yfinance format
df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]]


print(df.head())

df.to_csv('btc_data.csv')
print("Data saved to btc_data.csv")
