from binance.client import Client
import pandas as pd
import datetime


api_key = #api_key
api_secret = #secret_key


# Create Binance client
client = Client(api_key, api_secret)

# Set parameters
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR  # 15-minute interval
start_str = "1 Apr, 2024"
end_str = "1 Apr, 2025"

# Fetch historical K-line data
klines = client.get_historical_klines(symbol, interval, start_str, end_str)

# Define column names
columns = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Quote Asset Volume", "Number of Trades",
    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
]

# Convert to DataFrame
df = pd.DataFrame(klines, columns=columns)

# Convert timestamps to datetime
df["Date"] = pd.to_datetime(df["Date"], unit="ms")
df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

# Select relevant columns and set index
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
df.set_index('Date', inplace=True)
df = df[['Close', 'High', 'Low', 'Open', 'Volume']]


# Save data to CSV
df.to_csv('btc_15min_data.csv')
