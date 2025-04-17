from binance.client import Client
import pandas as pd
import datetime

# 若查詢公共數據，可以將 api_key 和 api_secret 留空字串
api_key = "GmguUeFXCrRfs1yAy5SvnI0MBip184HrCphw99qVyXoCncYeSb4ClDTVVv1bVLE4"
api_secret = "4i8R7o5Uu3QgJTzxrmlnwtACEhzYpOFXZoPn1ZRmkVayPYlIAsLP5YeO3EokL33O"

# 初始化 Client
client = Client(api_key, api_secret)

# 參數設定
symbol = "BTCUSDT"                      # 市場 (BTC 對 USDT)
interval = Client.KLINE_INTERVAL_1HOUR  # 1小時K線資料
start_str = "1 Jan, 2022"                # 起始時間，可以使用 Binance 可識別的時間字符串
end_str = "1 Apr, 2025"                  # 結束時間 (選填)

# 獲取歷史 K 線數據
# 如果不傳入 end_str，則返回起始時間到目前的數據
klines = client.get_historical_klines(symbol, interval, start_str, end_str)

# 將數據轉換為 DataFrame
# Binance 回傳的每筆資料包含：開盤時間, 開盤價, 最高價, 最低價, 收盤價, 成交量,
# 收盤時間, 成交額, 成交筆數, 主動買入成交量, 主動買入成交額, 忽略字段
columns = [
    "Open Time", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Quote Asset Volume", "Number of Trades",
    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
]
df = pd.DataFrame(klines, columns=columns)

# 將時間戳記轉換為日期時間格式（以毫秒為單位）
df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

# 若不需要最後兩個時間戳記，可以只保留部分欄位
df = df[["Open Time", "Open", "High", "Low", "Close", "Volume"]]

# 顯示 DataFrame 的前五筆資料
print(df.head())

df.to_csv('btc_data.csv')
print("Data saved to btc_data.csv")