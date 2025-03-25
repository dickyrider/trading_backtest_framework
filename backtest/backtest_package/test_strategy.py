import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
from frame import Backtest 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Strategy:
    def __init__(self, data, **kwargs):
        self.data = data
        self.short_window = 10
        self.long_window = 25
        self.volume_window = kwargs.get('volume_window', 1) 

    def add_reference_data(self, backtest, ticker):
        backtest.df['S_EWA'] = backtest.df[ticker].ewm(span=self.short_window, adjust=False).mean()
        backtest.df['L_EWA'] = backtest.df[ticker].ewm(span=self.long_window, adjust=False).mean()
        backtest.df['EWA_diff'] = backtest.df['L_EWA'] - backtest.df['S_EWA'] 
        backtest.df['Average_Volume'] = backtest.df['Volume'].ewm(span=self.volume_window, adjust=False).mean()

        #RSI
        delta = backtest.df[ticker].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        backtest.df['RSI'] = 100 - (100 / (1 + rs))

        #EMA
        short_ema = backtest.df[ticker].ewm(span=12, adjust=False).mean()
        long_ema = backtest.df[ticker].ewm(span=26, adjust=False).mean()

        #MACD
        backtest.df['MACD'] = short_ema - long_ema
        backtest.df['Signal_Line'] = backtest.df['MACD'].ewm(span=9, adjust=False).mean()
        backtest.df['MACD_diff'] = backtest.df['MACD'] - backtest.df['Signal_Line']

        #Bollinger Band
        backtest.df['Middle_Band'] = backtest.df[ticker].rolling(window=20).mean()
        std = backtest.df[ticker].rolling(window=20).std()
        backtest.df['Upper_Band'] = backtest.df['Middle_Band'] + (2 * std)
        backtest.df['Lower_Band'] = backtest.df['Middle_Band'] - (2 * std)
        backtest.df['band_width'] = (backtest.df['Upper_Band'] - backtest.df['Lower_Band']) / backtest.df['Middle_Band'] 
        rolling_window = 60
        backtest.df['bb_width_mean'] = backtest.df['band_width'].rolling(window=rolling_window).mean()

        #Drop NaN
        backtest.df.dropna(inplace=True)

    def next(self, ticker, backtest, index):
        current_index = backtest.df.index.get_loc(index)
        if current_index < self.long_window:  
            return
        
        #Take pervious data
        if current_index > 0:
            last_index = backtest.df.index[current_index - 1]
            last2_index = backtest.df.index[current_index - 2]
            backtest.df.loc[index, ticker + '_holding_position'] = backtest.df.loc[last_index, ticker + '_holding_position']
            backtest.df.loc[index, ticker + '_holding_market_value'] = backtest.df.loc[index, ticker + '_holding_position'] * backtest.df.loc[index, ticker]
            backtest.df.loc[index, 'Cash'] = backtest.df.loc[last_index, 'Cash']
            backtest.df.loc[index, 'Total_equity'] = backtest.df.loc[last_index, 'Total_equity']
            backtest.df.loc[index, ticker +'_average_cost']  = backtest.df.loc[last_index, ticker +'_average_cost']

        #Position 
        stock_price = backtest.df.loc[index, ticker]
        stock_open = backtest.df.loc[index, 'Open']
        total_equity = backtest.df.loc[index, 'Total_equity']
        volume = backtest.df.loc[index,'Volume']
        holding_position = backtest.df.loc[index, ticker + '_holding_position'] 
        cash = backtest.df.loc[index, 'Cash']
        volume = backtest.df.loc[index,'Volume']
        buy_qty = int(cash/stock_price)
        average_volume = backtest.df.loc[index,'Average_Volume']


        #TA indicator
        rsi = backtest.df.loc[last_index,'RSI']
        MACD =  backtest.df.loc[last_index,'MACD']
        MACD_singal_line =  backtest.df.loc[last_index,'Signal_Line']
        macd_diff =  backtest.df.loc[last_index,'MACD_diff']
        last_macd_diff =  backtest.df.loc[last2_index,'MACD_diff']
        bb_mid = backtest.df.loc[last_index, 'Middle_Band']
        bb_upper = backtest.df.loc[last_index, 'Upper_Band']
        bb_lower = backtest.df.loc[last_index, 'Lower_Band']
        bb_width = (bb_upper - bb_lower) / bb_mid 
        bb_width_rolling_mean =  backtest.df['bb_width_mean']
        ADX = backtest.df.loc[last_index, 'ADX_14']
        k_stochastic = backtest.df.loc[last_index, 'STOCHk_14_3_3']
        d_stochastic = backtest.df.loc[last_index, 'STOCHk_14_3_3']
        k_d_stochastic_diff = backtest.df.loc[last_index, 'STOCHk_14_3_3'] - backtest.df.loc[last_index, 'STOCHk_14_3_3']
        supertrend = backtest.df.loc[index,'SuperTrend_Direction']
        last_supertrend = backtest.df.loc[last_index,'SuperTrend_Direction']
        ema_10 = backtest.df.loc[last_index, 'S_EWA']
        long_mavg = backtest.df.loc[last_index,'L_EWA']




        # Trading condition
        buy_condition_1 = backtest.df.loc[last_index, 'Cash'] >= stock_price * 1
        sell_condition_1 = backtest.df.loc[index, ticker + '_holding_position']  > 0
        stop_lose_condition_1 = stock_price <= backtest.df.loc[index, ticker +'_average_cost']*0.9
        stop_short_condition_1 = backtest.df.loc[index, ticker + '_holding_market_value']* -1 >( backtest.df.loc[index, 'Cash']/2)*1.3

        # Trending trading condition
        trending_buy_condition_1 = macd_diff > 0 and last_macd_diff <= 0 and rsi < 80
        trending_buy_condition_2 = stock_price >= bb_mid
        trending_buy_condition_3 = supertrend == -1 and volume > average_volume * 1.5
        trending_sell_condition_1 = macd_diff < 0 and last_macd_diff >= 0 and rsi > 70
        trending_sell_condition_2 = stock_price >= bb_upper* 0.95

        # Consolidation trading condition
        consolidation_buy_condition_1 = rsi < 50 and k_stochastic < 40
        consolidation_buy_condition_2 = stock_price <= bb_mid
        consolidation_sell_condition_1 = rsi > 60
        consolidation_sell_condition_2 = stock_price >= bb_mid

        
        #Identify market trend
        trending_market = True

        dynamic_threshold = bb_width_rolling_mean 

        if bb_width > dynamic_threshold.loc[last_index] and ADX > 15:
            trending_market = True
        else:
            trending_market = False

        if trending_market:
            if  (buy_condition_1 and trending_buy_condition_1 and trending_buy_condition_2) or (buy_condition_1 and trending_buy_condition_3):
                if backtest.df.loc[index, ticker + '_holding_position'] + buy_qty != 0:
                    backtest.df.loc[index, ticker + '_average_cost'] = (
                    backtest.df.loc[index, ticker + '_average_cost'] * backtest.df.loc[index, ticker + '_holding_position'] + 
                    buy_qty * stock_open) / (backtest.df.loc[index, ticker + '_holding_position'] + buy_qty)
                backtest.df.loc[index, ticker + '_holding_position'] += buy_qty
                backtest.df.loc[index, ticker +'_action_signal'] = 1
            

            if (sell_condition_1 and trending_sell_condition_1 and trending_sell_condition_2):
                backtest.df.loc[index, ticker + '_holding_position'] -= holding_position
                backtest.df.loc[index, ticker +'_action_signal'] = -1
        
        else: #consolidation Market

            if  (buy_condition_1 and consolidation_buy_condition_1 and consolidation_buy_condition_2):
                if backtest.df.loc[index, ticker + '_holding_position'] + buy_qty != 0:
                    backtest.df.loc[index, ticker + '_average_cost'] = (
                    backtest.df.loc[index, ticker + '_average_cost'] * backtest.df.loc[index, ticker + '_holding_position'] + 
                    buy_qty * stock_open) / (backtest.df.loc[index, ticker + '_holding_position'] + buy_qty)
                backtest.df.loc[index, ticker + '_holding_position'] += buy_qty
                backtest.df.loc[index, ticker +'_action_signal'] = 1

            if (sell_condition_1 and consolidation_sell_condition_1 and consolidation_sell_condition_2):
                backtest.df.loc[index, ticker + '_holding_position'] -= holding_position
                backtest.df.loc[index, ticker +'_action_signal'] = -1

        #Equity & cash moving
        position_moving =   backtest.df.loc[index, ticker + '_holding_position'] - backtest.df.loc[last_index, ticker + '_holding_position']
        position_moving_value = position_moving*backtest.df.loc[index, 'Open']
        position_value = backtest.df.loc[index, ticker + '_holding_position']*backtest.df.loc[index, 'Open']
        if position_moving != 0:
            backtest.df.loc[index, 'Cash'] -= np.sign(position_moving) * abs(position_moving_value)
        backtest.df.loc[index, 'Total_equity'] = backtest.df.loc[index, 'Cash'] + position_value


ticker = 'NVDA'
start_day = "2024-01-01"
end_day = "2025-03-20"
stock_df = yf.download(ticker, start=start_day, end=end_day)
stock_df.columns = [col[0] for col in stock_df.columns]

#ADX
stock_df.ta.adx(length=14, append=True)

#Stochastic
stock_df.ta.stoch(append=True)

#supertrend
supertrend = ta.supertrend(stock_df['High'], stock_df['Low'], stock_df['Close'], length=10, multiplier=3)
stock_df['SuperTrend'] = supertrend['SUPERT_10_3.0']  # SuperTrend 線數值
stock_df['SuperTrend_Direction'] = supertrend['SUPERTd_10_3.0']

#set up best_sharpe
best_sharpe = float('-inf')

ratios = np.arange(20, 40, 5)
     
for i in ratios:
            
        backtest = Backtest(initial_cash=1000)
            
        backtest.add_data(ticker, stock_df)
            
        backtest.add_strategy(Strategy, volume_window=i)
            
        backtest.run()
            
        backtest.calculate_return()
            
        sharpe = backtest.analyse_tool.sharpe_ratio()
        max_drawdown = backtest.analyse_tool.maximum_drawdown()
        
        if sharpe:
            print(f'Stock:{ticker}')
            print(f'Volume window:{i}')
            print(f'SR:{sharpe}')
            print(f'MMD:{max_drawdown}')
            
        test_df = backtest.df
        test_df.to_csv('test2.csv')
            
        if sharpe > best_sharpe:
            best_stock = ticker
            best_sharpe = sharpe
            best_ratio = i
            best_mmd = max_drawdown
            optimized_df = backtest.df


print(best_stock, best_ratio, best_sharpe, best_mmd)

optimized_df.to_csv('backtest_result.csv')


fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
plt.subplots_adjust(hspace=0.1)

# Price (main)
ax_price.plot(optimized_df.index, optimized_df[ticker], label=f'{ticker} Price', color='orange')
ax_price.plot(optimized_df.index, optimized_df['Middle_Band'], label='Middle Band', color='blue', linestyle='--')
ax_price.plot(optimized_df.index, optimized_df['Upper_Band'], label='Upper Band', color='green', linestyle='--')
ax_price.plot(optimized_df.index, optimized_df['Lower_Band'], label='Lower Band', color='red', linestyle='--')
# Buy Sell Singal
buy_signals = np.where(optimized_df[f'{ticker}_action_signal'] == 1, optimized_df[ticker], np.nan)
sell_signals = np.where(optimized_df[f'{ticker}_action_signal'] == -1, optimized_df[ticker], np.nan)
ax_price.scatter(optimized_df.index[~np.isnan(buy_signals)], buy_signals[~np.isnan(buy_signals)],
                 marker='^', s=75, color='g', label='Buy')
ax_price.scatter(optimized_df.index[~np.isnan(sell_signals)], sell_signals[~np.isnan(sell_signals)],
                 marker='v', s=75, color='r', label='Sell')
ax_price.set_title(f"{ticker} Backtest")
ax_price.legend(loc='upper left')

# RSI 
ax_rsi.plot(optimized_df.index, optimized_df['RSI'], label='RSI', color='purple')
ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought')
ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold')
ax_rsi.set_ylabel('RSI')
ax_rsi.legend(loc='upper left')

# MACD 
ax_macd.plot(optimized_df.index, optimized_df['MACD'], label='MACD', color='blue')
ax_macd.plot(optimized_df.index, optimized_df['Signal_Line'], label='Signal Line', color='red')
ax_macd.bar(optimized_df.index, optimized_df['MACD_diff'], label='MACD Diff', 
            color='gray', alpha=0.3)
ax_macd.set_ylabel('MACD')
ax_macd.legend(loc='upper left')

ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
fig.autofmt_xdate()

vline_ax1 = ax_price.axvline(x=optimized_df.index[0], color='k', linestyle='--', alpha=0.5, visible=False)
vline_ax2 = ax_rsi.axvline(x=optimized_df.index[0], color='k', linestyle='--', alpha=0.5, visible=False)
vline_ax3 = ax_macd.axvline(x=optimized_df.index[0], color='k', linestyle='--', alpha=0.5, visible=False)

date_annotation = ax_price.text(0.98, 0.90, "", transform=ax_price.transAxes,
                             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

def on_mouse_move(event):

    if event.inaxes in [ax_price, ax_rsi, ax_macd] and event.xdata is not None:
        xdata = event.xdata
        cur_date = mdates.num2date(xdata) 
        date_str = cur_date.strftime("%Y-%m-%d")


        dt_index = pd.to_datetime(optimized_df.index)
        dates_num = mdates.date2num(dt_index.to_pydatetime())
        
        idx = (np.abs(dates_num - xdata)).argmin()
        nearest_date = optimized_df.index[idx]

        adx_value = optimized_df.iloc[idx]['ADX_14']
        bb_with_mean_value = optimized_df.iloc[idx]['bb_width_mean']
        bb_width_value = optimized_df.iloc[idx]['band_width']


        date_annotation.set_text(
            f"Date: {date_str}\nADX: {adx_value:.2f}\nBB Width Mean: {bb_with_mean_value:.2f}\nBB Width: {bb_width_value:.2f}"
        )


        for vline in [vline_ax1, vline_ax2, vline_ax3]:
            vline.set_xdata([xdata, xdata])
            vline.set_visible(True)
        fig.canvas.draw_idle()
    else:

        for vline in [vline_ax1, vline_ax2, vline_ax3]:
            vline.set_visible(False)
        date_annotation.set_text("")
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.show()
