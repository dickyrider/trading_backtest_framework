import pandas as pd
import pandas_ta as p_ta
import yfinance as yf
import numpy as np
import ta
from backtest import framework 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta.volatility import AverageTrueRange
from itertools import groupby
from operator import itemgetter
import talib


class Strategy:
    def __init__(self, data, **kwargs):
        self.data = data
        self.short_window = 10
        self.long_window = 25
        self.test_window = kwargs.get('test_window', 1) 
        self.contract_size = kwargs.get('contract_size', 0.001)
        self.multiplier  = kwargs.get('multiplier', 10)
        self.maintenance_margin_rate = kwargs.get('maintenance_margin_rate', 0.05)

    def add_reference_data(self, framework, ticker):
        framework.df['Maintenance_Margin'] = 0
        framework.df['force_close_out'] = 0
        framework.df['trend_strength'] = 0
        framework.df['trend'] = 0
        framework.df['trading_strategy'] = 0

        # EMA
        framework.df['EMA_24'] = framework.df[ticker].ewm(span=24, adjust=False).mean()
        framework.df['EMA_72'] = framework.df[ticker].ewm(span=72, adjust=False).mean()

        # RSI
        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        framework.df['RSI'] = compute_rsi(framework.df[ticker], period=14)
        framework.df['RSI_MA24'] = framework.df['RSI'].rolling(window=24).mean()



        # atr
        atr = AverageTrueRange(high=framework.df['High'], low=framework.df['Low'], close=framework.df[ticker], window=14)
        framework.df['ATR_14'] = atr.average_true_range()

        #MACD
        macd = ta.trend.MACD(framework.df[ticker])
        framework.df['macd'] = macd.macd()
        framework.df['macd_signal'] = macd.macd_signal()
        framework.df['macd_hist'] = macd.macd_diff()

        #Bollinger Band
        framework.df['Middle_Band'] = framework.df[ticker].rolling(window=20).mean()
        std = framework.df[ticker].rolling(window=20).std()
        framework.df['Upper_Band'] = framework.df['Middle_Band'] + (2 * std)
        framework.df['Lower_Band'] = framework.df['Middle_Band'] - (2 * std)

        #ADX
        adx = p_ta.adx(high=framework.df['High'], low=framework.df['Low'], close=framework.df[ticker], length=14)
        framework.df['ADX'] = adx['ADX_14']
        framework.df['+DI'] = adx['DMP_14']
        framework.df['-DI'] = adx['DMN_14']

        #KDJ
        stoch = ta.momentum.StochasticOscillator(
            high=framework.df["High"],
            low=framework.df["Low"],
            close=framework.df[ticker],
            window=28,
            smooth_window=3
        )

        framework.df["K"] = stoch.stoch()
        framework.df["D"] = stoch.stoch_signal()
        framework.df["J"] = 3 * framework.df["K"] - 2 * framework.df["D"]

        #Drop NaN
        framework.df.dropna(inplace=True)


    def next(self, ticker, framework, index):
        current_index = framework.df.index.get_loc(index)
        if current_index < 2:  #ignore first index 
            return
        

        last_index = framework.df.index[current_index - 1]
        last2_index = framework.df.index[current_index - 2]
        last3_index = framework.df.index[current_index - 3]
        last10_index = framework.df.index[current_index - 10]
        last14_index = framework.df.index[current_index - 14]

        
        # periouvs data
        framework.df.loc[index, 'Maintenance_Margin'] = framework.df.loc[last_index, 'Maintenance_Margin']
        framework.df.loc[index, ticker + '_holding_position'] = framework.df.loc[last_index, ticker + '_holding_position']
        framework.df.loc[index, ticker + '_initial_margin'] = framework.df.loc[last_index, ticker + '_initial_margin']
        framework.df.loc[index, ticker + '_long_trade'] = framework.df.loc[last_index, ticker + '_long_trade']
        framework.df.loc[index, ticker + '_long_win'] = framework.df.loc[last_index, ticker + '_long_win']
        framework.df.loc[index, ticker + '_long_win_rate'] = framework.df.loc[last_index, ticker + '_long_win_rate']
        framework.df.loc[index, ticker + '_short_trade'] = framework.df.loc[last_index, ticker + '_short_trade']
        framework.df.loc[index, ticker + '_short_win'] = framework.df.loc[last_index, ticker + '_short_win']
        framework.df.loc[index, ticker + '_short_win_rate'] = framework.df.loc[last_index, ticker + '_short_win_rate']
        if framework.df.loc[last_index, ticker + '_holding_position'] != 0:
            framework.df.loc[index, ticker + '_average_cost'] = framework.df.loc[last_index, ticker + '_average_cost']
        else:
            framework.df.loc[index, ticker + '_average_cost'] = 0
        framework.df.loc[index, 'Cash'] = framework.df.loc[last_index, 'Cash']
        framework.df.loc[index, 'Total_equity'] = framework.df.loc[last_index, 'Total_equity']
        if framework.df.loc[index, ticker + '_holding_position'] != 0:
            framework.df.loc[index, 'trading_strategy'] = framework.df.loc[last_index, 'trading_strategy']


        
        # current market data
        price = framework.df.loc[last_index, ticker]
        last_price = framework.df.loc[last2_index, ticker]
        prev_10_price = framework.df.loc[last10_index:last_index, ticker]
        open_price = framework.df.loc[index, 'Open']
        cash = framework.df.loc[index, 'Cash']
        contract_cost = open_price * self.contract_size 
        initial_margin = contract_cost/self.multiplier
        average_cost = framework.df.loc[last_index, ticker + '_average_cost']
        strategy = framework.df.loc[index, 'trading_strategy']
        last_trend = framework.df.loc[last2_index, 'trend']
        trend = framework.df.loc[last_index, 'trend']

        # position related 


        holding_position = framework.df.loc[index, ticker + '_holding_position']
        
        # TA indicator
        ema10 = framework.df.loc[last_index, 'EMA_24']
        last_ema10 = framework.df.loc[last2_index, 'EMA_24']
        last_ema40 = framework.df.loc[last2_index, 'EMA_72']
        ema40 = framework.df.loc[last_index, 'EMA_72']
        atr = framework.df.loc[last_index, 'ATR_14']
        k_line = framework.df.loc[last_index, 'K']
        d_line = framework.df.loc[last_index, 'D']
        j_line = framework.df.loc[last_index, 'J']
        last_k_line = framework.df.loc[last2_index, 'K']
        last_d_line = framework.df.loc[last2_index, 'D']
        rsi = framework.df.loc[last_index, 'RSI']
        last_rsi = framework.df.loc[last2_index, 'RSI']
        prev_10_rsi = framework.df.loc[last10_index:last_index, 'RSI']
        adx = framework.df.loc[last_index, 'ADX']
        last_adx = framework.df.loc[last2_index, 'ADX'] 
        prev_4_adx = framework.df.loc[last10_index:last_index, 'ADX']
        macd_hist  = framework.df.loc[last_index, 'macd_hist']
        last_macd_hist  = framework.df.loc[last2_index, 'macd_hist']
        bb_top = framework.df.loc[last_index, 'Upper_Band'] 
        bb_mid = framework.df.loc[last_index, 'Middle_Band'] 
        bb_bottom = framework.df.loc[last_index, 'Lower_Band'] 
        obv =  framework.df.loc[last_index, 'OBV'] 
        prev_10_obv = framework.df.loc[last10_index:last_index, 'OBV']
        last_obv = framework.df.loc[last2_index, 'OBV'] 
        obv_ma20 =  framework.df.loc[last_index, 'OBV_MA20'] 
        di_pos = framework.df.loc[last_index, '+DI'] 
        last_di_pos = framework.df.loc[last2_index, '+DI'] 
        di_neg = framework.df.loc[last_index, '-DI']
        last_di_neg = framework.df.loc[last2_index, '-DI'] 
        di_diff = abs(framework.df.loc[last_index, '+DI'] - framework.df.loc[last_index, '-DI'])
        last_di_diff = abs(framework.df.loc[last2_index, '+DI'] - framework.df.loc[last2_index, '-DI'])


        # trade quantity 
        buy_qty =int(((cash/2)-1)/initial_margin)




        #Trading condition
        #Open  
        basic_open_condition = framework.df.loc[index, 'Cash'] > initial_margin*(buy_qty+abs(holding_position))
        #long 
        


        # short 
        basic_close_condition = framework.df.loc[index, ticker + '_holding_position'] != 0


        # close 

        # force close
        force_close_condition = holding_position != 0 and framework.df.loc[index, 'Total_equity']  <=  price*self.contract_size*holding_position*self.maintenance_margin_rate

        # stop profit or loss
        stop_profit_condition = False
        stop_loss_condition = False



        atr = min((average_cost*0.015),atr)


        stop_profit_thersold = 2
        stop_loss_thersold = 1.3

        if adx > 40:
            stop_profit_thersold = 3
            stop_loss_thersold = 1
            atr = min((average_cost*0.02),atr)
        

        if holding_position > 0:
            stop_profit_condition = price >= average_cost + (stop_profit_thersold * atr)
            stop_loss_condition = price <= average_cost - (stop_loss_thersold * atr) 
        elif holding_position < 0:
            stop_profit_condition = price <= average_cost - (stop_profit_thersold * atr) 
            stop_loss_condition = price >= average_cost + (stop_loss_thersold * atr )

        trend_change_close_condition = (last_trend != -1 and trend == -1) or (last_trend == -1 and trend == 1) 

                               



        #Trend identify
        # strength


        if adx > 25 and adx < 40:
            framework.df.loc[index, 'trend_strength'] = 1
        elif adx > 40:
            framework.df.loc[index, 'trend_strength'] = 2

        if ema40 < ema10  and adx > 25:
            framework.df.loc[index, 'trend'] = 1

        elif  ema10 < ema40 and adx > 25:
            framework.df.loc[index, 'trend'] = -1

        atr_ratio = atr/price

        if atr_ratio > 0.05:
            buy_qty = 0

       #==========================================================================================================
        #Trade
        #==========================================================================================================

        # close position

        if basic_close_condition:
            if framework.df.loc[last_index, ticker + '_holding_position'] > 0: 
                if  framework.df.loc[last_index, 'trend'] != 1 and framework.df.loc[index, 'trend'] == 1  :
                    framework.close_position(index, ticker, open_price, holding_position, action_signal=-1, strategy_type = 'trend change')
                # weak up trend
                elif j_line > 100 or (adx > 25 and di_neg > di_pos):
                    framework.close_position(index, ticker, open_price, holding_position, action_signal=-1, strategy_type = 'close long position')

            if framework.df.loc[last_index, ticker + '_holding_position'] < 0: 
                if j_line < 0 or (adx > 25 and di_neg < di_pos):
                    framework.close_position(index, ticker, open_price, holding_position, action_signal=1, strategy_type = 'close short position')


        if force_close_condition :
            if framework.df.loc[last_index, ticker + '_holding_position'] > 0: 
                framework.close_position(index, ticker, open_price, holding_position, action_signal=-1, strategy_type = 'force close out')
            if framework.df.loc[last_index, ticker + '_holding_position'] < 0: 
                framework.close_position(index, ticker, open_price, holding_position, action_signal=1, strategy_type = 'force close out')

        if stop_profit_condition :
            if framework.df.loc[last_index, ticker + '_holding_position'] > 0 and framework.df.loc[index, ticker + '_holding_position'] > 0: 
                framework.close_position(index, ticker, open_price, holding_position, action_signal=-1, strategy_type = 'stop profit')
            if framework.df.loc[last_index, ticker + '_holding_position'] < 0 and framework.df.loc[index, ticker + '_holding_position'] < 0: 
                framework.close_position(index, ticker, open_price, holding_position, action_signal=1, strategy_type = 'stop profit')
        
        if stop_loss_condition:
            if framework.df.loc[last_index, ticker + '_holding_position'] > 0 and framework.df.loc[index, ticker + '_holding_position'] > 0: 
                framework.close_position(index, ticker, open_price, holding_position, action_signal=-1, strategy_type = 'stop loss')
            if framework.df.loc[last_index, ticker + '_holding_position'] < 0 and framework.df.loc[index, ticker + '_holding_position'] < 0: 
                framework.close_position(index, ticker, open_price, holding_position, action_signal=1, strategy_type = 'stop loss')


        #open position
        if basic_open_condition :
            # weak up trend 
            if framework.df.loc[index, 'trend_strength'] == 0 and j_line < 20 and ema10 > ema40:
                framework.open_position(index, ticker, open_price, buy_qty, direction='long',strategy_type = 'K D cross up long')

            # weak down trend  
            if framework.df.loc[index, 'trend_strength'] == 0 and j_line > 50 and ema10 < ema40 :
                framework.open_position(index, ticker, open_price, buy_qty, direction='short',strategy_type = 'kdj down short')


        #==========================================================================================================
        #Equity & cash moving
        #==========================================================================================================        
        #Maintenance Margin
        framework.df['Maintenance_Margin'] = (framework.df.loc[index, ticker + '_holding_position']* self.contract_size * framework.df.loc[index, ticker + '_average_cost'] *self.maintenance_margin_rate ) 
        
        framework.update_position_info(index, ticker)

        #Equity
        framework.df.loc[index, 'Total_equity'] = framework.df.loc[index, 'Cash'] + framework.df.loc[index, ticker +'_holding_PnL'] + framework.df.loc[index, ticker + '_initial_margin']



ticker = 'BTC'
start_day = "2024-12-01"
end_day = "2025-03-26"
stock_df = pd.read_csv('btc_1hour_data.csv')
stock_df.set_index('Date', inplace=True)  


#set up best_sharpe
best_sharpe = float('-inf')

ratios = np.arange(5, 10, 5)
     
for i in ratios:
            
        fw = framework(initial_cash=10000)
            
        fw.add_data(ticker, stock_df, contract_size = 0.001, multiplier = 10)
            
        fw.add_strategy(Strategy, test_window=i)
            
        fw.run()
            
        fw.calculate_return()
            
        sharpe = fw.analyse_tool.sharpe_ratio(periods = 8760)
        max_drawdown = fw.analyse_tool.maximum_drawdown()
        
            
        if sharpe >= best_sharpe:
            best_stock = ticker
            best_sharpe = sharpe
            best_ratio = i
            best_mmd = max_drawdown
            optimized_df = fw.df
            total_win_rate = (optimized_df[f'{ticker}_long_win'].iloc[-1] + optimized_df[f'{ticker}_short_win'].iloc[-1])/ (optimized_df[f'{ticker}_long_trade'].iloc[-1] + optimized_df[f'{ticker}_short_trade'].iloc[-1])
            print(f'Stock:{ticker}')
            print(f'Volume window:{i}')
            print(f'SR:{sharpe}')
            print(f'MMD:{max_drawdown}')
            print(f"long trades: {optimized_df[f'{ticker}_long_trade'].iloc[-1]}")
            print(f"long win rate: {(optimized_df[f'{ticker}_long_win_rate'].iloc[-1])*100}%")
            print(f"short trades: {optimized_df[f'{ticker}_short_trade'].iloc[-1]}")
            print(f"short win rate: {(optimized_df[f'{ticker}_short_win_rate'].iloc[-1])*100}%")
            print(f"total win rate: {total_win_rate*100}%")

optimized_df.to_csv('framework_result.csv')

optimized_df.index = pd.to_datetime(optimized_df.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')

if optimized_df.index.isna().any():
    print("Warning: Some index values could not be converted to datetime. Check the data:")
    print(optimized_df.index[optimized_df.index.isna()])


fig, (ax_price, ax_rsi, ax_di, ax_kdj) = plt.subplots(
    4, 1,
    figsize=(16, 14),
    sharex=True,
    gridspec_kw={'height_ratios': [2.5, 1, 1, 1]}  # 放大價格圖
)


# === 1. Price + EMA 24/72 ===
ax_price.plot(optimized_df.index, optimized_df['BTC'], label='Price', color='black', linewidth=1.2)
ax_price.plot(optimized_df.index, optimized_df['Upper_Band'], label='Upper Band', color='blue', linestyle='--', linewidth=0.8)
ax_price.plot(optimized_df.index, optimized_df['Middle_Band'], label='Mid band', color='grey', linestyle='--', linewidth=0.8)
ax_price.plot(optimized_df.index, optimized_df['Lower_Band'], label='Lower Band', color='red', linestyle='--', linewidth=0.8)
ax_price.plot(optimized_df.index, optimized_df['EMA_24'], label='EMA 24', color='orange', linewidth=1.3)
ax_price.plot(optimized_df.index, optimized_df['EMA_72'], label='EMA 72', color='green', linewidth=1.3)


# Buy/Sell signals
position = optimized_df['BTC_holding_position']
signal = optimized_df['BTC_action_signal']
price = optimized_df['BTC']

# Long Entry
long_entry_index = optimized_df[(signal == 1) & (position > 0)].index
long_entry_price = price.loc[long_entry_index]
ax_price.scatter(long_entry_index, long_entry_price, marker='^', s=75, color='green', label='Long Entry')

# Short Entry
short_entry_index = optimized_df[(signal == -1) & (position < 0)].index
short_entry_price = price.loc[short_entry_index]
ax_price.scatter(short_entry_index, short_entry_price, marker='v', s=75, color='red', label='Short Entry')

# Long Exit
long_exit_index = optimized_df[(signal == -1) & (position == 0)].index
long_exit_price = price.loc[long_exit_index]
ax_price.scatter(long_exit_index, long_exit_price, marker='o', s=60, color='lime', facecolors='none', label='Long Exit', linewidths=1.5)

# Short Exit
short_exit_index = optimized_df[(signal == 1) & (position == 0)].index
short_exit_price = price.loc[short_exit_index]
ax_price.scatter(short_exit_index, short_exit_price, marker='x', s=60, color='darkred', label='Short Exit')

ax_price.set_title("BTC Price + EMA 5/10/40")
ax_price.legend(loc='upper left')
ax_price.grid(True)

# === ADX > 25 area ===
adx_condition_long = (optimized_df['ADX'] > 25) & (optimized_df['+DI'] > optimized_df['-DI'])
idx_adx_long = np.where(adx_condition_long)[0]
for _, g in groupby(enumerate(idx_adx_long), lambda i: i[0] - i[1]):
    group = list(map(itemgetter(1), g))
    start = optimized_df.index[group[0]]
    end = optimized_df.index[group[-1]]
    ax_price.axvspan(start, end, color='darkgreen', alpha=0.3)

adx_condition_short = (optimized_df['ADX'] > 25) & (optimized_df['+DI'] < optimized_df['-DI'])
idx_adx_short = np.where(adx_condition_short)[0]
for _, g in groupby(enumerate(idx_adx_short), lambda i: i[0] - i[1]):
    group = list(map(itemgetter(1), g))
    start = optimized_df.index[group[0]]
    end = optimized_df.index[group[-1]]
    ax_price.axvspan(start, end, color='darkred', alpha=0.3)

# === 2. RSI + RSI MA ===
ax_rsi.plot(optimized_df.index, optimized_df['RSI'], label='RSI', color='purple')
ax_rsi.plot(optimized_df.index, optimized_df['RSI_upper'], color='red', linestyle='--', label='Overbought')
ax_rsi.plot(optimized_df.index, optimized_df['RSI_lower'], color='green', linestyle='--', label='Oversold')
ax_rsi.set_ylabel('RSI')
ax_rsi.legend(loc='upper left')
ax_rsi.grid(True)

# === 3. +DI / -DI ===
ax_di.plot(optimized_df.index, optimized_df['+DI'], label='+DI', color='green')
ax_di.plot(optimized_df.index, optimized_df['-DI'], label='-DI', color='red')
ax_di.set_ylabel('Directional Index')
ax_di.legend(loc='upper left')
ax_di.grid(True)

# === 4. OBV ===
jmin = optimized_df["J"].min()
jmax = optimized_df["J"].max()
jmid = (jmin + jmax) / 2
half_range = (jmax - jmin) / 2

macdmin = optimized_df["macd_hist"].min()
macdmax = optimized_df["macd_hist"].max()
macd_absmax = max(abs(macdmax), abs(macdmin))

if macd_absmax == 0:
    macd_hist_scaled = np.zeros_like(optimized_df["macd_hist"])
else:
    macd_hist_scaled = optimized_df["macd_hist"] / macd_absmax * half_range

macd_up = macd_hist_scaled.copy()
macd_up[macd_up < 0] = 0
macd_down = macd_hist_scaled.copy()
macd_down[macd_down > 0] = 0

ax_kdj.plot(optimized_df.index, optimized_df["K"], label="K", color="blue")
ax_kdj.plot(optimized_df.index, optimized_df["D"], label="D", color="orange")
ax_kdj.plot(optimized_df.index, optimized_df["J"], label="J", color="green")
ax_kdj.axhline(80, color="red", linestyle="--", linewidth=0.8)
ax_kdj.axhline(20, color="green", linestyle="--", linewidth=0.8)
ax_kdj.set_title("Stochastic Oscillator (KDJ) & MACD Histogram (Scaled, Centered)")
ax_kdj.set_ylabel("K / D / J")
ax_kdj.set_ylim(jmin - 10, jmax + 10)
ax_kdj.grid(True)

ax_kdj.fill_between(
    optimized_df.index, jmid, jmid + macd_up,
    facecolor="green", alpha=0.7, label="MACD Histogram (Up, scaled)"
)
ax_kdj.fill_between(
    optimized_df.index, jmid, jmid + macd_down,
    facecolor="red", alpha=0.7, label="MACD Histogram (Down, scaled)"
)

ax_kdj.legend(loc='upper left')

ax_kdj.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
fig.autofmt_xdate()

for ax in [ax_price, ax_rsi, ax_di, ax_kdj]:
    ax.tick_params(axis='x', labelsize=7)

# === tooltip ===
vline_price = ax_price.axvline(x=optimized_df.index[0], color='k', linestyle='--', alpha=0.5, visible=False)
vline_rsi = ax_rsi.axvline(x=optimized_df.index[0], color='k', linestyle='--', alpha=0.5, visible=False)
vline_di = ax_di.axvline(x=optimized_df.index[0], color='k', linestyle='--', alpha=0.5, visible=False)
vline_obv = ax_kdj.axvline(x=optimized_df.index[0], color='k', linestyle='--', alpha=0.5, visible=False)
hline_price = ax_price.axhline(y=optimized_df['BTC'].iloc[0], color='magenta', linestyle='-.', alpha=0.7, visible=False)
hline_rsi = ax_rsi.axhline(y=optimized_df['RSI'].iloc[0], color='magenta', linestyle='-.', alpha=0.7, visible=False)
hline_obv = ax_kdj.axhline(y=optimized_df['J'].iloc[0], color='magenta', linestyle='-.', alpha=0.7, visible=False)

date_annotation = ax_price.text(
    0.98, 0.90, "", transform=ax_price.transAxes,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    ha='right', va='top'
)

def on_mouse_move(event):
    if event.inaxes in [ax_price, ax_rsi, ax_di, ax_kdj] and event.xdata is not None:
        xdata = event.xdata
        cur_date = mdates.num2date(xdata)
        date_str = cur_date.strftime("%Y-%m-%d %H:%M")

        dt_index = pd.to_datetime(optimized_df.index)
        dates_num = mdates.date2num(dt_index.to_pydatetime())
        idx = (np.abs(dates_num - xdata)).argmin()

        # 抓取資料
        row = optimized_df.iloc[idx]
        price = row['BTC']
        rsi = row['RSI']
        adx = row['ADX']
        obv = row['OBV']
        di_pos = row['+DI']
        di_neg = row['-DI']
        trading_strategy = row['trading_strategy']
        ema10 = row['EMA_24']
        ema40 = row['EMA_72']
        atr  = row['ATR_14']
        slope = row['weighted_slope']
        long_win_rate = row[f'{ticker}_long_win_rate']
        short_win_rate = row[f'{ticker}_short_win_rate']
        long_trade = row[f'{ticker}_long_trade']
        short_trade = row[f'{ticker}_short_trade']
        long_win = row[f'{ticker}_long_win']
        short_win = row[f'{ticker}_short_win']
        if long_trade + short_trade > 0:
            total_win_rate = (long_win + short_win) / (short_trade + long_trade)
        else:
            total_win_rate = 0.0

        date_annotation.set_text(
            f"Date: {date_str}\n"
            f"Price: {price:.2f}\n"
            f"EMA 10: {ema10:.2f} | EMA 40: {ema40:.2f}\n"
            f"RSI: {rsi:.2f}\n"
            f"ADX: {adx:.2f}\n"
            f"ATR: {atr:.2f}\n" 
            f"+DI: {di_pos:.2f} / -DI: {di_neg:.2f}\n"
            f"Slope: {slope:.0f} \n"
            f"long: {long_win_rate:.2f} |short: {short_win_rate:.2f} | total: {total_win_rate:.2f}\n"
            f"strategy: {trading_strategy}"
        )

        for v in [vline_price, vline_rsi, vline_di, vline_obv]:
            v.set_xdata([xdata])
            v.set_visible(True)
        hline_price.set_ydata([price])
        hline_price.set_visible(True)
        hline_rsi.set_ydata([rsi])
        hline_rsi.set_visible(True)
        hline_obv.set_ydata([row['J']])
        hline_obv.set_visible(True)

        fig.canvas.draw_idle()
    else:
        for v in [vline_price, vline_rsi, vline_di, vline_obv]:
            v.set_visible(False)
        hline_price.set_visible(False)
        hline_rsi.set_visible(False)
        date_annotation.set_text("")
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.subplots_adjust(hspace=0.25, bottom=0.08, top=0.95)
plt.show()