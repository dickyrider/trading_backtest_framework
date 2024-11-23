import pandas as pd
import yfinance as yf
from frame import Backtest 
import matplotlib.pyplot as plt



class SimpleMovingAverageStrategy:
    def __init__(self, data):
        self.data = data
        self.short_window = 5  
        self.long_window = 20  

    def next(self, ticker, backtest, index):
        current_index = backtest.df.index.get_loc(index)
        if current_index < self.long_window:  
            return

        short_mavg = self.data[ticker].iloc[current_index - self.short_window + 1:current_index + 1].mean()
        long_mavg = self.data[ticker].iloc[current_index - self.long_window + 1:current_index + 1].mean()
        
        #Take pervious data
        if current_index > 0:
            index_to_date = backtest.df.index[current_index - 1]
            backtest.df.loc[index, ticker + '_holding_position'] = backtest.df.loc[index_to_date, ticker + '_holding_position']
            backtest.df.loc[index, 'Cash'] = backtest.df.loc[index_to_date, 'Cash']
            backtest.df.loc[index, 'Total_equity'] = backtest.df.loc[index_to_date, 'Total_equity']

        #Position 
        if short_mavg > long_mavg and  backtest.df.loc[index, ticker + '_holding_position'] == 0:
            backtest.df.loc[index, ticker + '_holding_position'] += 1



        elif short_mavg < long_mavg and  backtest.df.loc[index, ticker + '_holding_position'] > 0:
            backtest.df.loc[index, ticker + '_holding_position'] -= 1


        #Equity & cash moving
        position_moving =   backtest.df.loc[index, ticker + '_holding_position'] - backtest.df.loc[index_to_date, ticker + '_holding_position']
        position_moving_value = position_moving*backtest.df.loc[index, ticker]
        position_value = backtest.df.loc[index, ticker + '_holding_position']*backtest.df.loc[index, ticker]
        if position_moving > 0:
            backtest.df.loc[index, 'Cash'] = backtest.df.loc[index, 'Cash'] - position_moving_value
        elif position_moving < 0:
            backtest.df.loc[index, 'Cash'] = backtest.df.loc[index, 'Cash'] + position_moving_value*-1
        backtest.df.loc[index, 'Total_equity'] = backtest.df.loc[index, 'Cash'] + position_value
        





if __name__ == "__main__":

    backtest = Backtest(initial_cash=100)

    ticker = 'AAPL'
    data = yf.download(ticker, start='2019-11-11', end='2024-11-08')
    backtest.add_data(ticker, data['Close'])

    backtest.add_strategy(SimpleMovingAverageStrategy)

    backtest.run()

    test_df = backtest.df
    

