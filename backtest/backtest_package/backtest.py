import pandas as pd
import numpy as np

class framework:
    def __init__(self, initial_cash=0):
        self.df = pd.DataFrame()  # 存储不同股票的数据
        self.df['Total_equity'] = 0
        self.df['Cash'] = 0
        self.strategies = []
        self.results = {}
        self.cash = initial_cash
        self.equity = self.cash
        self.analyse_tool = self.AnalyseTool(self)
        self.ticker_lst = []
    
    def initital_cash(self, initial_cash):
        self.cash  = initial_cash
        self.equity = self.cash

    def add_data(self, dataname, data, contract_size = 1, multiplier = 1):
        self.ticker = dataname
        self.contract_size = contract_size
        self.multiplier = multiplier
        self.df = pd.concat([self.df, data], axis=1)
        self.df = self.df.rename(columns={'Close': dataname})
        self.df['Cash'] = self.df['Cash'].fillna(self.cash)
        self.df['Total_equity'] = self.df['Total_equity'].fillna(self.cash)
        self.df[dataname + '_holding_position'] = 0.0
        self.df[dataname + '_average_cost'] = 0.0
        self.df[dataname + '_action_signal'] = 0.0
        self.df[dataname + '_holding_market_value'] = 0.0
        self.df[dataname + '_holding_PnL'] = 0.0
        self.df[dataname + '_initial_margin'] = 0.0
        self.df[dataname + '_long_trade'] = 0.0
        self.df[dataname + '_long_win'] = 0.0
        self.df[dataname + '_long_win_rate'] = 0.0
        self.df[dataname + '_short_trade'] = 0.0
        self.df[dataname + '_short_win'] = 0.0
        self.df[dataname + '_short_win_rate'] = 0.0
        self.results[dataname] = []  

    def add_strategy(self, strategy_class, **kwargs):
        strategy_instance = strategy_class(self.df, **kwargs) 
        if hasattr(strategy_instance, 'add_reference_data'):
            print("Adding reference data.")
            strategy_instance.add_reference_data(self, self.ticker)
        else:
            print("add_reference_data method not found.")
        self.strategies.append(strategy_instance)


    def open_position(self, index, ticker, open_price, buy_qty, direction, strategy_type=None):
        last_index = self.df.index[self.df.index.get_loc(index) - 1]
        holding_position = self.df.loc[last_index, f'{ticker}_holding_position']
        last_avg_cost = self.df.loc[last_index, f'{ticker}_average_cost']

        if pd.isna(last_avg_cost) or holding_position == 0:
            new_avg_cost = open_price
        else:
            new_avg_cost = (
                last_avg_cost * abs(holding_position) + open_price * buy_qty
            ) / (abs(holding_position) + buy_qty)
        
        initial_margin = (open_price*self.contract_size * buy_qty)/self.multiplier

        if direction == 'long':
            self.df.loc[index, f'{ticker}_holding_position'] = holding_position + buy_qty
            self.df.loc[index, f'{ticker}_action_signal'] = 1
            self.df.loc[index, f'{ticker}_long_trade'] += 1
            self.df.loc[index, 'Cash'] -= initial_margin
            self.df.loc[index, f'{ticker}_initial_margin'] += initial_margin

        elif direction == 'short' :
            self.df.loc[index, f'{ticker}_holding_position'] = holding_position - buy_qty
            self.df.loc[index, f'{ticker}_action_signal'] = -1
            self.df.loc[index, f'{ticker}_short_trade'] += 1
            self.df.loc[index, 'Cash'] -= initial_margin
            self.df.loc[index, f'{ticker}_initial_margin'] += initial_margin

        self.df.loc[index, f'{ticker}_average_cost'] = new_avg_cost

        if strategy_type is not None:
            self.df.loc[index, 'trading_strategy'] = strategy_type

    def close_position(self, index, ticker, price, buy_qty, action_signal, strategy_type=None):
        last_index = self.df.index[self.df.index.get_loc(index) - 1]
        holding_position = self.df.loc[last_index, f'{ticker}_holding_position']
        realised_pnl = 0
        initial_margin_pre_unit = self.df.loc[index, f'{ticker}_initial_margin']/abs(holding_position)
        if holding_position == 0:
            self.df.loc[index, f'{ticker}_average_cost'] = 0
        if buy_qty > holding_position:
            self.df.loc[index, f'{ticker}_holding_position'] = 0.0
        else:
            self.df.loc[index, f'{ticker}_holding_position'] = holding_position - buy_qty

        if action_signal == 1:
            buy_qty = -buy_qty
            realised_pnl = (self.df.loc[index, f'{ticker}_average_cost'] - price)*self.contract_size
            cash_delta = (realised_pnl + initial_margin_pre_unit) * abs(buy_qty)
            self.df.loc[index, 'Cash'] += cash_delta
            self.df.loc[index, f'{ticker}_initial_margin'] -= initial_margin_pre_unit* abs(buy_qty)
            if realised_pnl > 0:
                self.df.loc[index, f'{ticker}_short_win'] += 1

        elif action_signal == -1:
            realised_pnl = (price - self.df.loc[index, f'{ticker}_average_cost'])*self.contract_size
            cash_delta = (realised_pnl + initial_margin_pre_unit) * abs(buy_qty)
            self.df.loc[index, 'Cash'] += cash_delta
            self.df.loc[index, f'{ticker}_initial_margin'] -= initial_margin_pre_unit* abs(buy_qty)
            if realised_pnl > 0:
                self.df.loc[index, f'{ticker}_long_win'] += 1        


        # win rate calculation 
        long_trades = self.df.loc[index, f'{ticker}_long_trade']
        short_trades = self.df.loc[index, f'{ticker}_short_trade']
        long_win = self.df.loc[index, f'{ticker}_long_win']
        short_win = self.df.loc[index, f'{ticker}_short_win']
        
        if long_trades > 0:
            self.df.loc[index, f'{ticker}_long_win_rate'] = long_win / long_trades
        else:
            self.df.loc[index, f'{ticker}_long_win_rate'] = 0.0

        if short_trades > 0:
            self.df.loc[index, f'{ticker}_short_win_rate'] = short_win / short_trades
        else:
            self.df.loc[index, f'{ticker}_short_win_rate'] = 0.0
        
        self.df.loc[index, f'{ticker}_action_signal'] = action_signal
        self.df.loc[index, 'trading_strategy'] = strategy_type



    def run(self):
        for strategy in self.strategies:
            for col in self.df.columns:
                if col[-17:] == '_holding_position':
                    ticker = col.split('_')[0]
                    self.ticker_lst.append(ticker)
                    for i in self.df.index:
                        strategy.next(ticker, self, i)

    def update_position_info(self, index, ticker):
        holding_position = self.df.loc[index, ticker + '_holding_position']
        price = self.df.loc[index, ticker]
        average_cost = self.df.loc[index, ticker + '_average_cost']

        # 市值
        self.df.loc[index, ticker + '_holding_market_value'] = holding_position * price

        # 多單
        if holding_position > 0:
            holding_pnl = (price - average_cost) * holding_position * self.contract_size
        # 空單
        elif holding_position < 0:
            holding_pnl = (average_cost - price) * abs(holding_position) * self.contract_size
        else:
            holding_pnl = 0.0

        self.df.loc[index, ticker + '_holding_PnL'] = holding_pnl
    
    def calculate_return(self):
        self.df['Daily_return'] = self.df['Total_equity'].pct_change().fillna(0)
        self.returns = self.df['Daily_return']

    def reset(self, initial_cash=0):
        self.df = pd.DataFrame() 
        self.strategies = []  
        self.results = {}  
        self.cash = 0  
        self.equity = self.cash 

    def delete_data(self, dataname):
         del_col = [col for col in self.df.columns if dataname in col]
         self.df.drop(columns=del_col, inplace=True)

    class AnalyseTool:
        def __init__(self, backtest):
            self.backtest = backtest

        def sharpe_ratio(self, risk_free_rate=0.043, periods=252):
            risk_free_rate = risk_free_rate 
            annual_return = self.backtest.returns.mean() * periods
            annual_volatility = self.backtest.returns.std() * np.sqrt(periods)
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            return sharpe_ratio

        def maximum_drawdown(self):
            equity_curve = self.backtest.df['Total_equity']
            drawdown = (equity_curve / equity_curve.cummax()) - 1
            max_drawdown = drawdown.min()
            return max_drawdown






    

