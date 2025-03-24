import pandas as pd
import numpy as np

class Backtest:
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

    def add_data(self, dataname, data):
        self.ticker = dataname
        self.df = pd.concat([self.df, data], axis=1)
        self.df = self.df.rename(columns={'Close': dataname})
        self.df['Cash'] = self.df['Cash'].fillna(self.cash)
        self.df['Total_equity'] = self.df['Total_equity'].fillna(self.cash)
        self.df[dataname + '_holding_position'] = 0
        self.df[dataname + '_average_cost'] = 0
        self.df[dataname + '_action_signal'] = 0
        self.df[dataname + '_holding_market_value'] = 0
        self.df[dataname + '_holding_PnL'] = 0
        self.results[dataname] = []  

    def add_strategy(self, strategy_class, **kwargs):
        strategy_instance = strategy_class(self.df, **kwargs) 
        if hasattr(strategy_instance, 'add_reference_data'):
            print("Adding reference data.")
            strategy_instance.add_reference_data(self, self.ticker)
        else:
            print("add_reference_data method not found.")
        self.strategies.append(strategy_instance)

    def run(self):
        for strategy in self.strategies:
            for col in self.df.columns:
                if col[-17:] == '_holding_position':
                    ticker = col.split('_')[0]
                    self.ticker_lst.append(ticker)
                    for i in self.df.index:
                        strategy.next(ticker, self, i)
    
    def calculate_return(self):
        for ticker in self.ticker_lst:
            self.df[ticker + '_holding_market_value'] = self.df[ticker + '_holding_position'] * self.df[ticker]
            self.df[ticker + '_holding_PnL'] = (
                (self.df[ticker] - self.df[ticker + '_average_cost']) * self.df[ticker + '_holding_position']
            ).fillna(0)
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

        def sharpe_ratio(self, risk_free_rate=0.043):
            risk_free_rate = risk_free_rate 
            annual_return = self.backtest.returns.mean() * 252
            annual_volatility = self.backtest.returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            return sharpe_ratio

        def maximum_drawdown(self):
            equity_curve = self.backtest.df['Total_equity']
            drawdown = (equity_curve / equity_curve.cummax()) - 1
            max_drawdown = drawdown.min()
            return max_drawdown








    

