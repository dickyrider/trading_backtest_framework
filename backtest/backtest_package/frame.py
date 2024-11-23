import pandas as pd

class Backtest:
    def __init__(self, initial_cash=0):
        self.df = pd.DataFrame()  # 存储不同股票的数据
        self.df['Total_equity'] = 0
        self.df['Cash'] = 0
        self.strategies = []
        self.results = {}
        self.cash = initial_cash
        self.equity = self.cash
    
    def initital_cash(self, initial_cash):
        self.cash  = initial_cash
        self.equity = self.cash

    def add_data(self, dataname, data):
        data.columns = [dataname] 
        self.df = pd.concat([self.df, data], axis=1)
        self.df['Cash'] = self.df['Cash'].fillna(self.cash)
        self.df['Total_equity'] = self.df['Total_equity'].fillna(self.cash )
        self.df[dataname + '_holding_position'] = 0
        self.df[dataname + '_holding_market_value'] = 0
        self.df[dataname + '_holding_PnL'] = 0
        self.results[dataname] = []  

    def add_strategy(self, strategy_class):
        strategy_instance = strategy_class(self.df)
        self.strategies.append(strategy_instance)

    def run(self):
        for strategy in self.strategies:
            for col in self.df.columns:
                print(col)
                if col[-17:] == '_holding_position':
                    ticker = col.split('_')[0]
                    for i in self.df.index:
                        strategy.next(ticker, self, i) 
                    self.df[ticker + '_holding_market_value'] = self.df[ticker + '_holding_position'] * self.df[ticker]
                    self.df[ticker + '_holding_PnL'] = self.df[ticker + '_holding_market_value'].pct_change() 
                    self.df[ticker + '_holding_PnL'] = self.df[ticker + '_holding_PnL'].replace([float('inf'), -float('inf')], 1)
                    
    def reset(self, initial_cash=0):
        self.df = pd.DataFrame() 
        self.strategies = []  
        self.results = {}  
        self.cash = 0  
        self.equity = self.cash 

    def delete_data(self, dataname):
         del_col = [col for col in self.df.columns if dataname in col]
         self.df.drop(columns=del_col, inplace=True)








    

