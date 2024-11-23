import pandas as pd

class Backtest:
    def __init__(self, initial_cash=0):
        self.df = pd.DataFrame()  
        self.strategies = []
        self.results = {}
        self.cash = 0
        self.equity = self.cash
    
    def initital_cash(self, initial_cash):
        self.cash  = initial_cash
        self.equity = self.cash

    def add_data(self, dataname, data):
        self.df[dataname] = data
        self.df[dataname + '_position'] = 0
        self.df[dataname + '_market_value'] = 0
        self.df[dataname + '_PnL'] = 0
        self.results[dataname] = []  

    def add_strategy(self, strategy_class):
        strategy_instance = strategy_class(self.data)
        self.strategies.append(strategy_instance)

    def run(self):
        for strategy in self.strategies:
            for dataname in self.df.columns:
                if dataname.endswith('_position'):  
                    ticker = dataname.split('_')[0]  
                for i in range(len(self.df)):
                    strategy.next(ticker, self, i)  
                    self.update_equity(ticker, i) 

    def reset(self, initial_cash=0):
        self.df = pd.DataFrame() 
        self.strategies = []  
        self.results = {}  
        self.cash = 0  
        self.equity = self.cash 

    def delete_data(self, dataname):
         del_col = [col for col in self.df.columns if dataname in col]
         self.df.drop(columns=del_col, inplace=True)






    


