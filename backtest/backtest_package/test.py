import pandas as pd
import yfinance as yf
from backtest_package import Backtest



class SimpleMovingAverageStrategy:
    def __init__(self, data):
        self.data = data
        self.short_window = 5  # 短期移动平均线窗口
        self.long_window = 20  # 长期移动平均线窗口
        self.position = 0  # 当前持仓

    def next(self, ticker, backtest, index):
        if index < self.long_window:  # 确保有足够的数据
            return
        
        # 计算短期和长期移动平均
        short_mavg = self.data[ticker].iloc[index - self.short_window + 1:index + 1].mean()
        long_mavg = self.data[ticker].iloc[index - self.long_window + 1:index + 1].mean()

        # 买入和卖出逻辑
        if short_mavg > long_mavg and self.position == 0:
            # 买入
            backtest.df.loc[index, ticker + '_position'] = 1
            self.position = 1
        elif short_mavg < long_mavg and self.position == 1:
            # 卖出
            backtest.df.loc[index, ticker + '_position'] = 0
            self.position = 0

# 测试代码
if __name__ == "__main__":
    # 创建回测实例
    backtest = frame(initial_cash=10000)

    # 添加示例数据（假设这是股票的价格数据）
    ticker = 'AAPL'
    data = yf.download(ticker, start='2019-11-11', end='2024-11-08')
    backtest.add_data('AAPL', data)

    # 添加策略
    backtest.add_strategy(SimpleMovingAverageStrategy)

    # 运行回测
    backtest.run()

    # 输出结果
    print(backtest.df)


ticker = 'AAPL'
data = yf.download(ticker, start='2019-11-11', end='2024-11-08')


import os
print(os.getcwd())

