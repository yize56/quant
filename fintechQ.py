import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import random
from PIL import Image

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')  # 修改日志级别为DEBUG

# 关闭matplotlib的调试信息，设置为只显示ERROR级别日志
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# 关闭PIL的日志输出
logging.getLogger('PIL').setLevel(logging.ERROR)


# 1. 文件读写类
class FileHandler:
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """读取股票数据"""
        try:
            data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            logging.info(f"成功读取文件: {file_path}")
            return data
        except Exception as e:
            logging.error(f"读取文件失败: {e}")
            raise

    @staticmethod
    def save_data(data: pd.DataFrame, file_path: str):
        """保存数据到文件"""
        try:
            data.to_csv(file_path)
            logging.info(f"成功保存数据到文件: {file_path}")
        except Exception as e:
            logging.error(f"保存数据失败: {e}")
            raise


# 2. 数据生成与管理类
class StockDataGenerator:
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

    def generate_random_data(self, symbol: str) -> pd.DataFrame:
        """生成随机的股票历史数据"""
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        prices = np.random.uniform(100, 500, size=len(dates))
        volume = np.random.randint(1000, 10000, size=len(dates))

        df = pd.DataFrame({'Date': dates, 'Close': prices, 'Volume': volume}, index=dates)
        df['Symbol'] = symbol
        logging.info(f"成功生成股票数据: {symbol}")
        return df


# 3. 回测类
class Backtest:
    def __init__(self, data: pd.DataFrame, initial_capital: float, transaction_fee: float = 0.001):
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.capital = initial_capital
        self.positions = 0  # 当前持股数
        self.trade_history = []
        self.asset_values = []  # 存储每次回测的资产总值

    def execute_trade(self, date: pd.Timestamp, action: str, price: float, amount: int, slippage: float = 0.01):
        """执行交易操作，考虑滑点和交易延迟"""
        price *= (1 + random.uniform(-slippage, slippage))  # 模拟滑点
        cost = price * amount * (1 + self.transaction_fee)
        if action == 'buy':
            if self.capital >= cost:
                self.capital -= cost
                self.positions += amount
                self.trade_history.append((date, 'buy', price, amount))
                logging.info(f"买入: {amount}股, 当前资金: {self.capital}, 当前持股: {self.positions}")
            else:
                logging.warning(f"资金不足，无法买入{amount}股")
        elif action == 'sell':
            if self.positions >= amount:
                self.capital += price * amount * (1 - self.transaction_fee)
                self.positions -= amount
                self.trade_history.append((date, 'sell', price, amount))
                logging.info(f"卖出: {amount}股, 当前资金: {self.capital}, 当前持股: {self.positions}")
            else:
                logging.warning(f"没有足够的股票，无法卖出{amount}股")
        self.asset_values.append(self.capital + self.positions * price)  # 更新资产总值

    def calculate_returns(self):
        """计算最终收益"""
        return_rate = (self.capital + self.positions * self.data['Close'].iloc[
            -1] - self.initial_capital) / self.initial_capital
        logging.info(f"回测完成，最终收益率: {return_rate * 100:.2f}%")
        return return_rate


# 4. 强化学习 Q-learning 策略
class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay  # ε 衰减因子
        self.q_table = np.zeros((state_space_size, action_space_size))  # 初始化 Q 表

    def choose_action(self, state):
        """选择动作：基于ε-贪婪策略"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)  # 探索：随机选择动作
        else:
            return np.argmax(self.q_table[state])  # 利用：选择Q值最大的动作

    def update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        best_next_action = np.argmax(self.q_table[next_state])  # 下一状态的最佳动作
        # Q学习更新公式
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (
                reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )
        self.epsilon *= self.epsilon_decay  # 更新ε值（衰减）


# 5. 策略类：使用 Q-learning
class Strategy:
    def __init__(self, capital_allocation: float = 1.0, state_space_size=100, action_space_size=3):
        self.capital_allocation = capital_allocation  # 投资比例
        self.agent = QLearningAgent(state_space_size, action_space_size)  # 初始化Q-learning智能体

    def next_action(self, state: int) -> str:
        """策略：基于Q-learning决策"""
        action = self.agent.choose_action(state)  # 使用Q-learning选择动作
        if action == 0:
            return 'buy'  # 选择买入
        elif action == 1:
            return 'sell'  # 选择卖出
        else:
            return 'hold'  # 选择持有


# 6. 计算技术指标（例如：简单移动平均线、RSI等）
def calculate_indicators(data: pd.DataFrame):
    """计算技术指标：SMA, RSI"""
    data['SMA'] = data['Close'].rolling(window=14).mean()  # 14日简单移动平均线
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))  # 相对强弱指数
    return data


# 7. 可视化类
class Visualization:
    @staticmethod
    def plot_returns(returns: pd.Series):
        """绘制收益率曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(returns.index, returns, label='Portfolio Returns')
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_asset_values(asset_values: list):
        """绘制资产总值变化图"""
        plt.figure(figsize=(10, 6))
        plt.plot(asset_values, label='Asset Value')
        plt.title('Asset Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Asset Value')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_trades(data: pd.DataFrame, trade_history: list):
        """绘制买入和卖出点标记"""
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label='Stock Price')
        buys = [x[0] for x in trade_history if x[1] == 'buy']
        sells = [x[0] for x in trade_history if x[1] == 'sell']
        plt.scatter(buys, data.loc[buys]['Close'], marker='^', color='g', label='Buy')
        plt.scatter(sells, data.loc[sells]['Close'], marker='v', color='r', label='Sell')
        plt.title('Buy and Sell Points')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


# 8. 运行回测和 Q-learning 策略
def run_backtest():
    # 假设有一个简单的市场数据生成器
    data_generator = StockDataGenerator('2020-01-01', '2020-12-31')
    data = data_generator.generate_random_data('AAPL')
    data = calculate_indicators(data)  # 调用技术指标计算函数

    # 初始化回测
    backtest = Backtest(data, initial_capital=10000)

    # 初始化策略
    strategy = Strategy(state_space_size=100, action_space_size=3)

    # 模拟每个时间步的交易
    for i in range(1, len(data)):
        state = i  # 状态是当前时间的索引
        action = strategy.next_action(state)
        price = data['Close'].iloc[i]

        # 模拟交易
        if action == 'buy':
            backtest.execute_trade(data.index[i], 'buy', price, amount=10)
        elif action == 'sell':
            backtest.execute_trade(data.index[i], 'sell', price, amount=10)

    # 输出回测结果
    backtest.calculate_returns()

    # 可视化
    Visualization.plot_asset_values(backtest.asset_values)
    Visualization.plot_trades(data, backtest.trade_history)


# 执行回测
run_backtest()
