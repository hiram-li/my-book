# Chapter 11: 回測系統設計

## 回測的重要性

回測（Backtesting）是量化策略開發的核心步驟——用歷史數據驗證策略在過去的表現，評估策略的可行性和穩健性。

> 「過去的表現不代表未來結果，但過去的失敗往往預示未來的失敗。」

## 回測框架選擇

### 主流回測框架對比

| 框架 | 語言 | 優點 | 缺點 |
|------|------|------|------|
| Backtrader | Python | 功能完整、文檔全 | 速度一般 |
| Zipline | Python | Quantopian 出品、事件驅動 | 安裝複雜 |
| VectorBT | Python | 速度快、靈活 | 需要自己實現風控 |
| 自定義 | Python | 完全控制 | 開發時間長 |
| QuantConnect | Python/C# | 雲端運行、數據完整 | 依賴平台 |

### Backtrader 示例

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (
        ('maperiod', 20),
        ('printlog', False),
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            self.order = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        print(f'獲利: 毛利 {trade.pnl:.2f} 淨利 {trade.pnlcomm:.2f}')
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.dataclose[0] < self.sma[0]:
                self.order = self.buy()
        else:
            if self.dataclose[0] > self.sma[0]:
                self.order = self.sell()

# 運行回測
cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)

data = bt.feeds.YahooFinanceData(
    dataname='0700.HK',
    fromdate=datetime(2022, 1, 1),
    todate=datetime(2024, 12, 31))

cerebro.adddata(data)
cerebro.broker.setcash(1000000.0)
cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

print(f'起始資金: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'最終資金: {cerebro.broker.getvalue():.2f}')
```

## 事件驅動回測 vs 向量化回測

### 向量化回測（更簡單但有偏差）

```python
def vectorized_backtest(prices, signals, initial_capital=1000000):
    """
    向量化回測（簡化版本）
    
    假設：根據信號在次日開盤執行交易
    """
    # 計算日收益率
    returns = prices.pct_change().fillna(0)
    
    # 策略收益 = 持倉信號 × 次日收益
    # signals 是昨日收盤時產生的，所以持倉到今天
    strategy_returns = signals.shift(1).fillna(0) * returns
    
    # 累計收益
    cumulative = (1 + strategy_returns).cumprod()
    
    # 資金曲線
    equity = initial_capital * cumulative
    
    return {
        'returns': strategy_returns,
        'cumulative_returns': cumulative,
        'equity': equity
    }
```

### 事件驅動回測（更準確但複雜）

```python
class EventDrivenBacktest:
    def __init__(self, initial_capital, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.position = 0
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.trades = []
    
    def execute_trade(self, date, symbol, action, quantity, price):
        """執行交易"""
        if action == 'BUY':
            cost = quantity * price * (1 + self.commission)
            if cost > self.cash:
                return False  # 資金不足
            self.cash -= cost
            self.position += quantity
        elif action == 'SELL':
            if self.position < quantity:
                return False  # 持倉不足
            proceeds = quantity * price * (1 - self.commission)
            self.cash += proceeds
            self.position -= quantity
        
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price
        })
        return True
    
    def update_portfolio_value(self, price):
        """更新組合價值"""
        self.portfolio_value = self.cash + self.position * price
    
    def get_metrics(self):
        """計算回測指標"""
        # 計算收益率序列
        portfolio_values = [self.initial_capital]
        # ... 實現完整邏輯
        return metrics
```

## 回測關鍵指標

### 收益率指標

```python
def calculate_returns_metrics(returns):
    """
    計算收益率指標
    """
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': returns.std() * np.sqrt(252)
    }

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    計算夏普比率
    """
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
    """
    計算索提諾比率（只考慮下行風險）
    """
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < target_return]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
```

### 風險指標

```python
def calculate_max_drawdown(equity_curve):
    """
    計算最大回撤
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    max_dd_duration = (drawdown == max_dd).idxmax()
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_date': max_dd_duration
    }

def calculate_calmar_ratio(annual_return, max_drawdown):
    """
    計算卡瑪比率
    """
    if max_drawdown == 0:
        return 0
    return annual_return / abs(max_drawdown)

def calculate_win_rate(returns):
    """
    計算勝率
    """
    wins = (returns > 0).sum()
    total = len(returns.dropna())
    return wins / total if total > 0 else 0

def calculate_profit_loss_ratio(returns):
    """
    計算盈虧比
    """
    avg_win = returns[returns > 0].mean()
    avg_loss = abs(returns[returns < 0].mean())
    
    if avg_loss == 0:
        return np.inf
    return avg_win / avg_loss
```

## 回測結果分析

```python
def generate_backtest_report(returns, equity_curve, trades=None):
    """
    生成完整的回測報告
    """
    returns_metrics = calculate_returns_metrics(returns)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd_info = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(returns)
    pl_ratio = calculate_profit_loss_ratio(returns)
    
    report = {
        '=== 收益指標 ===': '',
        '總收益率': f"{returns_metrics['total_return']*100:.2f}%",
        '年化收益率': f"{returns_metrics['annual_return']*100:.2f}%",
        '年化波動率': f"{returns_metrics['annual_volatility']*100:.2f}%",
        '': '',
        '=== 風險指標 ===': '',
        '夏普比率': f"{sharpe:.2f}",
        '索提諾比率': f"{sortino:.2f}",
        '最大回撤': f"{max_dd_info['max_drawdown']*100:.2f}%",
        '': '',
        '=== 交易統計 ===': '',
        '勝率': f"{win_rate*100:.2f}%",
        '盈虧比': f"{pl_ratio:.2f}",
    }
    
    return report
```

---

## 重點回顧

1. Backtrader 和 VectorBT 是 Python 中流行的回測框架
2. 事件驅動回測比向量化回測更準確，但開發複雜度更高
3. 夏普比率、索提諾比率、最大回撤是評估策略的核心指標
4. 完整的回測報告應包括收益、風險、交易統計三大維度

---

## 下一步

下一章節我們將探討回測中最常見的問題——過擬合，以及如何避免它。
