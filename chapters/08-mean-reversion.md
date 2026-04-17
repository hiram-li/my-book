# Chapter 8: 均值回歸策略

## 均值回歸理論

均值回歸（Mean Reversion）是量化交易的核心假設之一：資產價格在偏離其長期均值後，有傾向回歸到均值的趨勢。

> 「樹木不會長到天上，價格也不會永遠偏離價值。」—— 華爾街諺語

### 理論基礎

資產價格圍繞其內在價值波動。當價格低於價值時，最終會反彈；當價格高於價值時，最終會回落。這種波動創造了交易機會。

## 布林帶策略

布林帶（Bollinger Bands）是最經典的均值回歸工具：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """計算布林帶"""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std
    
    return ma, upper_band, lower_band

def bollinger_band_strategy(prices, window=20, num_std=2):
    """
    布林帶交易策略
    買入：價格觸及下軌
    賣出：價格觸及上軌
    """
    ma, upper, lower = calculate_bollinger_bands(prices, window, num_std)
    
    # 信號
    buy_signal = prices < lower  # 價格跌破下軌
    sell_signal = prices > upper  # 價格突破上軌
    
    return buy_signal, sell_signal, upper, lower, ma

# 計算信號
buy, sell, upper, lower, ma = bollinger_band_strategy(df['Close'])

# 策略回測
position = 0  # 0 = 空倉, 1 = 持倉
positions = []
for i in range(len(df)):
    if buy.iloc[i] and position == 0:
        position = 1  # 買入
    elif sell.iloc[i] and position == 1:
        position = 0  # 賣出
    positions.append(position)

df['Position'] = positions
df['Strategy_Returns'] = df['Close'].pct_change() * df['Position'].shift(1)
```

## RSI 均值回歸策略

相對強弱指數（RSI）可用於識別超買超賣：

```python
def calculate_rsi(prices, window=14):
    """計算 RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def rsi_strategy(prices, window=14, oversold=30, overbought=70):
    """
    RSI 交易策略
    買入：RSI < 30（超賣）
    賣出：RSI > 70（超買）
    """
    rsi = calculate_rsi(prices, window)
    
    buy_signal = rsi < oversold
    sell_signal = rsi > overbought
    
    return buy_signal, sell_signal, rsi

# 使用 RSI 策略
buy, sell, rsi = rsi_strategy(df['Close'])
```

## 價格通道策略

```python
def calculate_keltner_channels(prices, ema_window=20, atr_window=10, multiplier=2):
    """
    Keltner 通道策略
    """
    # 中軌：EMA
    ema = prices.ewm(span=ema_window).mean()
    
    # 計算 ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_window).mean()
    
    # 上下軌
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return ema, upper, lower

def keltner_strategy(prices, upper, lower):
    """
    Keltner 通道交易策略
    價格跌破下軌後反彈買入
    價格突破上軌後回落賣出
    """
    buy_signal = (prices < lower) & (prices.shift(1) >= lower.shift(1))
    sell_signal = (prices > upper) & (prices.shift(1) <= upper.shift(1))
    
    return buy_signal, sell_signal
```

## 配對交易（Pairs Trading）

配對交易是經典的均值回歸策略：

```python
def pairs_trading_strategy(stock1, stock2, lookback=60, entry_threshold=2, exit_threshold=0.5):
    """
    配對交易策略
    
    原理：當兩支股票的價差偏離歷史均值時，買入相對便宜的，賣出相對昂貴的
    """
    # 計算價差（Spread）
    ratio = stock1 / stock2
    
    # 滾動均值和標準差
    mean = ratio.rolling(window=lookback).mean()
    std = ratio.rolling(window=lookback).std()
    
    # Z-score
    z_score = (ratio - mean) / std
    
    # 交易信號
    # 當 z-score < -entry_threshold：買入 ratio（預期回歸）
    # 當 z-score > entry_threshold：賣出 ratio（預期回歸）
    # 當 |z-score| < exit_threshold：平倉
    
    buy_signal = z_score < -entry_threshold
    sell_signal = z_score > entry_threshold
    close_signal = np.abs(z_score) < exit_threshold
    
    return {
        'ratio': ratio,
        'mean': mean,
        'z_score': z_score,
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'close_signal': close_signal
    }

# 使用示例
result = pairs_trading_strategy(df['Stock1'], df['Stock2'])
```

## 均值回歸的風險管理

```python
def mean_reversion_risk_management(position_size, entry_price, stop_loss_pct=0.05):
    """
    均值回歸策略的風險管理
    """
    # 止損設置
    stop_loss = entry_price * (1 - stop_loss_pct)
    
    # 倉位大小（根據風險）
    max_loss_pct = 0.02  # 最多承受 2% 賬戶損失
    position_value = position_size * entry_price
    acceptable_loss = position_value * max_loss_pct / stop_loss_pct
    
    # 目標價（均值回歸目標）
    # 假設回到 20 日均線
    target_price = ma.iloc[-1]  # 需要傳入 MA 值
    
    return {
        'stop_loss': stop_loss,
        'target_price': target_price,
        'risk_reward_ratio': (target_price - entry_price) / (entry_price - stop_loss)
    }
```

---

## 重點回顧

1. 均值回歸基於資產價格會圍繞內在價值波動的假設
2. 布林帶、RSI、Keltner 通道是經典的均值回歸工具
3. 配對交易利用兩個相關資產的價差進行交易
4. 均值回歸策略需要設置止損以控制風險
5. 市場環境對均值回歸策略的有效性有顯著影響

---

## 下一步

下一章節我們將介紹趨勢追蹤策略，與均值回歸相反的一種策略思路。
