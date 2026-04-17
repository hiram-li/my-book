# Chapter 9: 趨勢追蹤策略

## 趨勢的概念

與均值回歸策略不同，趨勢追蹤策略（Trend Following）假設：

> 「趨勢是你的朋友」（The Trend is Your Friend）

趨勢追蹤者相信：
- 價格一旦形成趨勢，往往會持續一段時間
- 順勢而為，截斷虧損，讓利潤奔跑
- 不要嘗試預測市場的轉折點

## 移動平均交叉策略

最經典的趨勢追蹤策略是移動平均線交叉：

```python
def moving_average_crossover(prices, short_window=50, long_window=200):
    """
    雙均線交叉策略
    
    買入：短期均線上穿長期均線（金叉）
    賣出：短期均線下穿長期均線（死叉）
    """
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    
    # 信號
    buy_signal = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    sell_signal = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    
    return buy_signal, sell_signal, short_ma, long_ma

# 回測
buy, sell, short_ma, long_ma = moving_average_crossover(df['Close'], 50, 200)
```

## MACD 策略

MACD（Moving Average Convergence Divergence）是另一個廣泛使用的趨勢指標：

```python
def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    計算 MACD
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def macd_strategy(prices, fast=12, slow=26, signal=9):
    """
    MACD 交易策略
    """
    macd, signal_line, hist = calculate_macd(prices, fast, slow, signal)
    
    # 金叉：MACD 上穿信號線
    buy_signal = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
    
    # 死叉：MACD 下穿信號線
    sell_signal = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
    
    return buy_signal, sell_signal, macd, signal_line, hist
```

## 趨勢線突破策略

```python
def trendline_breakout(prices, window=20):
    """
    趨勢線突破策略
    
    當價格突破過去 N 天的最高點時買入
    當價格跌破過去 N 天的最低點時賣出
    """
    # 計算區間高點和低點
    highest = prices.rolling(window=window).max()
    lowest = prices.rolling(window=window).min()
    
    # 突破信號
    buy_signal = (prices > highest.shift(1)) & (prices.shift(1) <= highest.shift(1))
    sell_signal = (prices < lowest.shift(1)) & (prices.shift(1) >= lowest.shift(1))
    
    return buy_signal, sell_signal, highest, lowest

# 使用
buy, sell, high, low = trendline_breakout(df['Close'], 20)
```

## ATR 止損策略

Average True Range（ATR）可用於設置動態止損：

```python
def calculate_atr(df, window=14):
    """
    計算 ATR（平均真實波幅）
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr

def atr_trailing_stop(prices, atr, multiplier=3):
    """
    ATR 追蹤止損
    
    買入後，止損設置在：買入價 - ATR × 倍數
    隨著價格上漲，止損線同步上調
    """
    position = 0
    entry_price = 0
    stop_loss = 0
    
    stops = []
    for i in range(len(prices)):
        if position == 0 and prices.iloc[i] > prices.iloc[i-1]:  # 簡化買入條件
            position = 1
            entry_price = prices.iloc[i]
            stop_loss = entry_price - atr.iloc[i] * multiplier
        elif position == 1:
            # 追蹤止損：只上升，不下降
            new_stop = prices.iloc[i] - atr.iloc[i] * multiplier
            stop_loss = max(stop_loss, new_stop)
            
            if prices.iloc[i] < stop_loss:
                position = 0
                stop_loss = 0
        
        stops.append(stop_loss)
    
    return stops

atr = calculate_atr(df)
stops = atr_trailing_stop(df['Close'], atr)
```

## 趨勢追蹤的核心理念

### 1. 順勢而為

```python
def determine_trend(prices, short_ma, long_ma):
    """
    判斷市場趨勢
    """
    if short_ma > long_ma:
        return 'UPTREND'
    elif short_ma < long_ma:
        return 'DOWNTREND'
    else:
        return 'RANGE'

trend = determine_trend(df['Close'], short_ma, long_ma)
print(f"當前趨勢: {trend}")
```

### 2. 資金管理

```python
def position_sizing_vola(atrfactor, account_size, risk_per_trade=0.02):
    """
    根據 ATR 波動率調整倉位
    
    每次交易風險固定的資金
    ATR 越大，倉位越小
    """
    dollar_risk = account_size * risk_per_trade
    position_size = dollar_risk / atrfactor
    return position_size

# 根據波動率調整倉位
account = 100000  # 10 萬資金
risk = 0.02  # 每筆交易風險 2%
atr_value = atr.iloc[-1] * 3  # 3 倍 ATR 作為止損距離

shares = position_sizing_vola(atr_value, account, risk)
print(f"建議買入股數: {int(shares)}")
```

---

## 重點回顧

1. 趨勢追蹤假設趨勢會持續，順勢而為是核心原則
2. 移動平均線交叉、MACD、趨勢線突破是經典的趨勢追蹤工具
3. ATR 可用於設置合理的止損距離
4. 趨勢追蹤者「讓利潤奔跑」的關鍵是使用追蹤止損
5. 倉位大小應根據市場波動率調整

---

## 下一步

下一章節我們將介紹統計套利，一種更側重數學模型的中頻策略。
