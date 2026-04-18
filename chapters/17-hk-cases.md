# Chapter 17: 港股實戰案例

## 港股市場特點

香港股票市場是全球最活躍的市場之一，有其獨特的特性：

> 「港股是一個機構主導的市場，但又充滿散戶情緒。這種獨特性造就了獨特的量化機會。」

### 港股市場特徵

| 特徵 | 說明 |
|------|------|
| 行業集中 | 金融、地產、科技佔比高 |
| 北水南流 | 滬深港通改變市場結構 |
| 板塊效應強 | 題材炒作頻繁 |
| 波動性高 | 受中美兩地影響 |
|涡轮窩輪 | 窩輪市場活躍 |

## 案例一：騰訊控股（0700.HK）動量策略

### 策略背景

騰訊是港股龍頭，受多種因素影響，包括：
- 内地監管政策
- 遊戲業務收入
- 雲業務增長
- 宏觀經濟

### 策略設計

```python
def tencent_momentum_strategy(df, lookback=20, holding_period=5):
    """
    騰訊動量策略
    
    買入近期強勢的股票
    """
    # 計算動量信號
    df['momentum'] = df['close'].pct_change(lookback)
    
    # 計算相對強度
    df['relative_strength'] = df['momentum'] / df['momentum'].rolling(60).std()
    
    # 買入信號：動量創20日新高且相對強度 > 1
    df['signal'] = 0
    df.loc[(df['momentum'] > df['momentum'].rolling(20).max().shift(1)) &
           (df['relative_strength'] > 1), 'signal'] = 1
    
    # 止損：下跌8%止損
    df['stop_loss'] = df['close'] * 0.92
    
    return df

# 回測函數
def backtest_momentum(prices, signals, initial_capital=1000000):
    """
    動量策略回測
    """
    position = 0
    cash = initial_capital
    trades = []
    
    for i in range(len(prices)):
        if signals.iloc[i] == 1 and position == 0:
            # 買入
            shares = int(cash / prices.iloc[i] * 0.95)  # 預留5%手續費
            cost = shares * prices.iloc[i]
            if shares > 0:
                position = shares
                cash -= cost
                entry_price = prices.iloc[i]
                entry_date = prices.index[i]
                trades.append({'action': 'BUY', 'date': entry_date, 'price': entry_price, 'shares': shares})
        
        elif position > 0:
            # 檢查止損
            if prices.iloc[i] < entry_price * 0.92:
                # 止損賣出
                proceeds = position * prices.iloc[i]
                cash += proceeds * 0.995
                trades.append({'action': 'SELL', 'date': prices.index[i], 'price': prices.iloc[i], 'reason': 'STOP_LOSS'})
                position = 0
    
    # 最終平倉
    if position > 0:
        proceeds = position * prices.iloc[-1]
        cash += proceeds * 0.995
        trades.append({'action': 'SELL', 'date': prices.index[-1], 'price': prices.iloc[-1], 'reason': 'END'})
    
    final_value = cash
    return final_value, trades
```

## 案例二：盈富基金（2800.HK）均值回歸

### 策略背景

盈富基金是追蹤恒生指數的ETF，流動性好，適合均值回歸策略：

```python
def ihk_mean_reversion(df, window=20, entry_z=2, exit_z=0.5):
    """
    盈富基金均值回歸策略
    
    當偏離均價值超過2個標準差時入場
    """
    # 計算均值和標準差
    df['ma'] = df['close'].rolling(window).mean()
    df['std'] = df['close'].rolling(window).std()
    
    # Z分數
    df['z_score'] = (df['close'] - df['ma']) / df['std']
    
    # 交易信號
    df['signal'] = 0
    # Z < -2：價格過低，買入
    df.loc[df['z_score'] < -entry_z, 'signal'] = 1
    # Z > 2：價格過高，賣出
    df.loc[df['z_score'] > entry_z, 'signal'] = -1
    # |Z| < 0.5：平倉
    df.loc[abs(df['z_score']) < exit_z, 'signal'] = 0
    
    return df

def backtest_mean_reversion(df, initial_capital=1000000):
    """
    均值回歸策略回測
    """
    position = 0
    cash = initial_capital
    trades = []
    
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1 and position == 0:
            # 買入
            shares = int(cash / df['close'].iloc[i] * 0.995)
            if shares > 0:
                position = shares
                cash -= shares * df['close'].iloc[i]
                trades.append({'date': df.index[i], 'action': 'BUY', 'price': df['close'].iloc[i], 'shares': shares})
        
        elif df['signal'].iloc[i] == -1 and position == 0:
            # 賣出（做空需要先借股票，這裡簡化處理）
            pass
        
        elif df['signal'].iloc[i] == 0 and position > 0:
            # 平倉
            proceeds = position * df['close'].iloc[i] * 0.995
            cash += proceeds
            trades.append({'date': df.index[i], 'action': 'SELL', 'price': df['close'].iloc[i], 'reason': 'MEAN_REVERT'})
            position = 0
    
    if position > 0:
        proceeds = position * df['close'].iloc[-1] * 0.995
        cash += proceeds
    
    return cash, trades
```

## 案例三：高股息策略（5號仔）

### 策略背景

港股中有許多藍籌股派發穩定股息：

```python
def high_dividend_strategy(stocks, min_dividend_yield=4.0, min_liquidity=1000000):
    """
    高股息策略
    
    篩選股息率高、流動性好的股票
    """
    candidates = []
    
    for stock in stocks:
        # 計算股息率
        dividend_yield = stock['annual_dividend'] / stock['price']
        
        # 計算流動性（日均成交額）
        avg_volume = stock['volume'].rolling(20).mean()
        liquidity = avg_volume * stock['price']
        
        if dividend_yield >= min_dividend_yield and liquidity >= min_liquidity:
            candidates.append({
                'symbol': stock['symbol'],
                'dividend_yield': dividend_yield,
                'liquidity': liquidity,
                'price': stock['price']
            })
    
    # 按股息率排序
    candidates.sort(key=lambda x: x['dividend_yield'], reverse=True)
    
    return candidates[:10]  # 取前10

def rebalance_portfolio(holdings, target_weights, current_prices):
    """
    季度再平衡
    """
    trades = []
    
    for symbol in set(list(holdings.keys()) + list(target_weights.keys())):
        current_weight = holdings.get(symbol, 0) * current_prices.get(symbol, 0)
        target_weight = target_weights.get(symbol, 0)
        
        diff = target_weight - current_weight
        
        if abs(diff) > 10000:  # 差額超過1萬才交易
            action = 'BUY' if diff > 0 else 'SELL'
            shares = int(abs(diff) / current_prices[symbol])
            trades.append({
                'symbol': symbol,
                'action': action,
                'shares': shares,
                'value': abs(diff)
            })
    
    return trades
```

## 案例四：北水流入事件驅動

### 策略背景

滬深港通開通後，北水流向成為港股重要指標：

```python
def southbound_flow_strategy(df, flow_data, window=5, threshold=0.1):
    """
    北水流向事件驅動策略
    
    當北水大幅買入時跟進
    """
    # 計算北水淨流入
    df['net_flow'] = flow_data['buy_volume'] - flow_data['sell_volume']
    df['net_flow_pct'] = df['net_flow'] / flow_data['total_volume']
    
    # 滾動平均
    df['flow_ma'] = df['net_flow_pct'].rolling(window).mean()
    
    # 信號：北水流入超過門檻
    df['signal'] = 0
    df.loc[df['net_flow_pct'] > threshold, 'signal'] = 1
    df.loc[df['net_flow_pct'] < -threshold, 'signal'] = -1
    
    return df

def backtest_flow_strategy(df, initial_capital=1000000):
    """
    事件驅動回測
    """
    position = None
    cash = initial_capital
    trades = []
    
    for i in range(len(df)):
        if df['signal'].iloc[i] == 1 and position is None:
            # 買入
            shares = int(cash / df['close'].iloc[i] * 0.995)
            if shares > 0:
                position = {'shares': shares, 'entry': df['close'].iloc[i], 'date': df.index[i]}
                cash -= shares * df['close'].iloc[i]
                trades.append({'date': df.index[i], 'action': 'BUY', 'price': df['close'].iloc[i]})
        
        elif position is not None:
            # 持有3天後平倉
            holding_days = (df.index[i] - position['date']).days
            if holding_days >= 3:
                proceeds = position['shares'] * df['close'].iloc[i] * 0.995
                cash += proceeds
                pnl = proceeds - position['shares'] * position['entry']
                trades.append({'date': df.index[i], 'action': 'SELL', 'price': df['close'].iloc[i], 'pnl': pnl})
                position = None
    
    return cash, trades
```

## 實戰注意事項

### 1. 手續費計算

```python
def calculate_trading_cost(price, shares, market='HK'):
    """
    計算港股交易成本
    """
    commission = price * shares * 0.003  # 券商佣金0.3%
   印花税 = price * shares * 0.001  # 印花税0.1%
   交易所費用 = price * shares * 0.00005  # 交易所費用0.005%
    
    total_cost = commission + 印花税 + 交易所費用
    
    return {
        'commission': commission,
        'stamp_duty': 印花税,
        'exchange_fee': 交易所費用,
        'total': total_cost,
        'cost_rate': total_cost / (price * shares)
    }
```

### 2. 流動性檢查

```python
def check_liquidity(df, min_daily_volume=1000000, min_trading_days=20):
    """
    檢查股票流動性
    """
    avg_volume = df['volume'].rolling(min_trading_days).mean().iloc[-1]
    avg_value = avg_volume * df['close'].iloc[-1]
    
    return {
        'avg_daily_volume': avg_volume,
        'avg_daily_value': avg_value,
        'liquid': avg_value >= min_daily_volume
    }
```

---

## 重點回顧

1. 港股有其獨特的市場特性，包括北水流向、涡轮窩輪等
2. 騰訊動量策略利用強勢股動量效應，配合止損控制風險
3. 盈富基金均值回歸適合震盪市，能捕捉常態偏離
4. 高股息策略適合長線持有，結合季度再平衡
5. 北水流向事件驅動策略捕捉機構投資者行為

---

## 下一步

下一章節我們將總結實戰經驗，並探討未來的改進方向。
