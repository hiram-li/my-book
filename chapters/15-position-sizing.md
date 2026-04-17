# Chapter 15: 倉位管理與資金管理

## 資金管理的核心原則

資金管理（Position Sizing）是量化交易中最重要但最容易被忽視的部分。

> 「賺多少錢取決於策略，但虧多少錢取決於倉位管理。」

### 資金管理 vs 證券選擇

| 方面 | 證券選擇 | 資金管理 |
|------|----------|----------|
| 作用 | 選對的股票 | 對的股票下多少注 |
| 影響 | 決定收益方向 | 決定收益大小 |
| 重要性 | 高 | 極高 |
| 錯誤後果 | 錯過機會 | 爆倉破產 |

## 固定倉位法

### 1. 固定金額法

```python
def fixed_dollar_position(account_size, position_value):
    """
    固定倉位金額
    
    每次投入固定金額
    """
    return position_value

# 示例：每次投入 10 萬
position = fixed_dollar_position(1000000, 100000)
shares = position / current_price
```

### 2. 固定比例法

```python
def fixed_ratio_position(account_size, risk_ratio=0.02):
    """
    固定比例倉位
    
    每次投入账户的固定比例
    """
    return account_size * risk_ratio

# 示例：每次投入账户的 20%
position = fixed_ratio_position(1000000, 0.20)
shares = position / current_price
```

## 風險驅動的倉位管理

### 1. 固定風險倉位（Risk Parity）

```python
def risk_parity_position(account_size, entry_price, stop_loss_pct, risk_ratio=0.02):
    """
    固定風險倉位
    
    根據止損距離計算倉位
    每次交易風險不超過账户的固定比例
    """
    # 最大損失金額
    max_loss = account_size * risk_ratio
    
    # 止損距離
    stop_distance = entry_price * stop_loss_pct
    
    # 計算股數
    position_value = max_loss / stop_loss_pct
    shares = int(position_value / entry_price)
    
    return {
        'shares': shares,
        'position_value': shares * entry_price,
        'risk_amount': shares * stop_distance,
        'risk_ratio': (shares * stop_distance) / account_size
    }

# 示例：100 萬账户，最多承受 2% 風險，買入價 100 元，止損 5%
result = risk_parity_position(
    account_size=1000000,
    entry_price=100,
    stop_loss_pct=0.05,
    risk_ratio=0.02
)
print(f"買入股數: {result['shares']}")
print(f"實際風險: {result['risk_amount']:.0f} 元 ({result['risk_ratio']*100:.2f}%)")
```

### 2. 波動率調整倉位

```python
def volatility_adjusted_position(account_size, target_vol, current_vol, position_value):
    """
    波動率調整倉位
    
    根據市場波動率調整倉位大小
    """
    vol_ratio = target_vol / current_vol if current_vol > 0 else 1
    
    adjusted_position = position_value * vol_ratio
    
    return adjusted_position

# 示例：目標波動率 15%，當前波動率 20%
target_vol = 0.15
current_vol = calculate_volatility(returns)
adjusted = volatility_adjusted_position(1000000, target_vol, current_vol, 200000)
print(f"調整後倉位: {adjusted:.0f} 元")
```

### 3. ATR 倉位法

```python
def atr_position_size(account_size, entry_price, atr, atr_multiplier=3, risk_ratio=0.02):
    """
    ATR 倉位法
    
    使用 ATR 計算止損距離，根據風險比例計算倉位
    """
    # 止損價 = 買入價 - ATR × 倍數
    stop_price = entry_price - atr * atr_multiplier
    
    # 止損比例
    stop_pct = (entry_price - stop_price) / entry_price
    
    # 計算倉位
    result = risk_parity_position(account_size, entry_price, stop_pct, risk_ratio)
    
    return {
        **result,
        'stop_price': stop_price,
        'atr_multiplier': atr_multiplier
    }

# 示例
atr = calculate_atr(df).iloc[-1]
result = atr_position_size(1000000, entry_price=100, atr=atr, atr_multiplier=3)
print(f"止損價: {result['stop_price']:.2f}")
print(f"買入股數: {result['shares']}")
```

## 凱利公式（Kelly Criterion）

凱利公式是最優化的倉位管理公式：

```python
def kelly_criterion(win_rate, avg_win, avg_loss):
    """
    凱利公式
    
    f* = (b*p - q) / b
    其中：
    - b: 盈虧比（avg_win / avg_loss）
    - p: 勝率
    - q: 敗率 (1-p)
    """
    if avg_loss == 0:
        return 0
    
    b = avg_win / avg_loss  # 盈虧比
    p = win_rate  # 勝率
    q = 1 - win_rate  # 敗率
    
    kelly_fraction = (b * p - q) / b
    
    # 實際使用建議：只用 Kelly 的 50%（半 Kelly）
    half_kelly = kelly_fraction / 2
    
    # 限制最大倉位
    max_position = 0.25  # 最大不超過 25%
    
    final_fraction = min(half_kelly, max_position)
    
    return final_fraction

# 示例：勝率 55%，平均贏 1000，平均虧 800
kelly = kelly_criterion(0.55, 1000, 800)
print(f"建議倉位: {kelly*100:.1f}%")
```

## 倉位管理策略

### 1. 金字塔加倉法

```python
def pyramid_position(entry_price, current_price, base_shares, 
                    max_layers=3, layer_increment=0.5):
    """
    金字塔加倉策略
    
    隨著價格有利變化，逐步增加倉位
    """
    layers = []
    current_layer_price = entry_price
    remaining_layers = max_layers - 1  # 第一層已建立
    
    for i in range(remaining_layers):
        # 每層增加的比例
        additional_ratio = layer_increment * (1 - i * 0.2)
        additional_shares = int(base_shares * additional_ratio)
        
        layers.append({
            'layer': i + 2,
            'price': current_layer_price * 1.02,  # 比前層高 2%
            'shares': additional_shares
        })
        
        current_layer_price *= 1.02
    
    return layers

# 示例：第一層 1000 股，之後每層增加 50%
pyramid = pyramid_position(entry_price=100, current_price=100, base_shares=1000)
for layer in pyramid:
    print(f"第{layer['layer']}層: 價格 {layer['price']:.2f}, 股數 {layer['shares']}")
```

### 2. 槓桿管理

```python
def calculate_leverage(equity, position_value):
    """
    計算槓桿倍數
    """
    return position_value / equity

def leverage_management(equity, target_leverage=2.0, max_leverage=3.0):
    """
    槓桿管理
    
    根據目標槓桿調整倉位
    """
    if target_leverage > max_leverage:
        target_leverage = max_leverage
        print(f"警告：槓桿超過上限 {max_leverage}x，已調整為 {max_leverage}x")
    
    return target_leverage

# 計算建議倉位
leverage = 2.0
equity = 1000000
max_position = equity * leverage
print(f"最大倉位（{leverage}x槓桿）: {max_position/10000:.0f} 萬")
```

## 組合層面的倉位管理

### 行業配置

```python
def industry_allocation(account_size, n_industries, max_industry_weight=0.30):
    """
    行業配置
    
    避免過度集中於單一行業
    """
    base_weight = 1.0 / n_industries
    
    # 如果基礎權重超過上限，則需要調整
    if base_weight > max_industry_weight:
        print(f"警告：基礎權重 {base_weight*100:.1f}% 超過上限 {max_industry_weight*100:.0f}%")
        base_weight = max_industry_weight
    
    # 每個行業的倉位
    allocation = {
        f'Industry_{i+1}': account_size * base_weight 
        for i in range(n_industries)
    }
    
    return allocation

# 配置 10 個行業，最大每個 30%
allocation = industry_allocation(1000000, 10, 0.30)
for ind, value in allocation.items():
    print(f"{ind}: {value/10000:.0f} 萬")
```

---

## 重點回顧

1. 資金管理決定了虧損的幅度，比選股更重要
2. 固定風險倉位是風險驅動倉位管理的核心方法
3. ATR 和波動率調整倉位可以根據市場狀況動態調整
4. 凱利公式提供了最優倉位的理論上限，實際使用建議用半 Kelly
5. 組合層面需要控制行業集中度和槓桿倍數

---

## 下一步

下一章節我們將探討風險管理系統的設計與實踐。
