# Chapter 12: 過擬合與樣本外測試

## 什麼是過擬合？

過擬合（Overfitting）是量化策略開發中最常見也最危險的問題。

> 「如果一個策略看起來太完美，那它很可能是有問題的。」

### 過擬合的定義

當策略在歷史數據上表現優異，但在未來數據上表現糟糕時，我們稱這個策略「過擬合」——它過度適應了歷史數據中的噪音，而非捕捉到真正的市場規律。

### 過擬合的常見原因

| 原因 | 說明 |
|------|------|
| 參數過多 | 策略有太多可調整參數 |
| 數據窺視 | 使用了未來信息 |
| 曲線擬合 | 過度優化到噪音上 |
| 樣本量不足 | 數據點相對於參數數量太少 |
| 多次測試 | 反覆調整參數直到滿意 |

## 樣本內與樣本外測試

### 簡單的訓練/測試分割

```python
def train_test_split(data, train_ratio=0.7):
    """
    簡單的訓練/測試分割
    """
    split_idx = int(len(data) * train_ratio)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    return train, test

# 使用
train_data, test_data = train_test_split(df, 0.7)

# 在訓練集上優化策略
optimize_strategy(train_data)

# 在測試集上驗證
results = backtest_strategy(test_data)
```

### 交叉驗證

```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_cv(data, n_splits=5):
    """
    時間序列交叉驗證（Walk-Forward Validation）
    
    每次用一個滾動窗口進行訓練和測試
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []
    
    for train_idx, test_idx in tscv.split(data):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        
        # 訓練策略
        optimize_strategy(train)
        
        # 測試策略
        result = backtest_strategy(test)
        cv_results.append(result)
    
    return pd.DataFrame(cv_results)

# 交叉驗證結果
cv_results = walk_forward_cv(df)
print(f"平均樣本外夏普比率: {cv_results['sharpe'].mean():.2f}")
print(f"樣本外穩定性: {(cv_results['sharpe'] > 0).mean()*100:.1f}%")
```

## 樣本外測試的關鍵原則

### 1. 不要在測試集上調整參數

```python
# ❌ 錯誤做法
for param in param_grid:
    strategy.set_params(param)
    result = backtest(test_data)  # 這是測試集！
    if result['sharpe'] > best:
        best = result
        best_params = param  # 這就是過擬合

# ✅ 正確做法
# Step 1: 在訓練集上優化
best_params = optimize(train_data)

# Step 2: 在測試集上驗證（不改參數）
final_result = backtest(test_data, params=best_params)
```

### 2. 留出驗證集

```python
# 三段分割：訓練 / 驗證 / 測試
def three_way_split(data, train_ratio=0.6, val_ratio=0.2):
    """
    訓練集 60% / 驗證集 20% / 測試集 20%
    """
    train_end = int(len(data) * train_ratio)
    val_end = int(len(data) * (train_ratio + val_ratio))
    
    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]
    
    return train, val, test

train, val, test = three_way_split(df)

# 訓練 → 驗證調整 → 最終測試
best_params = optimize(train)
best_params = refine_on_validation(best_params, val)
final_result = backtest(test, params=best_params)
```

## 過擬合檢驗方法

### 1. 參數敏感性分析

```python
def parameter_sensitivity_analysis(data, param_name, param_range):
    """
    測試策略對單個參數的敏感度
    """
    results = []
    
    for param_value in param_range:
        strategy.set_params(**{param_name: param_value})
        result = backtest(data)
        results.append({
            'param_value': param_value,
            'sharpe': result['sharpe'],
            'return': result['return'],
            'drawdown': result['max_drawdown']
        })
    
    df = pd.DataFrame(results)
    
    # 敏感度指標：參數變化時結果的波動
    sensitivity = df['sharpe'].std()
    
    return df, sensitivity

# 分析 MA 週期的敏感度
ma_range = range(10, 200, 5)
results, sensitivity = parameter_sensitivity_analysis(train_data, 'ma_period', ma_range)

print(f"MA 週期敏感度: {sensitivity:.4f}")
if sensitivity > 0.5:
    print("警告：策略對 MA 週期非常敏感，可能存在過擬合")
```

### 2. 蒙特卡羅模擬

```python
def monte_carlo_simulation(returns, n_simulations=1000, n_periods=252):
    """
    蒙特卡羅模擬：模擬策略在不同隨機路徑上的表現
    """
    results = []
    
    for _ in range(n_simulations):
        # 隨機重組收益率序列
        random_returns = np.random.choice(returns, size=n_periods, replace=True)
        cumulative_return = (1 + random_returns).prod() - 1
        results.append(cumulative_return)
    
    results = np.array(results)
    
    return {
        'mean': results.mean(),
        'std': results.std(),
        'percentile_5': np.percentile(results, 5),
        'percentile_95': np.percentile(results, 95),
        'prob_positive': (results > 0).mean()
    }

# 模擬結果
simulation = monte_carlo_simulation(strategy_returns)
print(f"95% 置信區間: [{simulation['percentile_5']:.2%}, {simulation['percentile_95']:.2%}]")
```

### 3. bootstrap 檢驗

```python
def bootstrap_test(returns, n_bootstrap=1000, confidence_level=0.95):
    """
    Bootstrap 檢驗策略收益是否顯著
    """
    observed_return = returns.mean()
    bootstrap_returns = []
    
    for _ in range(n_bootstrap):
        # 有放回抽樣
        sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_returns.append(sample.mean())
    
    bootstrap_returns = np.array(bootstrap_returns)
    
    # 計算置信區間
    lower = np.percentile(bootstrap_returns, (1-confidence_level)/2 * 100)
    upper = np.percentile(bootstrap_returns, (1+confidence_level)/2 * 100)
    
    # p-value
    p_value = (bootstrap_returns < 0).mean()
    
    return {
        'observed_return': observed_return,
        'confidence_interval': (lower, upper),
        'p_value': p_value,
        'is_significant': p_value < (1-confidence_level)
    }
```

## 避免過擬合的策略

### 1. 參數數量原則

> 規則：樣本內數據點 / 策略參數 > 100

```python
def check_parameter_ratio(data, n_params):
    """
    檢查參數數量是否合理
    """
    n_data_points = len(data)
    ratio = n_data_points / n_params
    
    if ratio < 50:
        print(f"警告：數據點/參數比 = {ratio:.1f}，建議至少 100")
    else:
        print(f"比例合理：{ratio:.1f}")
    
    return ratio

check_parameter_ratio(train_data, n_params=5)
```

### 2. 簡化策略

```python
# ❌ 複雜策略（容易過擬合）
strategy = {
    'ma_fast': 10,      # 快速均線
    'ma_slow': 200,     # 慢速均線
    'ma_mid': 50,       # 中速均線
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'atr_period': 14,
    'atr_multiplier': 3,
    'bollinger_period': 20,
    'bollinger_std': 2,
    # ... 更多參數
}

# ✅ 簡單策略（更穩健）
strategy = {
    'ma_fast': 50,      # 快速均線
    'ma_slow': 200,     # 慢速均線
}
```

### 3. 約束參數範圍

```python
# 不要讓參數在任意範圍內搜索
# ❌ 任意範圍
param_grid = {
    'ma_period': range(5, 500)  # 太寬了
}

# ✅ 合理範圍
param_grid = {
    'ma_period': [20, 50, 100, 150, 200]  # 基於經驗的合理值
}
```

---

## 重點回顧

1. 過擬合是策略在歷史數據上表現好但未來失敗的現象
2. 必須嚴格區分訓練集、驗證集和測試集，不能在測試集上調整參數
3. Walk-Forward 交叉驗證是評估策略穩健性的金標準
4. 參數敏感性分析和蒙特卡羅模擬可以幫助識別過擬合
5. 簡單策略比複雜策略更穩健，數據點與參數比應 > 100

---

## 下一步

下一章節我們將討論策略優化技巧，在避免過擬合的前提下最大化策略潛力。
