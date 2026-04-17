# Chapter 10: 統計套利

## 統計套利概述

統計套利（Statistical Arbitrage）是一種基於數學模型的套利策略，利用市場價格一時的偏離來獲取風險中性收益。

> 「統計套利的核心不是『無風險』，而是『低風險』。它通過數學模型識別定價錯誤，在錯誤修復時獲利。」

### 與傳統套利的區別

| 維度 | 傳統套利 | 統計套利 |
|------|----------|----------|
| 風險 | 接近零 | 低但非零 |
| 利潤 | 薄 | 累積性 |
| 規模 | 有限 | 可擴展 |
| 工具 | 嚴格匹配 | 統計匹配 |

## 均值回歸型統計套利

### 歐布里恩-斯科蘭茲方法

```python
def ou_modelspread(price1, price2, half_life=20):
    """
    Ornstein-Uhlenbeck 模型模擬價差的均值回歸
    
    dS = θ(μ - S)dt + σdW
    
    θ: 回歸速度
    μ: 均值
    σ: 波動率
    """
    from scipy.optimize import minimize
    
    def nll(params):
        theta, mu, sigma = params
        dt = 1
        spread = price1 - price2
        
        # 計算負對數似然
        mu_adjusted = spread[:-1] + theta * (mu - spread[:-1]) * dt
        variance = sigma**2 * dt
        nll = 0.5 * np.log(2 * np.pi * variance) + (spread[1:] - mu_adjusted)**2 / (2 * variance)
        
        return np.sum(nll)
    
    # 初始猜測
    init_params = [0.1, np.mean(price1 - price2), np.std(price1 - price2)]
    
    # 優化
    result = minimize(nll, init_params, bounds=[(0.001, 1), (-10, 10), (0.001, 1)])
    
    return result.x

theta, mu, sigma = ou_modelspread(df['Stock1'], df['Stock2'])
print(f"OU參數: θ={theta:.4f}, μ={mu:.4f}, σ={sigma:.4f}")
```

## 因子型統計套利

### 行業中性策略

```python
def industry_neutral_strategy(stock_returns, industry_dummies, factor_returns):
    """
    行業中性統計套利
    
    對每個行業內的股票進行多空配對
    """
    residuals = []
    
    for stock in stock_returns.columns:
        # 對每支股票迴歸
        y = stock_returns[stock]
        X = pd.concat([industry_dummies, factor_returns], axis=1)
        
        # OLS 殘差
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ beta
        residuals.append(residual)
    
    residuals_df = pd.DataFrame(residuals).T
    residuals_df.columns = stock_returns.columns
    
    return residuals_df

def rank_portfolio(residuals, top_pct=0.1, bottom_pct=0.1):
    """
    根據殘差排名構建多空組合
    
    買入殘差最低的股票（相對低估）
    賣出殘差最高的股票（相對高估）
    """
    n_stocks = len(residuals.columns)
    n_long = int(n_stocks * top_pct)
    n_short = int(n_stocks * bottom_pct)
    
    # 按殘差排序
    mean_residuals = residuals.mean()
    sorted_stocks = mean_residuals.sort_values()
    
    # 做空
    short_stocks = sorted_stocks.tail(n_short).index
    # 做多
    long_stocks = sorted_stocks.head(n_long).index
    
    return long_stocks, short_stocks
```

## 均值方差優化

```python
def mean_variance_portfolio(expected_returns, cov_matrix, risk_aversion=1.0):
    """
    經典均值方差優化
    
    max E[w'R] - (λ/2) * w'Σw
    
    λ: 風險厭惡系數
    """
    n_assets = len(expected_returns)
    
    # 最大化效用
    def neg_utility(w):
        port_return = np.dot(w, expected_returns)
        port_variance = np.dot(w.T, np.dot(cov_matrix, w))
        return -(port_return - 0.5 * risk_aversion * port_variance)
    
    # 約束：權重之和為 1，允許做空（約束可調整）
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((-1, 1) for _ in range(n_assets))  # 允許100%做空
    
    result = minimize(neg_utility, np.ones(n_assets)/n_assets,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

## 風險平價策略

```python
def risk_parity_portfolio(returns, lookback=252):
    """
    風險平價組合
    
    每個資產對組合總風險的貢獻相等
    """
    # 計算滾動協方差矩陣
    cov_matrix = returns.rolling(window=lookback).cov().iloc[-1]
    
    # 風險平價目標：每個資產的邊際風險貢獻相等
    def risk_contribution(weights, cov):
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        marginal_contrib = np.dot(cov, weights)
        risk_contrib = weights * marginal_contrib / port_vol
        return risk_contrib
    
    def target_risk_contrib(weights, cov, target):
        rc = risk_contribution(weights, cov)
        return np.sum((rc - target)**2)
    
    n_assets = len(returns.columns)
    init_weights = np.ones(n_assets) / n_assets
    
    # 目標風險貢獻 = 總風險 / N
    port_vol_init = np.sqrt(np.dot(init_weights.T, np.dot(cov_matrix, init_weights)))
    target_rc = np.full(n_assets, port_vol_init / n_assets)
    
    result = minimize(target_risk_contrib, init_weights,
                     args=(cov_matrix.values, target_rc),
                     method='SLSQP')
    
    return result.x
```

## 協整套利

```python
def cointegration_arb(price1, price2, window=60, entry_threshold=2, exit_threshold=0.5):
    """
    協整套利策略
    """
    from statsmodels.tsa.stattools import coint
    
    # 檢驗協整關係
    score, pvalue, _ = coint(price1, price2)
    print(f"協整 p-value: {pvalue:.4f}")
    
    # 計算價差
    spread = price1 - price2
    
    # 滾動均值和標準差
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    
    # Z-score
    z = (spread - mean) / std
    
    # 交易信號
    # 當 Z < -2：spread 過低，預期回歸，做多 spread（買1賣2）
    # 當 Z > 2：spread 過高，預期回歸，做空 spread（賣1買2）
    # 當 |Z| < 0.5：平倉
    
    return z, mean, std
```

---

## 重點回顧

1. 統計套利利用數學模型識別市場定價錯誤，風險較傳統套利高但更具可擴展性
2. Ornstein-Uhlenbeck 模型描述了價差的均值回歸行為
3. 行業中性策略通過對沖行業和市場風險來隔離股票特有因子
4. 風險平價策略確保每個資產對組合總風險的貢獻相等
5. 協整套利利用兩個資產的長期均衡關係進行交易

---

## 下一步

下一章節我們將深入探討回測系統的設計，這是驗證策略有效性的關鍵步驟。
