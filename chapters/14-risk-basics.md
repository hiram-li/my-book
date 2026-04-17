# Chapter 14: 風險管理基礎

## 風險的本質

風險（Risk）是量化投資中最重要的概念之一。在金融市場中，**收益與風險永遠是一枚硬幣的兩面**。

> 「如果你不考慮風險，那麼你所謂的『收益』只是暫時的。」

### 風險的定義

在量化投資中，風險通常被定義為**收益的不確定性**。這種不確定性可能帶來損失，也可能帶來額外收益。

### 風險的類型

| 風險類型 | 描述 | 例子 |
|----------|------|------|
| 市場風險 | 整體市場下跌 | 2008 年金融危機 |
| 流動性風險 | 無法快速變現 | 小市值股票 |
| 信用風險 | 交易對手違約 | 債券違約 |
| 操作風險 | 系統或人為錯誤 | 2012 年摩根大通 London Whale |
| 模型風險 | 模型本身的缺陷 | Long-Term Capital Management |
| 槓桿風險 | 過度使用槓桿 | 2021 年 Archegos Capital |

## 現代投資組合理論（MPT）

### 馬科維茨的偉大洞察

1952 年，Harry Markowitz 提出了現代投資組合理論，核心洞見是：

> **資產的分散化可以降低組合風險，而不會相應降低預期收益。**

### 有效前沿（Efficient Frontier）

```python
import numpy as np
import matplotlib.pyplot as plt

def efficient_frontier(expected_returns, cov_matrix, n_portfolios=100):
    """
    計算有效前沿
    """
    results = np.zeros((3, n_portfolios))
    weights_record = []
    
    for i in range(n_portfolios):
        # 隨機生成權重
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # 計算組合收益和風險
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0, i] = portfolio_return * 252  # 年化收益
        results[1, i] = portfolio_std * np.sqrt(252)  # 年化風險
        results[2, i] = portfolio_return * 252 / (portfolio_std * np.sqrt(252)) if portfolio_std > 0 else 0
    
    return results, weights_record

# 繪製有效前沿
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('風險 (年化標準差)')
plt.ylabel('收益 (年化)')
plt.title('有效前沿')
plt.savefig('efficient_frontier.png', dpi=150)
```

## 風險度量指標

### 1. 標準差（Volatility）

最傳統的風險度量：

```python
def calculate_volatility(returns, annualize=True):
    """
    計算收益率標準差
    """
    daily_vol = returns.std()
    
    if annualize:
        return daily_vol * np.sqrt(252)
    return daily_vol
```

### 2. Value at Risk（VaR）

VaR 是最廣泛使用的風險度量：

```python
def calculate_var(returns, confidence_level=0.95, time_horizon=1):
    """
    計算歷史 VaR
    
    參數：
    - confidence_level: 置信度（默認 95%）
    - time_horizon: 時間範圍（天）
    
    意義：在給定置信度下，最壞情況的損失
    """
    if time_horizon == 1:
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    # 多日 VaR（假設獨立性）
    daily_var = np.percentile(returns, (1 - confidence_level) * 100)
    multi_day_var = daily_var * np.sqrt(time_horizon)
    
    return multi_day_var

# 計算 95% VaR
var_95 = calculate_var(returns, 0.95)
print(f"95% 日 VaR: {var_95:.4f} ({var_95*100:.2f}%)")
```

### 3. Conditional VaR（CVaR / Expected Shortfall）

CVaR 是 VaR 的補充，衡量超過 VaR 損失的平均值：

```python
def calculate_cvar(returns, confidence_level=0.95):
    """
    計算 CVaR（條件 VaR / Expected Shortfall）
    
    意義：在 VaR 邊界外的平均損失
    """
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    
    return cvar

# 計算 95% CVaR
cvar_95 = calculate_cvar(returns, 0.95)
print(f"95% CVaR: {cvar_95:.4f} ({cvar_95*100:.2f}%)")
```

### 4. 最大回撤（Maximum Drawdown）

```python
def calculate_max_drawdown(equity_curve):
    """
    計算最大回撤
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    # 找到回升點
    peak_idx = equity_curve[:max_dd_idx].idxmax()
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_duration': (max_dd_idx - peak_idx).days,
        'peak_date': peak_idx,
        'trough_date': max_dd_idx
    }

# 使用
dd_info = calculate_max_drawdown(equity_curve)
print(f"最大回撤: {dd_info['max_drawdown']*100:.2f}%")
print(f"回撤持續: {dd_info['max_drawdown_duration']} 天")
```

## Beta 與系統性風險

### 計算 Beta

```python
def calculate_beta(stock_returns, market_returns):
    """
    計算 Beta 值
    
    Beta > 1: 市場風險敞口大
    Beta < 1: 市場風險敞口小
    Beta = 0: 與市場無關
    Beta < 0: 負相關
    """
    covariance = stock_returns.cov(market_returns)
    market_variance = market_returns.var()
    
    beta = covariance / market_variance
    
    return beta

# 計算個股 Beta
beta = calculate_beta(stock_returns, market_returns)
print(f"股票 Beta: {beta:.4f}")
```

## 下行風險度量

### 系統性下行風險

```python
def calculate_downside_beta(stock_returns, market_returns, target_return=0):
    """
    計算下行 Beta（相對於無風險利率或目標收益）
    """
    # 只考慮市場下跌時的數據
    down_market = market_returns < target_return
    down_stock = stock_returns[down_market]
    down_market_returns = market_returns[down_market]
    
    if down_market_returns.std() == 0:
        return np.nan
    
    covariance = down_stock.cov(down_market_returns)
    market_variance = down_market_returns.var()
    
    downside_beta = covariance / market_variance
    
    return downside_beta
```

## 希臘字母（Greeks）——衍生品風險

如果你交易期權或期貨，需要了解 Greeks：

```python
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    計算 Black-Scholes Greeks（簡化版本）
    """
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))) - r*K*np.exp(-r*T)*norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2) if option_type == 'call' else -K*T*np.exp(-r*T)*norm.cdf(-d2)
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

# 假設股價 100，執行價 105，剩餘 30 天，年化波動率 20%
greeks = calculate_greeks(S=100, K=105, T=30/365, r=0.02, sigma=0.2)
print(f"Delta: {greeks['Delta']:.4f}")
print(f"Gamma: {greeks['Gamma']:.6f}")
print(f"Vega: {greeks['Vega']:.4f}")
```

---

## 重點回顧

1. 風險是收益的不確定性，包括市場、流動性、信用、操作等多種類型
2. 現代投資組合理論證明了分散化可以降低組合風險
3. VaR、CVaR、最大回撤是三大核心風險度量指標
4. Beta 衡量系統性風險，描述了個股與市場的關聯性
5. Greeks（Delta、Gamma、Theta、Vega、Rho）是衍生品風險管理的基礎

---

## 下一步

下一章節我們將深入探討倉位管理與資金管理的具體方法。
