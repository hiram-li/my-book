# Chapter 7: 因子模型的構建

## 什麼是因子？

因子（Factor）是解釋資產回報的共同變量。在量化投資中，因子是連接風險與收益的橋樑。一個因子代表了市場上一組資產的共同風險暴露。

> 「因子是市場風險溢价的來源，也是量化策略設計的核心構建模塊。」

## Fama-French 三因子模型

1992 年，Eugene Fama 和 Kenneth French 提出了著名的三因子模型，解釋了股票回報的很大一部分變異：

### 模型公式

$$R_i - R_f = \alpha_i + \beta_i (R_m - R_f) + s_i \cdot SMB + h_i \cdot HML + \epsilon_i$$

其中：
- $R_i$：資產 i 的回報
- $R_f$：無風險利率
- $R_m$：市場回報
- $SMB$（Small Minus Big）：規模因子（小盤股溢價）
- $HML$（High Minus Low）：價值因子（價值股溢價）

### Python 實作

```python
import pandas as pd
import numpy as np
from scipy import stats

def calculate_fama_french_factors(returns_df, market_returns, rf_rate=0.0):
    """
    計算 Fama-French 三因子迴歸
    """
    # 對齊數據
    aligned_data = returns_df.align(market_returns, join='inner')
    asset_returns = aligned_data[0]
    mkt_returns = aligned_data[1]
    
    # 市場溢價
    market_excess = mkt_returns - rf_rate
    
    # 簡化：假設 SMB 和 HML 為已知
    # 實際應用中需要從 FF 官網下載這些因子數據
    
    results = {}
    for col in asset_returns.columns:
        y = asset_returns[col] - rf_rate
        X = market_excess
        
        # OLS 迴歸
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        
        results[col] = {
            'alpha': intercept,
            'beta': slope,
            'r_squared': r_value**2,
            'p_value': p_value
        }
    
    return pd.DataFrame(results).T

# 使用示例
# factors = calculate_fama_french_factors(stock_returns, market_returns)
```

## Barra 因子模型

Barra CNE5 模型是機構投資者最廣泛使用的多因子模型之一，包含 10 個行業因子和 10 個風格因子：

### 行業因子

| 行業因子 | 描述 |
|----------|------|
| Basic Materials | 基礎材料 |
| Consumer Discretionary | 非必需消費 |
| Consumer Staples | 必需消費 |
| Energy | 能源 |
| Financials | 金融 |
| Health Care | 醫療保健 |
| Industrials | 工業 |
| Technology | 科技 |
| Telecommunications | 電信 |
| Utilities | 公用事業 |

### 風格因子

| 風格因子 | 描述 |
|----------|------|
| Size | 規模因子 |
| Value | 價值因子 |
| Momentum | 動量因子 |
| Quality | 質量因子 |
| Volatility | 波動率因子 |
| Growth | 成長因子 |
| Leverage | 槓桿因子 |
| Liquidity | 流動性因子 |
| Dividend Yield | 股息率因子 |
| Betting Against Beta | 低 Beta 因子 |

## 自定義因子設計

### 價值因子

```python
def calculate_value_factors(df):
    """計算多個價值因子"""
    
    # 市盈率 (P/E)
    df['PE'] = df['MarketCap'] / df['NetIncome']
    
    # 市凈率 (P/B)
    df['PB'] = df['MarketCap'] / df['BookValue']
    
    # 市銷率 (P/S)
    df['PS'] = df['MarketCap'] / df['Revenue']
    
    # EV/EBITDA
    df['EV_EBITDA'] = df['EnterpriseValue'] / df['EBITDA']
    
    # 股息率
    df['DividendYield'] = df['DividendPerShare'] / df['Price']
    
    return df

def calculate_momentum_factors(df, prices):
    """計算動量因子"""
    
    # 12個月動量
    df['Momentum_12M'] = prices.pct_change(periods=252)
    
    # 6個月動量
    df['Momentum_6M'] = prices.pct_change(periods=126)
    
    # 1個月動量
    df['Momentum_1M'] = prices.pct_change(periods=21)
    
    # 相對強度指數 (RSI)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_quality_factors(df):
    """計算質量因子"""
    
    # 資產回報率 (ROA)
    df['ROA'] = df['NetIncome'] / df['TotalAssets']
    
    # 股本回報率 (ROE)
    df['ROE'] = df['NetIncome'] / df['ShareholdersEquity']
    
    # 毛利率
    df['GrossMargin'] = (df['Revenue'] - df['COGS']) / df['Revenue']
    
    # 資產周轉率
    df['AssetTurnover'] = df['Revenue'] / df['TotalAssets']
    
    # 負債權益比
    df['DebtToEquity'] = df['TotalLiabilities'] / df['ShareholdersEquity']
    
    # 流動比率
    df['CurrentRatio'] = df['CurrentAssets'] / df['CurrentLiabilities']
    
    return df
```

## IC（Information Coefficient）分析

IC 是衡量因子預測能力的重要指標：

```python
def calculate_IC(factor_values, forward_returns, method='pearson'):
    """
    計算因子 IC 值
    
    Parameters:
    -----------
    factor_values : pd.DataFrame
        因子值
    forward_returns : pd.DataFrame
        未來收益
    method : str
        'pearson' 或 'spearman'
    """
    if method == 'pearson':
        ic = factor_values.corrwith(forward_returns, axis=0)
    else:
        ic = factor_values.corrwith(forward_returns, axis=0, method='spearman')
    
    return ic

def calculate_IC_series(factor_df, returns_df, rolling_window=12):
    """
    計算滾動 IC
    """
    ic_series = pd.DataFrame(index=factor_df.index[rolling_window:])
    
    for i in range(rolling_window, len(factor_df)):
        start_idx = i - rolling_window
        end_idx = i
        
        factor_window = factor_df.iloc[start_idx:end_idx]
        returns_window = returns_df.iloc[start_idx:end_idx]
        
        ic = calculate_IC(factor_window.mean(), returns_window.mean())
        ic_series.loc[factor_df.index[i]] = ic
    
    return ic_series

# 計算 IC 統計
def IC_statistics(ic_series):
    """計算 IC 的各項統計"""
    return {
        'IC_mean': ic_series.mean(),
        'IC_std': ic_series.std(),
        'IC_IR': ic_series.mean() / ic_series.std(),  # Information Ratio
        'IC_positive_rate': (ic_series > 0).mean(),
        'IC_t_stat': ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)))
    }
```

## 多因子模型構建

### 標準化因子

```python
def normalize_factors(factor_df, winsorize=True, percentile_range=(1, 99)):
    """
    標準化因子值
    """
    normalized = factor_df.copy()
    
    if winsorize:
        # 去極值
        for col in normalized.columns:
            low = normalized[col].quantile(percentile_range[0]/100)
            high = normalized[col].quantile(percentile_range[1]/100)
            normalized[col] = normalized[col].clip(low, high)
    
    # Z-score 標準化
    for col in normalized.columns:
        mean = normalized[col].mean()
        std = normalized[col].std()
        normalized[col] = (normalized[col] - mean) / std
    
    return normalized

def build_factor_portfolio(factor_values, n_quantiles=5, long_short=True):
    """
    根據因子值構建分層組合
    """
    # 將股票按因子值分組
    quantile_labels = pd.qcut(factor_values, q=n_quantiles, labels=False, duplicates='drop')
    
    portfolios = {}
    for q in range(n_quantiles):
        portfolios[f'Q{q+1}'] = (quantile_labels == q)
    
    # 計算每組的未來收益
    results = {}
    for name, mask in portfolios.items():
        results[name] = forward_returns[mask].mean(axis=1)
    
    return pd.DataFrame(results)
```

## 因子有效性檢驗

```python
def factor_validation(factor_df, returns_df, n_quantiles=5):
    """
    完整的因子有效性檢驗
    """
    results = {}
    
    for col in factor_df.columns:
        # 1. IC 分析
        ic = calculate_IC(factor_df[col], returns_df)
        ic_series = calculate_IC_series(factor_df[[col]], returns_df[[col]])
        ic_stats = IC_statistics(ic_series[col])
        
        # 2. 分層回測
        portfolios = build_factor_portfolio(factor_df[[col]], n_quantiles)
        portfolio_returns = portfolios.apply(lambda x: returns_df[x].mean(axis=1))
        
        # 3. 多空組合收益
        long_short_return = portfolio_returns[f'Q{n_quantiles}'] - portfolio_returns['Q1']
        
        results[col] = {
            'IC': ic[col],
            'IC_IR': ic_stats['IC_IR'],
            'IC_positive_rate': ic_stats['IC_positive_rate'],
            'long_short_mean': long_short_return.mean(),
            'long_short_std': long_short_return.std(),
            'long_short_sharpe': long_short_return.mean() / long_short_return.std() * np.sqrt(252)
        }
    
    return pd.DataFrame(results).T

# 顯示有效因子
# valid_factors = factor_validation(factor_df, forward_returns)
# print(valid_factors.sort_values('IC_IR', ascending=False))
```

---

## 重點回顧

1. 因子是解釋資產回報的共同變量，代表市場風險溢价的來源
2. Fama-French 三因子模型開創了多因子研究的先河
3. Barra 模型是機構最廣泛使用的多因子框架
4. IC（Information Coefficient）是衡量因子預測能力的核心指標
5. 因子有效性需要通過 IC 分析、分層回測和多空組合收益來全面評估

---

## 下一步

下一章節我們將介紹均值回歸策略，這是量化交易中最經典的策略類型之一。
