# Chapter 5: 統計基礎與假設檢定

## 描述性統計

在開始複雜的統計分析之前，我們需要了解數據的基本特徵：

```python
import pandas as pd
import numpy as np

# 假設 df 是股價收益率數據
returns = df['Returns'].dropna()

# 描述性統計
print(returns.describe())
```

輸出：
```
count    504.000000
mean       0.000521
std        0.015231
min       -0.089123
25%       -0.007234
50%       -0.000123
75%        0.008567
max        0.073421
```

## 常用統計量

### 均值（Mean）

```python
mean = returns.mean()
print(f"日均收益率: {mean:.6f} ({mean*100:.4f}%)")
```

### 標準差（Standard Deviation）

```python
std = returns.std()
annual_vol = std * np.sqrt(252)  # 年化波動率
print(f"日標準差: {std:.6f}")
print(f"年化波動率: {annual_vol:.4f} ({annual_vol*100:.2f}%)")
```

### 偏度（Skewness）

偏度衡量數據分佈的對稱性：

```python
from scipy import stats

skewness = stats.skew(returns)
print(f"偏度: {skewness:.4f}")
# 負偏度表示左側尾部更長，極端負收益的可能性更大
```

### 峰度（Kurtosis）

```python
kurtosis = stats.kurtosis(returns)
print(f"峰度: {kurtosis:.4f}")
# 正峰度表示「肥尾」，極端事件比正態分佈更頻繁
```

## 正態性檢驗

金融數據往往不服從正態分佈，這對風險管理很重要：

### Jarque-Bera 檢驗

```python
from scipy.stats import jarque_bera

jb_stat, p_value = jarque_bera(returns)
print(f"JB統計量: {jb_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("結論：拒絕正態分佈假設（收益率不服從正態分佈）")
else:
    print("結論：不能拒絕正態分佈假設")
```

### 常見做法：使用學生-t分佈

由於金融數據普遍存在肥尾現象，建議使用學生-t分佈進行建模：

```python
from scipy.stats import t

# 擬合學生-t分佈
df_fit, loc_fit, scale_fit = t.fit(returns)
print(f"自由度: {df_fit:.2f}, loc: {loc_fit:.6f}, scale: {scale_fit:.6f}")

# 計算 VaR（95%置信度）
var_95 = t.ppf(0.05, df_fit, loc=loc_fit, scale=scale_fit)
print(f"歷史VaR (95%): {var_95:.4f}")
```

## 相關性分析

### 皮爾遜相關係數

```python
# 計算兩個資產的相關係數
corr = df['Returns'].corr(df['Market_Returns'])
print(f"與市場收益的相關係數: {corr:.4f}")
```

### 滾動相關性

```python
# 計算滾動60天的相關係數
rolling_corr = df['Returns'].rolling(window=60).corr(df['Market_Returns'])
```

## 假設檢定

### t檢定：檢驗收益率是否顯著不為零

```python
from scipy.stats import ttest_1samp

t_stat, p_value = ttest_1samp(returns, 0)
print(f"t統計量: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("結論：日均收益率顯著不為零")
else:
    print("結論：日均收益率與零沒有統計顯著差異")
```

### 兩個樣本的均值比較

```python
from scipy.stats import ttest_ind

# 比較兩段時間的收益率
returns_2023 = returns['2023']
returns_2024 = returns['2024']

t_stat, p_value = ttest_ind(returns_2023, returns_2024)
print(f"兩年均值差異的p-value: {p_value:.4f}")
```

## 多重比較問題

當進行多個統計檢驗時，需要考慮多重比較問題（Familywise Error Rate）：

```python
from statsmodels.stats.multitest import multipletests

# 假設有10個策略的p值
p_values = [0.01, 0.03, 0.04, 0.05, 0.10, 
            0.15, 0.20, 0.30, 0.40, 0.50]

# Bonferroni 校正
reject, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')
print(f"校正後的p值: {pvals_corrected}")
print(f"顯著的策略: {sum(reject)}")
```

---

## 重點回顧

1. 描述性統計提供了數據分佈的基本特徵，包括均值、標準差、偏度和峰度
2. 金融數據普遍不服從正態分佈，存在「肥尾」現象
3. Jarque-Bera 檢驗可用於檢驗收益率的正態性
4. 假設檢定是判斷策略效果顯著性的統計基礎
5. 進行多個統計檢驗時需要考慮多重比較問題

---

## 下一步

下一章節我們將深入探討時間序列分析，這是理解和預測金融市場數據的核心工具。
