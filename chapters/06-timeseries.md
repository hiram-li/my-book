# Chapter 6: 時間序列分析

## 時間序列的基本概念

時間序列是量化分析的核心研究對象。與橫截面數據不同，時間序列數據具有時間依賴性，這使得分析方法有所不同。

### 平穩性（Stationarity）

時間序列分析中最基本也最重要的概念是平穩性：

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    """增強迪基-富勒檢驗"""
    result = adfuller(series.dropna())
    print(f'{name} - ADF統計量: {result[0]:.4f}')
    print(f'{name} - p-value: {result[1]:.6f}')
    print(f'{name} - 臨界值:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.4f}')
    
    if result[1] < 0.05:
        print(f'結論: {name} 是平穩的 (reject null)')
    else:
        print(f'結論: {name} 是非平穩的 (fail to reject null)')
    return result[1]

# 測試收益率的平穩性
adf_test(returns, '日收益率')
```

### 為什麼平穩性重要？

很多統計方法（如回歸分析）都假設數據是平穩的。如果數據是非平穩的，可能會出現「偽回歸」問題——兩個不相關的變量可能顯示出虛假的相關性。

## 自我相關（Autocorrelation）

### 計算自我相關係數

```python
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt

# 計算ACF和PACF
acf_values = acf(returns, nlags=20)
pacf_values = pacf(returns, nlags=20)

# 繪製相關圖
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(range(len(acf_values)), acf_values)
axes[0].axhline(y=0, linestyle='-', color='black')
axes[0].axhline(y=1.96/np.sqrt(len(returns)), linestyle='--', color='gray')
axes[0].axhline(y=-1.96/np.sqrt(len(returns)), linestyle='--', color='gray')
axes[0].set_title('自相關函數 (ACF)')

axes[1].bar(range(len(pacf_values)), pacf_values)
axes[1].axhline(y=0, linestyle='-', color='black')
axes[1].axhline(y=1.96/np.sqrt(len(returns)), linestyle='--', color='gray')
axes[1].axhline(y=-1.96/np.sqrt(len(returns)), linestyle='--', color='gray')
axes[1].set_title('偏自相關函數 (PACF)')

plt.tight_layout()
plt.savefig('acf_pacf.png', dpi=150)
```

### Ljung-Box 檢驗

Ljung-Box 檢驗用於檢驗序列是否存在顯著的自我相關：

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Ljung-Box檢驗
lb_result = acorr_ljungbox(returns, lags=[10, 20, 30], return_df=True)
print(lb_result)

# 如果p-value < 0.05，則拒絕「無自我相關」的零假設
```

## AR、MA、ARMA 模型

### 自動 ARIMA 模型選擇

```python
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# 自動選擇最佳 ARIMA 參數
model = auto_arima(
    returns.dropna(),
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,  # 自動確定差分階數
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(model.summary())
print(f'最佳參數: ARIMA{model.order}')
```

### 預測未來收益率

```python
# 預測未來5天的收益率
forecast = model.predict(n_periods=5)
print(f'預測收益率: {forecast}')

# 計算置信區間
forecast_with_ci = model.predict(n_periods=5, return_conf_int=True)
print(f'95%置信區間: {forecast_with_ci[1]}')
```

## 波動性建模：GARCH 模型

金融市場的波動性往往具有聚集性（volatility clustering）——大漲後往往跟隨大漲或大跌後跟隨大跌。GARCH 模型專門用於建模這種現象：

```python
from arch import arch_model

# 擬合 GARCH(1,1) 模型
garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1, mean='Constant', dist='t')
garch_result = garch_model.fit(disp='off')

print(garch_result.summary())

# 提取條件波動率
conditional_vol = garch_result.conditional_volatility / 100  # 轉換回小數
print(f'最新條件波動率（年化）: {conditional_vol.iloc[-1] * np.sqrt(252):.4f}')
```

## 協整檢驗與應用

### Engle-Granger 協整檢驗

如果兩個價格序列存在長期均衡關係，則稱它們是「協整」的。這對於配對交易策略非常重要：

```python
from statsmodels.tsa.stattools import coint

# 檢驗兩支股票是否協整
stock1 = df['Stock1_Close']
stock2 = df['Stock2_Close']

score, pvalue, _ = coint(stock1, stock2)
print(f'協整檢驗 p-value: {pvalue:.4f}')

if pvalue < 0.05:
    print('結論：兩支股票存在協整關係，適合配對交易')
else:
    print('結論：兩支股票不存在顯著的協整關係')
```

### 計算價差（Spread）

```python
# 使用線性回歸計算價差
from sklearn.linear_model import LinearRegression

# 擬合：stock1 = alpha + beta * stock2 + spread
reg = LinearRegression()
reg.fit(stock2.values.reshape(-1, 1), stock1.values)
spread = stock1 - reg.intercept_ - reg.coef_[0] * stock2

print(f'Alpha: {reg.intercept_:.4f}')
print(f'Beta: {reg.coef_[0]:.4f}')
print(f'價差均值: {spread.mean():.4f}')
print(f'價差標準差: {spread.std():.4f}')
```

---

## 重點回顧

1. 平穩性是時間序列分析的基礎，非平穩序列可能導致偽回歸
2. 自我相關分析揭示了時間序列中不同時間點之間的關聯性
3. ARMA 模型適用於平穩時間序列的建模和預測
4. GARCH 模型專門用於捕捉金融市場的波動性聚集現象
5. 協整檢驗是配對交易策略的理論基礎

---

## 下一步

下一章節我們將介紹因子模型的構建，這是量化選股策略的核心方法。
