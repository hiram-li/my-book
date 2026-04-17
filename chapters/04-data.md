# Chapter 4: 金融數據來源與處理

## 金融數據的類型

量化分析的基礎是數據。按照不同的維度，金融數據可以分為以下幾類：

### 按時間頻率分類

| 類型 | 頻率 | 典型用途 |
|------|------|----------|
| Tick Data | 毫秒級 | 高頻交易、流動性分析 |
| 分時數據 | 1min/5min/15min | 日內策略、短期分析 |
| 日線數據 | Daily | 日內交易、波段策略 |
| 週/月線 | Weekly/Monthly | 長期投資、風險分析 |

### 按資產類別分類

- **股票數據**：開盤價、最高價、最低價、收盤價、成交量（OHLCV）
- **期貨數據**：合約價格、持倉量、結算價
- **外匯數據**：匯率報價、央行基準利率
- **加密貨幣**：24/7交易的數字資產數據
- **基本面數據**：財務報表、經濟指標、公司公告

## 獲取港股數據

### 方法一：Yahoo Finance（推薦）

Yahoo Finance 是獲取港股數據最方便的方式：

```python
import yfinance as yf

# 下載騰訊控股的日線數據
ticker = yf.Ticker("0700.HK")
df = ticker.history(period="2y")

print(df.tail())
```

輸出：
```
                 Open    High     Low   Close  Volume  Dividends  Stock Splits
Date                                                                         
2024-01-15  298.00  301.20  297.50  300.80  2345678       0.0            0.0
2024-01-16  301.00  303.50  300.20  302.40  1923456       0.0            0.0
```

### 方法二：AKShare（中文市場）

AKShare 是中文市場的專業金融數據庫：

```python
import akshare as ak

# 獲取A股日線數據
df = ak.stock_zh_a_hist(symbol="000001", period="daily", 
                         start_date="20230101", end_date="20240101")

# 獲取港股實時行情
df = ak.hk_stock_zh_a_spot_em()
```

## 數據清洗與預處理

原始數據往往存在各種問題，需要進行清洗：

### 常見數據問題

1. **缺失值**：某些交易日沒有數據
2. **異常值**：價格暴漲暴跌（非正常交易）
3. **重複數據**：同一日期有多條記錄
4. **錯誤格式**：日期格式不一致

### 數據清洗實例

```python
import pandas as pd
import numpy as np

# 讀取數據
df = pd.read_csv('stock_data.csv', parse_dates=['Date'])
df = df.set_index('Date')

# 1. 處理缺失值
# 方法1：刪除有缺失值的行
df = df.dropna()

# 方法2：線性插值（適用於時間序列）
df = df.interpolate(method='linear')

# 2. 處理異常值（3個標準差之外的數據）
for col in ['Open', 'High', 'Low', 'Close']:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] > mean - 3*std) & (df[col] < mean + 3*std)]

# 3. 刪除重複數據
df = df[~df.index.duplicated(keep='last')]

# 4. 確保數據按日期排序
df = df.sort_index()

print(f"清洗後數據量: {len(df)} 行")
```

## 調整價格與收益率計算

### 收益率的類型

```python
# 簡單收益率
df['Returns'] = df['Close'].pct_change()

# 對數收益率（更適合統計分析）
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# 累計收益率
df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1

print(df[['Close', 'Returns', 'Log_Returns', 'Cumulative_Returns']].tail())
```

### 調整後收盤價

股票分紅和拆股會影響歷史價格走勢，需要進行調整：

```python
# yfinance 自動提供調整後的價格
ticker = yf.Ticker("0700.HK")
df = ticker.history(period="1y")

# Adj Close 就是調整後的收盤價
print(df[['Close', 'Adj Close']].head(10))
```

## 創建技術指標

```python
import numpy as np

def calculate_ma(df, window):
    """計算移動平均線"""
    return df['Close'].rolling(window=window).mean()

def calculate_rsi(df, window=14):
    """計算相對強弱指數"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 添加技術指標
df['MA5'] = calculate_ma(df, 5)
df['MA20'] = calculate_ma(df, 20)
df['MA60'] = calculate_ma(df, 60)
df['RSI'] = calculate_rsi(df)

print(df[['Close', 'MA5', 'MA20', 'RSI']].tail(10))
```

## 數據存儲策略

### 層級化存儲架構

```
量化數據/
├── raw/              # 原始數據（不改動）
│   ├── hk_stocks/
│   └── us_stocks/
├── processed/        # 清洗後的數據
│   ├── daily/       # 日線數據
│   └── minute/      # 分時數據
├── factors/         # 計算好的因子
└── backtest/       # 回測專用數據
```

### SQLite 數據庫範例

```python
import sqlite3
import pandas as pd

def save_to_sqlite(df, table_name, db_path='quant_data.db'):
    """保存數據到 SQLite 數據庫"""
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=True)
    conn.close()
    print(f"已保存 {len(df)} 行到 {table_name}")

# 使用
save_to_sqlite(df, 'hk_0700_daily')
```

---

## 重點回顧

1. 金融數據按頻率可分為 Tick、分時、日線、週/月線等類型
2. Yahoo Finance 和 AKShare 是獲取港股和A股數據的便捷工具
3. 數據清洗是量化分析的關鍵步驟，包括處理缺失值、異據和重複數據
4. 調整後價格考慮了分紅和拆股，是計算準確收益率的必要條件

---

## 下一步

下一章節我們將介紹統計學基礎，幫助你建立對市場數據的統計直覺，為因子設計和策略開發打下基礎。
