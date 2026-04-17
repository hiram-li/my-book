# Chapter 3: 所需工具與技能

## 量化分析的技術棧

建立一套完整的量化研究環境需要多方面工具的配合。以下是推薦的工具棧：

```
┌─────────────────────────────────────────────────────┐
│                   研究層 (Research)                  │
│  Jupyter Notebook / VS Code / Python                │
├─────────────────────────────────────────────────────┤
│                  數據層 (Data)                       │
│  pandas / numpy / SQLite / PostgreSQL               │
├─────────────────────────────────────────────────────┤
│                  策略層 (Strategy)                  │
│  TA-Lib / scipy / statsmodels / sklearn            │
├─────────────────────────────────────────────────────┤
│                  回測層 (Backtesting)               │
│  Backtrader / Zipline / VectorBT / custom          │
├─────────────────────────────────────────────────────┤
│                  執行層 (Execution)                 │
│  Broker API / Interactive Brokers / Alpaca         │
├─────────────────────────────────────────────────────┤
│                  基礎設施 (Infrastructure)           │
│  Linux / Docker / Git / Cloud Servers              │
└─────────────────────────────────────────────────────┘
```

## 編程語言：Python

Python 是量化分析領域最流行的編程語言，原因如下：

| 優勢 | 說明 |
|------|------|
| 生態豐富 | NumPy, pandas, scikit-learn 等庫幾乎涵蓋所有需求 |
| 易學易用 | 語法簡潔，適合快速原型開發 |
| 社區活躍 | 有大量量化交易的教程和示例 |
| 整合方便 | 容易與其他系統和 API 對接 |

### 必裝的 Python 庫

```bash
# 數據處理
pip install numpy pandas scipy

# 金融數據
pip install yfinance akshare

# 回測框架
pip install backtrader vectorbt

# 技術分析
pip install ta-lib

# 機器學習
pip install scikit-learn statsmodels

# 視覺化
pip install matplotlib seaborn plotly

# 數據庫
pip install sqlalchemy

# 經紀商 API
pip install ib_insync alpaca-trade-api
```

## 數據來源

### 港股數據

| 來源 | 說明 | 費用 |
|------|------|------|
| Yahoo Finance | 歷史數據，延遲15分鐘 | 免費 |
| 聚亨 (AKShare) | A股、港股期貨數據 | 免費 |
| TuShare | A股詳細數據 | 免費/付費 |
| 萬得 (Wind) | 專業金融數據 | 付費 |
| Bloomberg | 機構級數據 | 昂貴 |

### 美股數據

| 來源 | 說明 | 費用 |
|------|------|------|
| Yahoo Finance | 歷史數據 | 免費 |
| Alpha Vantage | 股票、FX、加密 | 免費/付費 |
| IEX Cloud | 實時股票數據 | 付費 |
| Polygon | 專業級市場數據 | 付費 |
| Bloomberg | 機構級數據 | 昂貴 |

## 數據庫存儲

### 方案一：SQLite（輕量級）

適合業餘愛好者和小型策略研究：

```python
import sqlite3

conn = sqlite3.connect('quant.db')
df.to_sql('stock_prices', conn, if_exists='replace')
```

### 方案二：PostgreSQL（生產級）

適合機構和複雜策略：

```python
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@localhost:5432/quant')
df.to_sql('stock_prices', engine, if_exists='replace')
```

## 版本控制：Git

量化項目強烈建議使用 Git 進行版本控制：

```bash
# 初始化倉庫
git init

# 添加文件
git add .

# 提交
git commit -m "Initial commit: strategy v1"

# 推送到 GitHub
git remote add origin https://github.com/your-name/quant-project.git
git push -u origin main
```

好處：
- 追蹤策略的修改歷史
- 方便團隊協作
- 策略版本管理
- 異地備份

## 雲端計算資源

### 性價比高的選擇

| 提供商 | 特點 | 適合場景 |
|--------|------|----------|
| AWS EC2 | 實例類型多文檔全 | 靈活部署 |
| Google Cloud | 免費額度大 | 學習實驗 |
| 阿里雲 | 中文界面支付便利 | A股策略 |
| 騰訊雲 | 香港機房延遲低 | 港股策略 |
| Vultr | 按小時計費性價比高 | 短期實驗 |

### 推薦配置

對於個人量化投資者：
- CPU: 4核+
- RAM: 16GB+
- SSD: 100GB+
- 網絡: 穩定的互聯網連接

## 所需的數學與統計知識

### 必備基礎

1. **統計學**
   - 均值、方差、標準差
   - 正態分佈與其他分佈
   - 假設檢定（t檢定、F檢定等）
   - p值與顯著性水平

2. **線性代數**
   - 矩陣運算
   - 特徵值與特徵向量
   - 主成分分析（PCA）

3. **概率論**
   - 期望值與方差
   - 條件概率與貝葉斯定理
   - 隨機過程基礎

4. **時間序列分析**
   - 平穩性（Stationarity）
   - 自我相關（Autocorrelation）
   - AR、MA、ARMA 模型

### 進階知識

- 隨機微分方程（適用於衍生品定價）
- 蒙特卡羅模擬
- 優化理論
- 機器學習算法

## 所需的金融知識

1. **市場微觀結構**
   - 訂單類型（市價單、限價單、止損單）
   - 交易所運作機制
   - 流動性與市場深度

2. **資產定價理論**
   - CAPM 模型
   - APT 模型
   - 有效市場假說（EMH）

3. **風險管理基礎**
   - Value at Risk（VaR）
   - 期望 Shortfall（CVaR）
   - 希臘字母（Greeks）

4. **衍生品基礎**
   - 期權的定義與類型
   - 期貨合約基礎
   - 對沖概念

---

## 重點回顧

1. Python 是量化分析的首選語言，擁有豐富的生態系統
2. 數據是量化策略的基礎，選擇合適的數據來源至關重要
3. Git 版本控制是量化項目管理的必備工具
4. 量化分析需要金融、數學、統計和編程的跨學科知識

---

## 下一步

下一章節我們將深入探討金融數據的獲取與處理，這是量化研究的核心基礎。
