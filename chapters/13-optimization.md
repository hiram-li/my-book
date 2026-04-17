# Chapter 13: 策略優化技巧

## 優化的目標與原則

策略優化是在保持策略穩健性的前提下，最大化策略的風險調整後收益。

> 「優化的目標不是找到最高收益的參數，而是找到最穩健的參數區間。」

### 正確的優化心態

| ❌ 錯誤心態 | ✅ 正確心態 |
|------------|------------|
| 找到最好的參數 | 找到穩健的參數區間 |
| 最大化夏普比率 | 最大化信息比率 |
| 避免任何回撤 | 接受合理的回撤 |
| 完美擬合歷史 | 保持一定容錯空間 |

## 網格搜索（Grid Search）

```python
def grid_search(data, param_grid):
    """
    網格搜索：遍歷所有參數組合
    """
    import itertools
    
    # 生成所有參數組合
    param_names = param_grid.keys()
    param_combinations = itertools.product(*param_grid.values())
    
    results = []
    for params in param_combinations:
        param_dict = dict(zip(param_names, params))
        
        try:
            result = backtest(data, param_dict)
            result['params'] = param_dict
            results.append(result)
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

# 使用示例
param_grid = {
    'ma_fast': [20, 50, 100],
    'ma_slow': [100, 200, 300],
    'rsi_period': [7, 14, 21],
    'rsi_oversold': [20, 30],
    'rsi_overbought': [70, 80]
}

results = grid_search(train_data, param_grid)

# 找出最佳參數
best_params = results.loc[results['sharpe'].idxmax(), 'params']
print(f"最佳參數: {best_params}")
```

## 遺傳算法（Genetic Algorithm）

遺傳算法適合參數空間大且複雜的情況：

```python
def genetic_algorithm(data, param_bounds, population_size=50, generations=100):
    """
    遺傳算法優化
    """
    np.random.seed(42)
    
    # 初始化種群
    def create_individual():
        return {k: np.random.uniform(v[0], v[1]) for k, v in param_bounds.items()}
    
    def fitness(individual):
        try:
            return backtest(data, individual)['sharpe']
        except:
            return -999
    
    def crossover(parent1, parent2):
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if np.random.random() > 0.5 else parent2[key]
        return child
    
    def mutate(individual, mutation_rate=0.1):
        for key in individual.keys():
            if np.random.random() < mutation_rate:
                individual[key] = np.random.uniform(
                    param_bounds[key][0], param_bounds[key][1])
        return individual
    
    population = [create_individual() for _ in range(population_size)]
    
    for gen in range(generations):
        # 評估適應度
        fitness_scores = [fitness(ind) for ind in population]
        
        # 選擇
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        population = sorted_pop[:population_size//2]
        
        # 交叉
        while len(population) < population_size:
            p1 = np.random.choice(population)
            p2 = np.random.choice(population)
            child = crossover(p1, p2)
            child = mutate(child)
            population.append(child)
        
        if gen % 10 == 0:
            print(f"Generation {gen}: Best Sharpe = {max(fitness_scores):.3f}")
    
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual

# 使用
param_bounds = {
    'ma_fast': (10, 100),
    'ma_slow': (100, 300),
    'atr_multiplier': (1, 5)
}

best_params = genetic_algorithm(train_data, param_bounds)
```

## 貝葉斯優化

貝葉斯優化在參數空間大且評估成本高時特別有效：

```python
from skopt import gp_minimize
from skopt.space import Real, Integer

def bayesian_optimization(data, param_space, n_calls=50):
    """
    貝葉斯優化（使用高斯過程）
    """
    def objective(params_dict):
        result = backtest(data, params_dict)
        return -result['sharpe']  # 最小化負 Sharpe = 最大化 Sharpe
    
    # 定義參數空間
    dimensions = [
        Integer(10, 100, name='ma_fast'),
        Integer(100, 300, name='ma_slow'),
        Real(1.0, 5.0, name='atr_multiplier')
    ]
    
    # 優化
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )
    
    best_params = dict(zip([d.name for d in dimensions], result.x))
    
    return best_params, result

best_params, opt_result = bayesian_optimization(train_data, param_space)
```

## 穩健性評估

### 1. 參數平原分析

```python
def parameter_plateau_analysis(data, param_name, param_values, other_params):
    """
    分析策略在某參數上的表現平原
    """
    results = []
    
    for val in param_values:
        params = {**other_params, param_name: val}
        result = backtest(data, params)
        results.append({
            'param_value': val,
            'sharpe': result['sharpe'],
            'return': result['return'],
            'drawdown': result['max_drawdown']
        })
    
    df = pd.DataFrame(results)
    
    # 識別平原：如果多個相鄰參數值表現相近，說明有平原
    plateau_threshold = 0.1  # Sharpe 差異 < 0.1 認為是同一個 plateau
    
    return df

# 分析 MA 周期的 plateau
df = parameter_plateau_analysis(
    train_data, 'ma_slow', 
    range(150, 250, 10),
    {'ma_fast': 50, 'atr_multiplier': 3}
)

# 找出 Sharpe 最高的平原區間
stable_region = df[df['sharpe'] > df['sharpe'].max() - plateau_threshold]
print(f"穩健區間: MA slow = {stable_region['param_value'].min()} - {stable_region['param_value'].max()}")
```

### 2. 敏感性熱力圖

```python
def sensitivity_heatmap(data, param1_name, param1_range, param2_name, param2_range):
    """
    兩參數敏感性熱力圖
    """
    results = np.zeros((len(param1_range), len(param2_range)))
    
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            params = {param1_name: p1, param2_name: p2}
            result = backtest(data, params)
            results[i, j] = result['sharpe']
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(results, cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel(param2_name)
    plt.ylabel(param1_name)
    plt.xticks(range(len(param2_range)), param2_range)
    plt.yticks(range(len(param1_range)), param1_range)
    plt.title(f'{param1_name} vs {param2_name} 敏感性分析')
    plt.savefig('sensitivity_heatmap.png', dpi=150)
    
    return results

# 雙參數敏感性分析
sensitivity_heatmap(train_data, 'ma_fast', [20, 50, 100], 'ma_slow', [100, 150, 200, 250])
```

## 樣本外驗證流程

```python
def robust_optimization_pipeline(data, param_grid, n_walk_forward=5):
    """
    完整的穩健優化流程
    """
    all_results = []
    
    # Walk-forward 窗口
    tscv = TimeSeriesSplit(n_splits=n_walk_forward)
    
    for train_idx, val_idx in tscv.split(data):
        train = data.iloc[train_idx]
        val = data.iloc[val_idx]
        
        # Step 1: 在訓練集上網格搜索
        train_results = grid_search(train, param_grid)
        best_train_params = train_results.loc[
            train_results['sharpe'].idxmax(), 'params']
        
        # Step 2: 在驗證集上測試
        val_result = backtest(val, best_train_params)
        
        all_results.append({
            'train_params': best_train_params,
            'val_sharpe': val_result['sharpe'],
            'train_sharpe': train_results['sharpe'].max()
        })
    
    results_df = pd.DataFrame(all_results)
    
    # 評估穩健性
    avg_oos_sharpe = results_df['val_sharpe'].mean()
    sharpe_std = results_df['val_sharpe'].std()
    
    # 參數一致性
    param_agreement = {}
    for param in param_grid.keys():
        values = [r['train_params'][param] for r in all_results]
        param_agreement[param] = {
            'values': values,
            'unique_count': len(set(values))
        }
    
    return {
        'results': results_df,
        'avg_oos_sharpe': avg_oos_sharpe,
        'sharpe_std': sharpe_std,
        'param_agreement': param_agreement
    }
```

---

## 重點回顧

1. 網格搜索適合參數空間小且規則的情況，遺傳算法適合複雜空間
2. 貝葉斯優化樣本效率高，適合評估成本高的場景
3. 識別「參數平原」比找到單一最佳參數更有意義
4. Walk-forward 驗證是評估策略真實穩健性的金標準
5. 參數在不同窗口的一致性是策略穩健的重要標誌

---

## 下一步

下一章節我們將介紹風險管理的基礎知識，這是量化投資中不可或缺的一環。
