# Chapter 16: Risk Management Systems Design

## Risk Management System Architecture

A complete risk management system needs to cover multiple levels:

```
┌─────────────────────────────────────────────────────────┐
│                   Portfolio Risk Management              │
│  (Asset allocation, sector concentration, liquidity)      │
├─────────────────────────────────────────────────────────┤
│                   Strategy Risk Management               │
│  (Position limits, drawdown control, leverage)          │
├─────────────────────────────────────────────────────────┤
│                   Trading Risk Control                  │
│  (Real-time monitoring, automated stops, alerts)         │
├─────────────────────────────────────────────────────────┤
│                  后台風控                              │
│  (Settlement, reconciliation, anomaly detection)          │
└─────────────────────────────────────────────────────────┘
```

## Real-Time Risk Monitoring

### Position Limits System

```python
class PositionLimits:
    """
    Position limits system
    """
    def __init__(self, config):
        self.max_position_size = config['max_position_size']
        self.max_sector_exposure = config['max_sector_exposure']
        self.max_total_exposure = config['max_total_exposure']
        self.max_leverage = config['max_leverage']
        self.max_drawdown_limit = config['max_drawdown_limit']
    
    def check_position(self, symbol, quantity, price, current_positions, total_equity):
        """
        Check if new position is compliant
        """
        position_value = quantity * price
        leverage = self._calculate_leverage(current_positions, total_equity)
        
        checks = {
            'position_size': position_value <= self.max_position_size,
            'leverage': leverage <= self.max_leverage,
            'total_exposure': self._calculate_total_exposure(current_positions) + position_value <= total_equity * self.max_total_exposure
        }
        
        violations = [k for k, v in checks.items() if not v]
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'current_leverage': leverage
        }
```

### Drawdown Control

```python
class DrawdownController:
    """
    Drawdown control system
    """
    def __init__(self, max_drawdown=0.20, high_water_mark=None):
        self.max_drawdown = max_drawdown
        self.high_water_mark = high_water_mark
        self.current_drawdown = 0
        self.trading_allowed = True
    
    def update(self, current_equity):
        """
        Update drawdown status
        """
        if self.high_water_mark is None:
            self.high_water_mark = current_equity
        
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
        
        self.current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
        
        # Check if risk control triggered
        if self.current_drawdown >= self.max_drawdown:
            self.trading_allowed = False
            return {
                'action': 'STOP_TRADING',
                'reason': f'Drawdown {self.current_drawdown*100:.1f}% exceeds limit'
            }
        
        # Gradually reduce position based on drawdown
        if self.current_drawdown > self.max_drawdown * 0.5:
            reduction = self._calculate_position_reduction(self.current_drawdown)
            return {
                'action': 'REDUCE_POSITION',
                'reduction': reduction,
                'reason': f'Drawdown {self.current_drawdown*100:.1f}%, reduce position'
            }
        
        return {'action': 'NORMAL'}
    
    def _calculate_position_reduction(self, drawdown):
        """
        Calculate position reduction ratio based on drawdown
        """
        base = (drawdown - 0.5 * self.max_drawdown) / (0.5 * self.max_drawdown)
        reduction = min(base, 0.5)
        return max(reduction, 0)
```

## Risk Alert System

### Real-Time Monitoring Framework

```python
import logging
from datetime import datetime

class RiskMonitor:
    """
    Real-time risk monitoring system
    """
    def __init__(self, risk_limits, alert_callback=None):
        self.limits = risk_limits
        self.alert_callback = alert_callback
        self.logger = logging.getLogger('RiskMonitor')
        
        self.ALERT_NONE = 0
        self.ALERT_WARNING = 1
        self.ALERT_CRITICAL = 2
    
    def check_risk(self, portfolio_state):
        """
        Check all risk indicators
        """
        alerts = []
        
        var_status = self._check_var_limit(portfolio_state)
        if var_status['level'] > self.ALERT_NONE:
            alerts.append(var_status)
        
        dd_status = self._check_drawdown_limit(portfolio_state)
        if dd_status['level'] > self.ALERT_NONE:
            alerts.append(dd_status)
        
        lev_status = self._check_leverage_limit(portfolio_state)
        if lev_status['level'] > self.ALERT_NONE:
            alerts.append(lev_status)
        
        return alerts
    
    def _check_var_limit(self, state):
        """Check VaR limit"""
        var = state.get('var_95', 0)
        var_limit = self.limits.get('var_limit', 0.02)
        
        if var > var_limit * 1.5:
            return {'metric': 'VaR', 'level': self.ALERT_CRITICAL,
                    'message': f'VaR {var*100:.2f}% critically exceeds limit'}
        elif var > var_limit:
            return {'metric': 'VaR', 'level': self.ALERT_WARNING,
                    'message': f'VaR {var*100:.2f}% exceeds limit'}
        
        return {'metric': 'VaR', 'level': self.ALERT_NONE}
    
    def _check_leverage_limit(self, state):
        """Check leverage limit"""
        lev = state.get('leverage', 1.0)
        lev_limit = self.limits.get('max_leverage', 2.0)
        
        if lev > lev_limit * 1.2:
            return {'metric': 'Leverage', 'level': self.ALERT_CRITICAL,
                    'message': f'Leverage {lev:.2f}x critically exceeds limit'}
        elif lev > lev_limit:
            return {'metric': 'Leverage', 'level': self.ALERT_WARNING,
                    'message': f'Leverage {lev:.2f}x exceeds limit'}
        
        return {'metric': 'Leverage', 'level': self.ALERT_NONE}
```

## Stress Testing

### Scenario Analysis

```python
def stress_test(portfolio, scenarios):
    """
    Stress testing
    
    Simulate portfolio performance under different market scenarios
    """
    results = {}
    
    for scenario_name, scenario in scenarios.items():
        pnl = 0
        for position in portfolio['positions']:
            symbol = position['symbol']
            quantity = position['quantity']
            
            if symbol in scenario.get('price_changes', {}):
                price_change = scenario['price_changes'][symbol]
                position_pnl = quantity * position['entry_price'] * price_change
                pnl += position_pnl
        
        loss = pnl / portfolio['total_equity']
        
        results[scenario_name] = {
            'pnl': pnl,
            'loss_rate': loss,
            'loss_amount': abs(pnl) if loss < 0 else 0,
            'status': 'PASS' if loss > -portfolio['max_acceptable_loss'] else 'FAIL'
        }
    
    return results

# Define stress test scenarios
scenarios = {
    '2008_crisis': {
        'description': '2008 Financial Crisis',
        'price_changes': {
            'stock_a': -0.50,
            'stock_b': -0.40,
            'stock_c': -0.60
        }
    },
    'market_crash_20': {
        'description': 'Market drops 20%',
        'price_changes': {
            'stock_a': -0.20,
            'stock_b': -0.20,
            'stock_c': -0.20
        }
    }
}
```

## Risk Report System

```python
def generate_risk_report(portfolio_state, date):
    """
    Generate daily risk report
    """
    report = {
        'date': date,
        'total_equity': portfolio_state['total_equity'],
        'daily_pnl': portfolio_state.get('daily_pnl', 0),
        'daily_return': portfolio_state.get('daily_return', 0),
        'ytd_return': portfolio_state.get('ytd_return', 0),
        'current_drawdown': portfolio_state.get('current_drawdown', 0),
        'leverage': portfolio_state.get('leverage', 1.0),
        'var_95': portfolio_state.get('var_95', 0),
        'positions': []
    }
    
    for pos in portfolio_state.get('positions', []):
        report['positions'].append({
            'symbol': pos['symbol'],
            'quantity': pos['quantity'],
            'value': pos['value'],
            'weight': pos['value'] / portfolio_state['total_equity'],
            'unrealized_pnl': pos.get('unrealized_pnl', 0)
        })
    
    report['risk_metrics'] = {
        'var_95_daily': portfolio_state.get('var_95', 0),
        'var_95_annual': portfolio_state.get('var_95', 0) * np.sqrt(252),
        'volatility_30d': portfolio_state.get('volatility_30d', 0)
    }
    
    return report
```

---

## Summary

1. Risk management system needs to cover portfolio, strategy, trading, and back-office levels
2. Position limits and drawdown control are key to preventing catastrophic losses
3. VaR and CVaR are core risk indicators for real-time monitoring
4. Stress testing simulates extreme scenarios to ensure portfolio resilience
5. Regular risk reports help track and manage overall risk exposure

---

## Next Step

The next section covers practical case studies applying the book's theories to the Hong Kong market.
