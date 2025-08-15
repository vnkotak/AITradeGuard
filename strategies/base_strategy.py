"""
Base Strategy Framework
Abstract base class for implementing trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio

from config import Config
from database.db_manager import DatabaseManager
from utils.technical_indicators import TechnicalIndicators
from utils.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    """
    
    def __init__(self, name: str, config: Config = None):
        self.name = name
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        self.technical_indicators = TechnicalIndicators(self.config)
        self.statistical_analysis = StatisticalAnalysis()
        
        # Strategy parameters
        self.parameters = {}
        self.default_parameters = {}
        
        # Performance tracking
        self.performance_metrics = {}
        self.trade_history = []
        
        # Strategy state
        self.is_active = False
        self.last_signal_time = None
        self.positions = {}
        
        # Risk management
        self.max_position_size = 0.1  # 10% of portfolio
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
    async def initialize(self):
        """Initialize strategy"""
        logger.info(f"Initializing strategy: {self.name}")
        
        try:
            # Set default parameters
            self.parameters = self.default_parameters.copy()
            
            # Initialize strategy-specific components
            await self._initialize_strategy()
            
            # Load historical performance
            await self._load_performance_history()
            
            self.is_active = True
            logger.info(f"Strategy {self.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing strategy {self.name}: {e}")
            raise
            
    @abstractmethod
    async def _initialize_strategy(self):
        """Strategy-specific initialization"""
        pass
        
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Generate trading signals from market data
        
        Args:
            market_data: Dictionary of symbol -> DataFrame with OHLCV data
            
        Returns:
            List of signal dictionaries with format:
            {
                'symbol': str,
                'signal_type': 'BUY' | 'SELL' | 'HOLD',
                'signal_strength': float (0-1),
                'confidence': float (0-1),
                'rationale': str,
                'target_price': float,
                'stop_loss': float,
                'take_profit': float,
                'position_size': float,
                'metadata': dict
            }
        """
        pass
        
    @abstractmethod
    async def update_positions(self, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Update existing positions and generate exit signals
        
        Args:
            current_data: Current market data
            
        Returns:
            List of position update signals
        """
        pass
        
    @abstractmethod
    def calculate_position_size(self, signal: Dict, portfolio_value: float) -> float:
        """
        Calculate position size for a signal
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Position size as fraction of portfolio
        """
        pass
        
    async def backtest(self, start_date: str, end_date: str, initial_capital: float = 1000000) -> Dict[str, Any]:
        """
        Backtest the strategy over a given period
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for backtest
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Backtesting strategy {self.name} from {start_date} to {end_date}")
        
        try:
            # Load historical data
            historical_data = await self._load_backtest_data(start_date, end_date)
            
            if not historical_data:
                raise ValueError("No historical data available for backtesting")
                
            # Initialize backtest state
            backtest_state = {
                'capital': initial_capital,
                'positions': {},
                'trades': [],
                'portfolio_values': [],
                'dates': []
            }
            
            # Run backtest simulation
            for date, market_data in historical_data.items():
                await self._process_backtest_day(date, market_data, backtest_state)
                
            # Calculate performance metrics
            performance = await self._calculate_backtest_performance(backtest_state)
            
            return {
                'strategy_name': self.name,
                'period': f"{start_date} to {end_date}",
                'initial_capital': initial_capital,
                'final_capital': backtest_state['capital'],
                'total_trades': len(backtest_state['trades']),
                'performance_metrics': performance,
                'trade_history': backtest_state['trades'],
                'portfolio_curve': list(zip(backtest_state['dates'], backtest_state['portfolio_values']))
            }
            
        except Exception as e:
            logger.error(f"Error backtesting strategy {self.name}: {e}")
            return {'error': str(e)}
            
    async def _process_backtest_day(self, date: datetime, market_data: Dict[str, pd.DataFrame], state: Dict):
        """Process a single day in the backtest"""
        
        try:
            # Generate signals for the day
            signals = await self.generate_signals(market_data)
            
            # Process each signal
            for signal in signals:
                await self._execute_backtest_signal(signal, state, date)
                
            # Update existing positions
            position_updates = await self.update_positions(market_data)
            
            # Process position updates
            for update in position_updates:
                await self._process_backtest_position_update(update, state, date)
                
            # Calculate portfolio value
            portfolio_value = await self._calculate_backtest_portfolio_value(state, market_data)
            
            state['portfolio_values'].append(portfolio_value)
            state['dates'].append(date)
            
        except Exception as e:
            logger.error(f"Error processing backtest day {date}: {e}")
            
    async def _execute_backtest_signal(self, signal: Dict, state: Dict, date: datetime):
        """Execute a signal in the backtest"""
        
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            
            if signal_type == 'HOLD':
                return
                
            # Calculate position size
            position_size = self.calculate_position_size(signal, state['capital'])
            
            if position_size <= 0:
                return
                
            # Execute trade
            if signal_type == 'BUY':
                await self._execute_backtest_buy(signal, position_size, state, date)
            elif signal_type == 'SELL':
                await self._execute_backtest_sell(signal, position_size, state, date)
                
        except Exception as e:
            logger.error(f"Error executing backtest signal: {e}")
            
    async def _execute_backtest_buy(self, signal: Dict, position_size: float, state: Dict, date: datetime):
        """Execute buy signal in backtest"""
        
        try:
            symbol = signal['symbol']
            price = signal.get('target_price', signal.get('current_price', 0))
            
            if price <= 0:
                return
                
            trade_value = state['capital'] * position_size
            quantity = trade_value / price
            
            if trade_value > state['capital']:
                return  # Insufficient capital
                
            # Update capital
            state['capital'] -= trade_value
            
            # Add position
            if symbol in state['positions']:
                # Average down
                existing = state['positions'][symbol]
                total_quantity = existing['quantity'] + quantity
                total_value = existing['quantity'] * existing['avg_price'] + trade_value
                avg_price = total_value / total_quantity
                
                state['positions'][symbol] = {
                    'quantity': total_quantity,
                    'avg_price': avg_price,
                    'entry_date': existing['entry_date'],
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit')
                }
            else:
                state['positions'][symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_date': date,
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit')
                }
                
            # Record trade
            state['trades'].append({
                'date': date,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'signal_confidence': signal.get('confidence', 0),
                'rationale': signal.get('rationale', '')
            })
            
        except Exception as e:
            logger.error(f"Error executing backtest buy: {e}")
            
    async def _execute_backtest_sell(self, signal: Dict, position_size: float, state: Dict, date: datetime):
        """Execute sell signal in backtest"""
        
        try:
            symbol = signal['symbol']
            
            if symbol not in state['positions']:
                return  # No position to sell
                
            position = state['positions'][symbol]
            price = signal.get('target_price', signal.get('current_price', 0))
            
            if price <= 0:
                return
                
            # Calculate quantity to sell
            sell_quantity = min(position['quantity'], position['quantity'] * position_size)
            sell_value = sell_quantity * price
            
            # Update capital
            state['capital'] += sell_value
            
            # Calculate P&L
            cost_basis = sell_quantity * position['avg_price']
            pnl = sell_value - cost_basis
            pnl_pct = (price / position['avg_price'] - 1) * 100
            
            # Update position
            position['quantity'] -= sell_quantity
            
            if position['quantity'] <= 0:
                del state['positions'][symbol]
                
            # Record trade
            state['trades'].append({
                'date': date,
                'symbol': symbol,
                'action': 'SELL',
                'quantity': sell_quantity,
                'price': price,
                'value': sell_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'hold_days': (date - position['entry_date']).days,
                'signal_confidence': signal.get('confidence', 0),
                'rationale': signal.get('rationale', '')
            })
            
        except Exception as e:
            logger.error(f"Error executing backtest sell: {e}")
            
    async def _process_backtest_position_update(self, update: Dict, state: Dict, date: datetime):
        """Process position update in backtest"""
        
        try:
            symbol = update['symbol']
            action = update.get('action', 'HOLD')
            
            if symbol not in state['positions'] or action == 'HOLD':
                return
                
            if action in ['STOP_LOSS', 'TAKE_PROFIT', 'EXIT']:
                # Close position
                position = state['positions'][symbol]
                price = update.get('price', 0)
                
                if price > 0:
                    sell_value = position['quantity'] * price
                    state['capital'] += sell_value
                    
                    # Calculate P&L
                    cost_basis = position['quantity'] * position['avg_price']
                    pnl = sell_value - cost_basis
                    pnl_pct = (price / position['avg_price'] - 1) * 100
                    
                    # Record trade
                    state['trades'].append({
                        'date': date,
                        'symbol': symbol,
                        'action': action,
                        'quantity': position['quantity'],
                        'price': price,
                        'value': sell_value,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'hold_days': (date - position['entry_date']).days,
                        'exit_reason': action
                    })
                    
                    del state['positions'][symbol]
                    
        except Exception as e:
            logger.error(f"Error processing backtest position update: {e}")
            
    async def _calculate_backtest_portfolio_value(self, state: Dict, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value in backtest"""
        
        try:
            total_value = state['capital']
            
            for symbol, position in state['positions'].items():
                if symbol in market_data:
                    current_price = market_data[symbol]['close'].iloc[-1]
                    position_value = position['quantity'] * current_price
                    total_value += position_value
                    
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating backtest portfolio value: {e}")
            return state['capital']
            
    async def _calculate_backtest_performance(self, state: Dict) -> Dict[str, float]:
        """Calculate performance metrics from backtest"""
        
        try:
            trades = pd.DataFrame(state['trades'])
            portfolio_values = np.array(state['portfolio_values'])
            
            if len(trades) == 0:
                return {}
                
            # Basic metrics
            total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
            
            # Trade metrics
            completed_trades = trades[trades['action'] == 'SELL']
            
            if len(completed_trades) > 0:
                win_rate = (completed_trades['pnl'] > 0).mean() * 100
                avg_win = completed_trades[completed_trades['pnl'] > 0]['pnl'].mean()
                avg_loss = completed_trades[completed_trades['pnl'] <= 0]['pnl'].mean()
                profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0
                avg_hold_days = completed_trades['hold_days'].mean()
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                avg_hold_days = 0
                
            # Risk metrics
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            if len(daily_returns) > 0:
                volatility = np.std(daily_returns) * np.sqrt(252) * 100
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
                
                # Max drawdown
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (peak - portfolio_values) / peak
                max_drawdown = np.max(drawdown) * 100
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
                
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_hold_days': avg_hold_days,
                'total_trades': len(completed_trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating backtest performance: {e}")
            return {}
            
    async def _load_backtest_data(self, start_date: str, end_date: str) -> Dict[datetime, Dict[str, pd.DataFrame]]:
        """Load historical data for backtesting"""
        
        try:
            query = """
            SELECT date, symbol, open, high, low, close, volume
            FROM market_data 
            WHERE date >= %s AND date <= %s
            ORDER BY date, symbol
            """
            
            data = await self.db_manager.fetch_dataframe(query, (start_date, end_date))
            
            if data.empty:
                return {}
                
            # Group by date
            historical_data = {}
            
            for date in data['date'].unique():
                date_data = data[data['date'] == date]
                
                daily_market_data = {}
                for symbol in date_data['symbol'].unique():
                    symbol_data = date_data[date_data['symbol'] == symbol]
                    daily_market_data[symbol] = symbol_data
                    
                historical_data[pd.to_datetime(date)] = daily_market_data
                
            return historical_data
            
        except Exception as e:
            logger.error(f"Error loading backtest data: {e}")
            return {}
            
    async def _load_performance_history(self):
        """Load historical performance metrics"""
        
        try:
            query = """
            SELECT * FROM strategy_performance 
            WHERE strategy_name = %s 
            ORDER BY date DESC 
            LIMIT 100
            """
            
            performance_data = await self.db_manager.fetch_all(query, (self.name,))
            self.performance_metrics = {
                'historical_performance': performance_data
            }
            
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set strategy parameters"""
        
        for key, value in parameters.items():
            if key in self.default_parameters:
                self.parameters[key] = value
            else:
                logger.warning(f"Unknown parameter {key} for strategy {self.name}")
                
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        return self.parameters.copy()
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        
        return {
            'strategy_name': self.name,
            'is_active': self.is_active,
            'last_signal_time': self.last_signal_time,
            'current_positions': len(self.positions),
            'parameters': self.parameters,
            'performance_metrics': self.performance_metrics
        }
        
    async def optimize_parameters(self, parameter_ranges: Dict[str, List], metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            parameter_ranges: Dictionary of parameter names and their ranges
            metric: Optimization target metric
            
        Returns:
            Optimization results
        """
        logger.info(f"Optimizing parameters for strategy {self.name}")
        
        try:
            from itertools import product
            
            # Generate parameter combinations
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            combinations = list(product(*param_values))
            
            optimization_results = []
            
            # Test each combination
            for combination in combinations[:50]:  # Limit for performance
                params = dict(zip(param_names, combination))
                
                # Temporarily set parameters
                original_params = self.parameters.copy()
                self.set_parameters(params)
                
                # Run backtest
                backtest_result = await self.backtest(
                    start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                if 'performance_metrics' in backtest_result:
                    performance = backtest_result['performance_metrics']
                    optimization_score = performance.get(metric, 0)
                    
                    optimization_results.append({
                        'parameters': params,
                        'performance': performance,
                        'optimization_score': optimization_score
                    })
                    
                # Restore original parameters
                self.parameters = original_params
                
            if optimization_results:
                # Find best parameters
                best_result = max(optimization_results, key=lambda x: x['optimization_score'])
                
                return {
                    'strategy_name': self.name,
                    'optimization_metric': metric,
                    'best_parameters': best_result['parameters'],
                    'best_performance': best_result['performance'],
                    'best_score': best_result['optimization_score'],
                    'total_combinations': len(optimization_results),
                    'all_results': optimization_results
                }
            else:
                return {'error': 'No valid optimization results'}
                
        except Exception as e:
            logger.error(f"Error optimizing parameters for strategy {self.name}: {e}")
            return {'error': str(e)}
            
    def validate_signal(self, signal: Dict) -> bool:
        """Validate a trading signal"""
        
        required_fields = ['symbol', 'signal_type', 'confidence']
        
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Signal missing required field: {field}")
                return False
                
        if signal['signal_type'] not in ['BUY', 'SELL', 'HOLD']:
            logger.warning(f"Invalid signal type: {signal['signal_type']}")
            return False
            
        if not 0 <= signal['confidence'] <= 1:
            logger.warning(f"Invalid confidence value: {signal['confidence']}")
            return False
            
        return True
        
    async def save_strategy_state(self):
        """Save current strategy state to database"""
        
        try:
            state_data = {
                'strategy_name': self.name,
                'parameters': self.parameters,
                'is_active': self.is_active,
                'positions': self.positions,
                'last_update': datetime.now()
            }
            
            # Implementation depends on database schema
            logger.info(f"Strategy state saved for {self.name}")
            
        except Exception as e:
            logger.error(f"Error saving strategy state: {e}")
            
    async def load_strategy_state(self):
        """Load strategy state from database"""
        
        try:
            # Implementation depends on database schema
            logger.info(f"Strategy state loaded for {self.name}")
            
        except Exception as e:
            logger.error(f"Error loading strategy state: {e}")
