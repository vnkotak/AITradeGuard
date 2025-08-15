"""
Advanced Backtesting Engine
Comprehensive backtesting with regime-aware analysis and sophisticated metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ProcessPoolExecutor
import pickle

from config import Config
from database.db_manager import DatabaseManager
from models.ml_models import EnsembleMLModel
from models.regime_detection import MarketRegimeDetector
from trading.ai_gate import AIGate
from trading.portfolio_manager import PortfolioManager
from utils.statistical_analysis import StatisticalAnalysis
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Advanced backtesting engine with regime-aware analysis
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        self.statistical_analysis = StatisticalAnalysis()
        
        # Backtesting components
        self.ai_gate = None
        self.portfolio_manager = None
        self.regime_detector = None
        
        # Backtest parameters
        self.start_date = None
        self.end_date = None
        self.initial_capital = 1000000
        self.benchmark_symbol = self.config.BENCHMARK_SYMBOL
        
        # Results storage
        self.backtest_results = {}
        self.performance_metrics = {}
        self.trade_history = []
        self.portfolio_history = []
        self.regime_history = []
        
    async def initialize(self):
        """Initialize backtesting engine"""
        logger.info("Initializing Backtest Engine...")
        
        try:
            # Initialize components for backtesting
            self.ai_gate = AIGate(self.config)
            self.portfolio_manager = PortfolioManager(self.config)
            self.regime_detector = MarketRegimeDetector(self.config)
            
            # Initialize database tables for backtest results
            await self._initialize_backtest_tables()
            
            logger.info("Backtest Engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Backtest Engine: {e}")
            
    async def _initialize_backtest_tables(self):
        """Initialize database tables for backtest results"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id SERIAL PRIMARY KEY,
                run_id VARCHAR(50) UNIQUE NOT NULL,
                strategy_name VARCHAR(100) NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                initial_capital DECIMAL(15,2) NOT NULL,
                final_value DECIMAL(15,2),
                total_return DECIMAL(8,4),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(8,4),
                win_rate DECIMAL(8,4),
                profit_factor DECIMAL(8,4),
                total_trades INTEGER,
                parameters JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id SERIAL PRIMARY KEY,
                run_id VARCHAR(50) NOT NULL,
                trade_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                entry_date DATE NOT NULL,
                exit_date DATE,
                entry_price DECIMAL(12,4) NOT NULL,
                exit_price DECIMAL(12,4),
                quantity DECIMAL(15,2) NOT NULL,
                pnl DECIMAL(15,2),
                pnl_pct DECIMAL(8,4),
                hold_days INTEGER,
                regime VARCHAR(20),
                ai_confidence DECIMAL(8,4),
                signal_strength DECIMAL(8,4),
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS backtest_performance (
                id SERIAL PRIMARY KEY,
                run_id VARCHAR(50) NOT NULL,
                date DATE NOT NULL,
                portfolio_value DECIMAL(15,2) NOT NULL,
                benchmark_value DECIMAL(15,2),
                daily_return DECIMAL(8,4),
                benchmark_return DECIMAL(8,4),
                drawdown DECIMAL(8,4),
                num_positions INTEGER,
                cash_balance DECIMAL(15,2),
                regime VARCHAR(20),
                FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            )
            """
        ]
        
        for table_sql in tables:
            await self.db_manager.execute_query(table_sql)
            
    async def run_backtest(
        self, 
        start_date: str, 
        end_date: str, 
        strategy_name: str = "AI_GATE_STRATEGY",
        initial_capital: float = 1000000,
        parameters: Dict = None
    ) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        
        logger.info(f"Starting backtest: {strategy_name} from {start_date} to {end_date}")
        
        try:
            # Setup backtest
            run_id = await self._setup_backtest(
                start_date, end_date, strategy_name, initial_capital, parameters
            )
            
            # Load historical data
            historical_data = await self._load_historical_data(start_date, end_date)
            
            if historical_data.empty:
                raise ValueError("No historical data available for backtest period")
                
            # Run simulation
            await self._run_simulation(run_id, historical_data, parameters or {})
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(run_id)
            
            # Generate detailed analysis
            analysis_results = await self._generate_analysis(run_id, performance_metrics)
            
            # Store results
            await self._store_backtest_results(run_id, performance_metrics)
            
            logger.info(f"Backtest completed: {run_id}")
            
            return {
                'run_id': run_id,
                'performance_metrics': performance_metrics,
                'analysis_results': analysis_results,
                'trade_count': len(self.trade_history),
                'final_value': performance_metrics.get('final_portfolio_value', initial_capital)
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
            
    async def _setup_backtest(
        self, 
        start_date: str, 
        end_date: str, 
        strategy_name: str, 
        initial_capital: float,
        parameters: Dict
    ) -> str:
        """Setup backtest run"""
        
        from uuid import uuid4
        run_id = str(uuid4())
        
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        
        # Reset state
        self.trade_history = []
        self.portfolio_history = []
        self.regime_history = []
        
        # Initialize portfolio for backtesting
        self.portfolio_manager.cash_balance = initial_capital
        self.portfolio_manager.initial_capital = initial_capital
        self.portfolio_manager.total_portfolio_value = initial_capital
        self.portfolio_manager.current_positions = {}
        
        # Store run metadata
        insert_query = """
        INSERT INTO backtest_runs 
        (run_id, strategy_name, start_date, end_date, initial_capital, parameters)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        await self.db_manager.execute_query(
            insert_query, 
            (run_id, strategy_name, start_date, end_date, initial_capital, str(parameters))
        )
        
        return run_id
        
    async def _load_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting"""
        
        try:
            # Load market data with features
            query = """
            SELECT 
                md.date, md.symbol, md.open, md.high, md.low, md.close, md.volume,
                tf.feature_rsi, tf.feature_macd, tf.feature_bb_upper, tf.feature_bb_lower,
                tf.feature_atr, tf.feature_volume_ratio, tf.feature_momentum,
                tf.feature_relative_strength, tf.feature_volatility,
                sr.score as screening_score, sr.technical_score, sr.momentum_score
            FROM market_data md
            LEFT JOIN trade_features tf ON md.symbol = tf.symbol AND md.date = tf.date
            LEFT JOIN screening_results sr ON md.symbol = sr.symbol AND md.date = sr.date
            WHERE md.date >= %s AND md.date <= %s
            ORDER BY md.date, md.symbol
            """
            
            data = await self.db_manager.fetch_dataframe(query, (start_date, end_date))
            
            logger.info(f"Loaded {len(data)} data points for backtesting")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
            
    async def _run_simulation(self, run_id: str, data: pd.DataFrame, parameters: Dict):
        """Run backtesting simulation"""
        
        logger.info("Running backtesting simulation...")
        
        try:
            # Group data by date
            dates = sorted(data['date'].unique())
            
            for i, current_date in enumerate(dates):
                current_date_data = data[data['date'] == current_date]
                
                # Initialize AI components with historical data up to current date
                historical_subset = data[data['date'] <= current_date]
                
                if len(historical_subset) < 50:  # Need minimum data for AI
                    continue
                    
                # Update market regime
                regime_info = await self._update_regime_for_date(current_date, historical_subset)
                
                # Generate signals for current date
                signals = await self._generate_signals_for_date(current_date_data, historical_subset)
                
                # Process signals through AI gate
                approved_trades = []
                
                for signal in signals:
                    # Simulate AI decision
                    ai_decision = await self._simulate_ai_decision(signal, regime_info, historical_subset)
                    
                    if ai_decision['approved']:
                        approved_trades.append((signal, ai_decision))
                        
                # Execute approved trades
                for signal, ai_decision in approved_trades:
                    await self._execute_backtest_trade(run_id, signal, ai_decision, current_date)
                    
                # Update portfolio values
                await self._update_portfolio_for_date(current_date, current_date_data)
                
                # Record daily performance
                await self._record_daily_performance(run_id, current_date, regime_info)
                
                # Check stop losses and take profits
                await self._check_exit_conditions(run_id, current_date, current_date_data)
                
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(dates)} days")
                    
            logger.info("Simulation completed")
            
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            raise
            
    async def _update_regime_for_date(self, current_date: pd.Timestamp, historical_data: pd.DataFrame) -> Dict:
        """Update market regime for specific date"""
        
        try:
            # Get market data for regime detection
            market_data = historical_data[
                historical_data['symbol'] == self.benchmark_symbol
            ].tail(60)  # Last 60 days
            
            if len(market_data) < 20:
                return {'regime_type': 'SIDEWAYS', 'confidence': 0.5}
                
            # Simulate regime detection
            returns = market_data['close'].pct_change().dropna()
            
            if len(returns) == 0:
                return {'regime_type': 'SIDEWAYS', 'confidence': 0.5}
                
            # Simple regime classification
            avg_return = returns.mean()
            volatility = returns.std()
            
            if avg_return > 0.001 and volatility < 0.02:
                regime_type = 'BULL'
                confidence = 0.8
            elif avg_return < -0.001 and volatility > 0.025:
                regime_type = 'BEAR'
                confidence = 0.8
            else:
                regime_type = 'SIDEWAYS'
                confidence = 0.6
                
            regime_info = {
                'regime_type': regime_type,
                'confidence': confidence,
                'date': current_date
            }
            
            self.regime_history.append(regime_info)
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error updating regime for {current_date}: {e}")
            return {'regime_type': 'SIDEWAYS', 'confidence': 0.5}
            
    async def _generate_signals_for_date(self, current_data: pd.DataFrame, historical_data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals for current date"""
        
        signals = []
        
        try:
            # Filter stocks with good screening scores
            good_stocks = current_data[
                (current_data['screening_score'] >= 60) |
                (current_data['technical_score'] >= 70)
            ]
            
            for _, stock_data in good_stocks.iterrows():
                # Generate signal based on technical conditions
                signal = await self._generate_stock_signal(stock_data, historical_data)
                
                if signal:
                    signals.append(signal)
                    
            return signals[:5]  # Limit to top 5 signals per day
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
            
    async def _generate_stock_signal(self, stock_data: pd.Series, historical_data: pd.DataFrame) -> Optional[Dict]:
        """Generate signal for individual stock"""
        
        try:
            symbol = stock_data['symbol']
            
            # Get stock's historical data
            stock_history = historical_data[historical_data['symbol'] == symbol].tail(20)
            
            if len(stock_history) < 10:
                return None
                
            # Check technical conditions
            rsi = stock_data.get('feature_rsi', 50)
            macd = stock_data.get('feature_macd', 0)
            volume_ratio = stock_data.get('feature_volume_ratio', 1)
            momentum = stock_data.get('feature_momentum', 0)
            
            # Signal conditions
            buy_signals = 0
            
            if 30 < rsi < 70:  # RSI in good range
                buy_signals += 1
            if macd > 0:  # Positive MACD
                buy_signals += 1
            if volume_ratio > 1.2:  # Above average volume
                buy_signals += 1
            if momentum > 1:  # Positive momentum
                buy_signals += 1
                
            if buy_signals >= 3:
                signal_strength = buy_signals / 4
                
                signal = {
                    'symbol': symbol,
                    'signal_type': 'BUY',
                    'signal_strength': signal_strength,
                    'current_price': stock_data['close'],
                    'volume': stock_data['volume'],
                    'features': {
                        'feature_rsi': rsi,
                        'feature_macd': macd,
                        'feature_volume_ratio': volume_ratio,
                        'feature_momentum': momentum
                    },
                    'technical_indicators': {
                        'rsi': rsi,
                        'macd': macd,
                        'volume_ratio': volume_ratio
                    }
                }
                
                return signal
                
        except Exception as e:
            logger.error(f"Error generating stock signal: {e}")
            
        return None
        
    async def _simulate_ai_decision(self, signal: Dict, regime_info: Dict, historical_data: pd.DataFrame) -> Dict:
        """Simulate AI gate decision"""
        
        try:
            # Simulate ensemble model prediction
            features = signal['features']
            
            # Simple decision logic for backtesting
            confidence_factors = []
            
            # Technical confidence
            rsi = features.get('feature_rsi', 50)
            if 40 <= rsi <= 60:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
                
            # Signal strength confidence
            signal_strength = signal.get('signal_strength', 0)
            confidence_factors.append(signal_strength)
            
            # Regime confidence
            regime_confidence = regime_info.get('confidence', 0.5)
            if regime_info.get('regime_type') == 'BULL':
                confidence_factors.append(regime_confidence * 1.2)
            else:
                confidence_factors.append(regime_confidence * 0.8)
                
            # Overall confidence
            overall_confidence = np.mean(confidence_factors)
            
            # Decision threshold
            approved = overall_confidence >= self.config.MIN_CONFIDENCE_THRESHOLD
            
            # Position sizing
            position_size = 0
            if approved:
                base_size = self.config.MAX_POSITION_SIZE
                position_size = base_size * (overall_confidence / 1.0)
                
            # Risk management
            current_price = signal['current_price']
            stop_loss = current_price * 0.98  # 2% stop loss
            take_profit = current_price * 1.04  # 4% take profit
            
            decision = {
                'approved': approved,
                'confidence': overall_confidence,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'rationale': f"AI Decision: {overall_confidence:.3f} confidence"
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"Error simulating AI decision: {e}")
            return {
                'approved': False,
                'confidence': 0,
                'position_size': 0,
                'rationale': f"Error: {str(e)}"
            }
            
    async def _execute_backtest_trade(self, run_id: str, signal: Dict, ai_decision: Dict, trade_date: pd.Timestamp):
        """Execute trade in backtest"""
        
        try:
            symbol = signal['symbol']
            entry_price = signal['current_price']
            position_size = ai_decision['position_size']
            
            # Calculate quantity
            portfolio_value = self.portfolio_manager.total_portfolio_value
            position_value = portfolio_value * position_size
            quantity = position_value / entry_price
            
            # Check if we have enough cash
            if position_value > self.portfolio_manager.cash_balance:
                return  # Skip trade
                
            # Create trade record
            trade_id = f"BT_{run_id}_{len(self.trade_history)}"
            
            trade = {
                'trade_id': trade_id,
                'symbol': symbol,
                'entry_date': trade_date,
                'entry_price': entry_price,
                'quantity': quantity,
                'ai_confidence': ai_decision['confidence'],
                'signal_strength': signal['signal_strength'],
                'regime': self.regime_history[-1]['regime_type'] if self.regime_history else 'UNKNOWN',
                'stop_loss': ai_decision.get('stop_loss'),
                'take_profit': ai_decision.get('take_profit'),
                'status': 'OPEN'
            }
            
            # Update portfolio
            self.portfolio_manager.cash_balance -= position_value
            self.portfolio_manager.current_positions[symbol] = {
                'quantity': quantity,
                'avg_price': entry_price,
                'market_value': position_value,
                'unrealized_pnl': 0,
                'entry_date': trade_date,
                'trade_id': trade_id
            }
            
            self.trade_history.append(trade)
            
            # Store in database
            await self._store_backtest_trade(run_id, trade)
            
        except Exception as e:
            logger.error(f"Error executing backtest trade: {e}")
            
    async def _update_portfolio_for_date(self, current_date: pd.Timestamp, current_data: pd.DataFrame):
        """Update portfolio values for current date"""
        
        try:
            # Create price dictionary
            price_data = {}
            for _, row in current_data.iterrows():
                price_data[row['symbol']] = row['close']
                
            # Update portfolio
            await self.portfolio_manager.update_market_values(price_data)
            
        except Exception as e:
            logger.error(f"Error updating portfolio for {current_date}: {e}")
            
    async def _check_exit_conditions(self, run_id: str, current_date: pd.Timestamp, current_data: pd.DataFrame):
        """Check stop loss and take profit conditions"""
        
        try:
            price_data = {}
            for _, row in current_data.iterrows():
                price_data[row['symbol']] = row['close']
                
            positions_to_close = []
            
            for symbol, position in self.portfolio_manager.current_positions.items():
                if symbol not in price_data:
                    continue
                    
                current_price = price_data[symbol]
                entry_price = position['avg_price']
                
                # Find corresponding trade
                trade = None
                for t in self.trade_history:
                    if t['symbol'] == symbol and t['status'] == 'OPEN':
                        trade = t
                        break
                        
                if not trade:
                    continue
                    
                # Check exit conditions
                should_exit = False
                exit_reason = 'UNKNOWN'
                
                # Stop loss
                if trade.get('stop_loss') and current_price <= trade['stop_loss']:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'
                    
                # Take profit
                elif trade.get('take_profit') and current_price >= trade['take_profit']:
                    should_exit = True
                    exit_reason = 'TAKE_PROFIT'
                    
                # Time-based exit (hold for max 10 days)
                elif (current_date - trade['entry_date']).days >= 10:
                    should_exit = True
                    exit_reason = 'TIME_EXIT'
                    
                if should_exit:
                    positions_to_close.append((symbol, current_price, exit_reason, trade))
                    
            # Close positions
            for symbol, exit_price, exit_reason, trade in positions_to_close:
                await self._close_backtest_position(run_id, symbol, exit_price, exit_reason, trade, current_date)
                
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            
    async def _close_backtest_position(self, run_id: str, symbol: str, exit_price: float, exit_reason: str, trade: Dict, exit_date: pd.Timestamp):
        """Close position in backtest"""
        
        try:
            position = self.portfolio_manager.current_positions[symbol]
            quantity = position['quantity']
            entry_price = position['avg_price']
            
            # Calculate P&L
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price / entry_price - 1) * 100
            
            # Update cash
            exit_value = quantity * exit_price
            self.portfolio_manager.cash_balance += exit_value
            
            # Update realized P&L
            self.portfolio_manager.realized_pnl += pnl
            
            # Update trade record
            trade['exit_date'] = exit_date
            trade['exit_price'] = exit_price
            trade['pnl'] = pnl
            trade['pnl_pct'] = pnl_pct
            trade['hold_days'] = (exit_date - trade['entry_date']).days
            trade['status'] = 'CLOSED'
            trade['exit_reason'] = exit_reason
            
            # Remove position
            del self.portfolio_manager.current_positions[symbol]
            
            # Update in database
            await self._update_backtest_trade(run_id, trade)
            
        except Exception as e:
            logger.error(f"Error closing backtest position: {e}")
            
    async def _record_daily_performance(self, run_id: str, current_date: pd.Timestamp, regime_info: Dict):
        """Record daily performance"""
        
        try:
            portfolio_value = self.portfolio_manager.total_portfolio_value
            
            # Calculate daily return
            if self.portfolio_history:
                prev_value = self.portfolio_history[-1]['portfolio_value']
                daily_return = (portfolio_value / prev_value - 1) * 100
            else:
                daily_return = 0
                
            # Calculate drawdown
            peak_value = max([p['portfolio_value'] for p in self.portfolio_history] + [self.initial_capital])
            drawdown = (peak_value - portfolio_value) / peak_value * 100
            
            performance_record = {
                'date': current_date,
                'portfolio_value': portfolio_value,
                'daily_return': daily_return,
                'drawdown': drawdown,
                'num_positions': len(self.portfolio_manager.current_positions),
                'cash_balance': self.portfolio_manager.cash_balance,
                'regime': regime_info.get('regime_type', 'UNKNOWN')
            }
            
            self.portfolio_history.append(performance_record)
            
            # Store in database
            await self._store_daily_performance(run_id, performance_record)
            
        except Exception as e:
            logger.error(f"Error recording daily performance: {e}")
            
    async def _calculate_performance_metrics(self, run_id: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        try:
            if not self.portfolio_history:
                return {}
                
            # Basic metrics
            final_value = self.portfolio_history[-1]['portfolio_value']
            total_return = (final_value / self.initial_capital - 1) * 100
            
            # Daily returns
            daily_returns = [p['daily_return'] for p in self.portfolio_history[1:]]
            
            if not daily_returns:
                return {'total_return': total_return, 'final_portfolio_value': final_value}
                
            # Risk metrics
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = (np.mean(daily_returns) * 252) / volatility if volatility > 0 else 0
            
            # Drawdown
            max_drawdown = max([p['drawdown'] for p in self.portfolio_history])
            
            # Trade metrics
            completed_trades = [t for t in self.trade_history if t['status'] == 'CLOSED']
            
            if completed_trades:
                winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in completed_trades if t.get('pnl', 0) <= 0]
                
                win_rate = len(winning_trades) / len(completed_trades) * 100
                
                avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                
                avg_hold_days = np.mean([t['hold_days'] for t in completed_trades])
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                avg_hold_days = 0
                
            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino ratio
            negative_returns = [r for r in daily_returns if r < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0
            sortino_ratio = (np.mean(daily_returns) * 252) / downside_deviation if downside_deviation > 0 else 0
            
            metrics = {
                'final_portfolio_value': final_value,
                'total_return': total_return,
                'annualized_return': (final_value / self.initial_capital) ** (252 / len(self.portfolio_history)) - 1 if len(self.portfolio_history) > 0 else 0,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(completed_trades),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_hold_days': avg_hold_days,
                'best_trade': max([t.get('pnl', 0) for t in completed_trades]) if completed_trades else 0,
                'worst_trade': min([t.get('pnl', 0) for t in completed_trades]) if completed_trades else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
            
    async def _generate_analysis(self, run_id: str, metrics: Dict) -> Dict:
        """Generate detailed backtest analysis"""
        
        try:
            analysis = {
                'summary': {
                    'performance_grade': self._grade_performance(metrics),
                    'key_strengths': [],
                    'key_weaknesses': [],
                    'recommendations': []
                },
                'regime_analysis': await self._analyze_regime_performance(),
                'trade_analysis': await self._analyze_trade_patterns(),
                'risk_analysis': await self._analyze_risk_metrics(metrics)
            }
            
            # Performance grading
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = metrics.get('max_drawdown', 100)
            win_rate = metrics.get('win_rate', 0)
            
            # Identify strengths and weaknesses
            if sharpe > 1.5:
                analysis['summary']['key_strengths'].append("Excellent risk-adjusted returns")
            if max_dd < 10:
                analysis['summary']['key_strengths'].append("Low maximum drawdown")
            if win_rate > 60:
                analysis['summary']['key_strengths'].append("High win rate")
                
            if sharpe < 0.5:
                analysis['summary']['key_weaknesses'].append("Poor risk-adjusted returns")
            if max_dd > 20:
                analysis['summary']['key_weaknesses'].append("High maximum drawdown")
            if win_rate < 40:
                analysis['summary']['key_weaknesses'].append("Low win rate")
                
            # Recommendations
            if max_dd > 15:
                analysis['summary']['recommendations'].append("Consider tighter risk management")
            if win_rate < 50:
                analysis['summary']['recommendations'].append("Improve signal quality and entry criteria")
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return {'error': str(e)}
            
    def _grade_performance(self, metrics: Dict) -> str:
        """Grade overall performance"""
        
        try:
            score = 0
            
            # Sharpe ratio scoring
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe > 2:
                score += 30
            elif sharpe > 1:
                score += 20
            elif sharpe > 0.5:
                score += 10
                
            # Return scoring
            total_return = metrics.get('total_return', 0)
            if total_return > 20:
                score += 25
            elif total_return > 10:
                score += 15
            elif total_return > 5:
                score += 10
                
            # Drawdown scoring
            max_dd = metrics.get('max_drawdown', 100)
            if max_dd < 5:
                score += 25
            elif max_dd < 10:
                score += 20
            elif max_dd < 15:
                score += 15
            elif max_dd < 25:
                score += 10
                
            # Win rate scoring
            win_rate = metrics.get('win_rate', 0)
            if win_rate > 70:
                score += 20
            elif win_rate > 60:
                score += 15
            elif win_rate > 50:
                score += 10
                
            # Grade assignment
            if score >= 80:
                return 'A+'
            elif score >= 70:
                return 'A'
            elif score >= 60:
                return 'B+'
            elif score >= 50:
                return 'B'
            elif score >= 40:
                return 'C+'
            elif score >= 30:
                return 'C'
            else:
                return 'D'
                
        except Exception:
            return 'N/A'
            
    async def _analyze_regime_performance(self) -> Dict:
        """Analyze performance by market regime"""
        
        try:
            regime_performance = {}
            
            for regime_type in ['BULL', 'BEAR', 'SIDEWAYS']:
                regime_trades = [
                    t for t in self.trade_history 
                    if t.get('regime') == regime_type and t['status'] == 'CLOSED'
                ]
                
                if regime_trades:
                    total_pnl = sum(t.get('pnl', 0) for t in regime_trades)
                    winning_trades = sum(1 for t in regime_trades if t.get('pnl', 0) > 0)
                    win_rate = winning_trades / len(regime_trades) * 100
                    
                    regime_performance[regime_type] = {
                        'total_trades': len(regime_trades),
                        'total_pnl': total_pnl,
                        'win_rate': win_rate,
                        'avg_pnl': total_pnl / len(regime_trades)
                    }
                    
            return regime_performance
            
        except Exception as e:
            logger.error(f"Error analyzing regime performance: {e}")
            return {}
            
    async def _analyze_trade_patterns(self) -> Dict:
        """Analyze trade patterns"""
        
        try:
            completed_trades = [t for t in self.trade_history if t['status'] == 'CLOSED']
            
            if not completed_trades:
                return {}
                
            # Hold time analysis
            hold_days = [t['hold_days'] for t in completed_trades]
            
            # Performance by hold time
            short_term = [t for t in completed_trades if t['hold_days'] <= 2]
            medium_term = [t for t in completed_trades if 2 < t['hold_days'] <= 5]
            long_term = [t for t in completed_trades if t['hold_days'] > 5]
            
            analysis = {
                'hold_time_stats': {
                    'avg_hold_days': np.mean(hold_days),
                    'median_hold_days': np.median(hold_days),
                    'max_hold_days': max(hold_days),
                    'min_hold_days': min(hold_days)
                },
                'performance_by_hold_time': {
                    'short_term': {
                        'count': len(short_term),
                        'avg_pnl': np.mean([t.get('pnl', 0) for t in short_term]) if short_term else 0,
                        'win_rate': sum(1 for t in short_term if t.get('pnl', 0) > 0) / len(short_term) * 100 if short_term else 0
                    },
                    'medium_term': {
                        'count': len(medium_term),
                        'avg_pnl': np.mean([t.get('pnl', 0) for t in medium_term]) if medium_term else 0,
                        'win_rate': sum(1 for t in medium_term if t.get('pnl', 0) > 0) / len(medium_term) * 100 if medium_term else 0
                    },
                    'long_term': {
                        'count': len(long_term),
                        'avg_pnl': np.mean([t.get('pnl', 0) for t in long_term]) if long_term else 0,
                        'win_rate': sum(1 for t in long_term if t.get('pnl', 0) > 0) / len(long_term) * 100 if long_term else 0
                    }
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {e}")
            return {}
            
    async def _analyze_risk_metrics(self, metrics: Dict) -> Dict:
        """Analyze risk metrics"""
        
        try:
            analysis = {
                'risk_assessment': 'MODERATE',
                'risk_factors': [],
                'risk_recommendations': []
            }
            
            # Risk assessment
            max_dd = metrics.get('max_drawdown', 0)
            volatility = metrics.get('volatility', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            
            if max_dd > 20 or volatility > 0.3 or sharpe < 0.5:
                analysis['risk_assessment'] = 'HIGH'
            elif max_dd < 10 and volatility < 0.2 and sharpe > 1:
                analysis['risk_assessment'] = 'LOW'
                
            # Risk factors
            if max_dd > 15:
                analysis['risk_factors'].append(f"High maximum drawdown: {max_dd:.1f}%")
            if volatility > 0.25:
                analysis['risk_factors'].append(f"High volatility: {volatility:.1f}%")
            if sharpe < 1:
                analysis['risk_factors'].append(f"Low Sharpe ratio: {sharpe:.2f}")
                
            # Recommendations
            if max_dd > 15:
                analysis['risk_recommendations'].append("Implement tighter stop losses")
            if volatility > 0.25:
                analysis['risk_recommendations'].append("Reduce position sizes")
            if sharpe < 1:
                analysis['risk_recommendations'].append("Improve signal quality")
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk metrics: {e}")
            return {}
            
    async def _store_backtest_trade(self, run_id: str, trade: Dict):
        """Store backtest trade"""
        
        try:
            insert_query = """
            INSERT INTO backtest_trades 
            (run_id, trade_id, symbol, entry_date, entry_price, quantity,
             regime, ai_confidence, signal_strength)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                run_id,
                trade['trade_id'],
                trade['symbol'],
                trade['entry_date'],
                trade['entry_price'],
                trade['quantity'],
                trade.get('regime'),
                trade.get('ai_confidence', 0),
                trade.get('signal_strength', 0)
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing backtest trade: {e}")
            
    async def _update_backtest_trade(self, run_id: str, trade: Dict):
        """Update backtest trade"""
        
        try:
            update_query = """
            UPDATE backtest_trades SET
                exit_date = %s,
                exit_price = %s,
                pnl = %s,
                pnl_pct = %s,
                hold_days = %s
            WHERE run_id = %s AND trade_id = %s
            """
            
            values = (
                trade.get('exit_date'),
                trade.get('exit_price'),
                trade.get('pnl'),
                trade.get('pnl_pct'),
                trade.get('hold_days'),
                run_id,
                trade['trade_id']
            )
            
            await self.db_manager.execute_query(update_query, values)
            
        except Exception as e:
            logger.error(f"Error updating backtest trade: {e}")
            
    async def _store_daily_performance(self, run_id: str, performance: Dict):
        """Store daily performance"""
        
        try:
            insert_query = """
            INSERT INTO backtest_performance 
            (run_id, date, portfolio_value, daily_return, drawdown,
             num_positions, cash_balance, regime)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                run_id,
                performance['date'],
                performance['portfolio_value'],
                performance['daily_return'],
                performance['drawdown'],
                performance['num_positions'],
                performance['cash_balance'],
                performance['regime']
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing daily performance: {e}")
            
    async def _store_backtest_results(self, run_id: str, metrics: Dict):
        """Store final backtest results"""
        
        try:
            update_query = """
            UPDATE backtest_runs SET
                final_value = %s,
                total_return = %s,
                sharpe_ratio = %s,
                max_drawdown = %s,
                win_rate = %s,
                profit_factor = %s,
                total_trades = %s
            WHERE run_id = %s
            """
            
            values = (
                metrics.get('final_portfolio_value'),
                metrics.get('total_return'),
                metrics.get('sharpe_ratio'),
                metrics.get('max_drawdown'),
                metrics.get('win_rate'),
                metrics.get('profit_factor'),
                metrics.get('total_trades'),
                run_id
            )
            
            await self.db_manager.execute_query(update_query, values)
            
        except Exception as e:
            logger.error(f"Error storing backtest results: {e}")
            
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get backtest engine summary"""
        
        return {
            'total_trades': len(self.trade_history),
            'completed_trades': len([t for t in self.trade_history if t['status'] == 'CLOSED']),
            'portfolio_snapshots': len(self.portfolio_history),
            'regime_history': len(self.regime_history),
            'current_portfolio_value': self.portfolio_manager.total_portfolio_value if self.portfolio_manager else 0,
            'performance_metrics': self.performance_metrics
        }
