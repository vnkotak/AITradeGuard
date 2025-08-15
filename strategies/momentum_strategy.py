"""
Momentum Trading Strategy
Implementation of a sophisticated momentum-based trading strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio

from strategies.base_strategy import BaseStrategy
from config import Config

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    """
    Advanced momentum trading strategy using multiple timeframes and indicators
    """
    
    def __init__(self, config: Config = None):
        super().__init__("Momentum Strategy", config)
        
        # Strategy-specific parameters
        self.default_parameters = {
            'short_ma_period': 10,
            'long_ma_period': 30,
            'momentum_period': 14,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_threshold': 1.5,  # Volume should be 1.5x average
            'momentum_threshold': 2.0,  # Minimum momentum in %
            'confidence_threshold': 0.7,
            'max_positions': 5,
            'holding_period_min': 1,  # Minimum holding period in days
            'holding_period_max': 10,  # Maximum holding period in days
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_ratio': 2.5,  # Risk:Reward ratio
            'regime_adjustment': True,
            'volatility_adjustment': True
        }
        
        self.parameters = self.default_parameters.copy()
        
        # Strategy state
        self.current_momentum_stocks = []
        self.blacklist = set()  # Stocks to avoid
        self.momentum_history = {}
        
    async def _initialize_strategy(self):
        """Initialize momentum strategy specific components"""
        
        try:
            # Initialize momentum tracking
            await self._initialize_momentum_tracking()
            
            # Load strategy-specific data
            await self._load_momentum_universe()
            
            logger.info("Momentum strategy initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing momentum strategy: {e}")
            raise
            
    async def _initialize_momentum_tracking(self):
        """Initialize momentum tracking system"""
        
        try:
            # Create tables for momentum tracking if needed
            momentum_table = """
            CREATE TABLE IF NOT EXISTS momentum_signals (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                momentum_score DECIMAL(8,4),
                volume_score DECIMAL(8,4),
                technical_score DECIMAL(8,4),
                overall_score DECIMAL(8,4),
                signal_strength DECIMAL(8,4),
                regime VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, symbol)
            )
            """
            
            await self.db_manager.execute_query(momentum_table)
            
        except Exception as e:
            logger.error(f"Error initializing momentum tracking: {e}")
            
    async def _load_momentum_universe(self):
        """Load universe of stocks for momentum analysis"""
        
        try:
            # Get active, liquid stocks from recent screening
            query = """
            SELECT DISTINCT symbol 
            FROM screening_results 
            WHERE date >= %s 
            AND score >= 60 
            AND volume_score >= 50
            ORDER BY score DESC
            LIMIT 200
            """
            
            cutoff_date = datetime.now() - timedelta(days=7)
            stocks = await self.db_manager.fetch_all(query, (cutoff_date,))
            
            self.momentum_universe = [stock['symbol'] for stock in stocks]
            
            logger.info(f"Loaded {len(self.momentum_universe)} stocks for momentum analysis")
            
        except Exception as e:
            logger.error(f"Error loading momentum universe: {e}")
            self.momentum_universe = []
            
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate momentum-based trading signals"""
        
        try:
            signals = []
            
            # Filter market data to momentum universe
            filtered_data = {
                symbol: data for symbol, data in market_data.items() 
                if symbol in self.momentum_universe
            }
            
            # Analyze each stock for momentum
            for symbol, data in filtered_data.items():
                if len(data) < max(self.parameters['long_ma_period'], self.parameters['momentum_period']) + 10:
                    continue  # Need sufficient data
                    
                # Skip if in blacklist
                if symbol in self.blacklist:
                    continue
                    
                # Skip if already have maximum positions
                if len(self.positions) >= self.parameters['max_positions']:
                    break
                    
                # Calculate momentum indicators
                momentum_data = await self._calculate_momentum_indicators(data)
                
                if momentum_data is None:
                    continue
                    
                # Generate signal
                signal = await self._evaluate_momentum_signal(symbol, momentum_data)
                
                if signal and self.validate_signal(signal):
                    signals.append(signal)
                    
            # Rank signals by strength and return top ones
            signals = sorted(signals, key=lambda x: x['signal_strength'], reverse=True)
            
            # Store momentum signals
            await self._store_momentum_signals(signals)
            
            return signals[:3]  # Return top 3 signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return []
            
    async def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Optional[Dict]:
        """Calculate momentum indicators for a stock"""
        
        try:
            df = data.copy()
            
            # Calculate technical indicators
            df = await self.technical_indicators.calculate_all_indicators(df)
            
            if df.empty:
                return None
                
            # Get latest values
            latest = df.iloc[-1]
            
            # Price momentum
            momentum_period = self.parameters['momentum_period']
            current_price = latest['close']
            past_price = df['close'].iloc[-momentum_period-1] if len(df) > momentum_period else df['close'].iloc[0]
            
            price_momentum = ((current_price / past_price) - 1) * 100
            
            # Volume momentum
            current_volume = latest['volume']
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_momentum = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Technical momentum indicators
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            
            # Moving averages
            short_ma = df['close'].rolling(self.parameters['short_ma_period']).mean().iloc[-1]
            long_ma = df['close'].rolling(self.parameters['long_ma_period']).mean().iloc[-1]
            
            # ATR for volatility
            atr = latest.get('atr', 0)
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
            
            # Trend strength
            trend_strength = ((short_ma / long_ma) - 1) * 100 if long_ma > 0 else 0
            
            # Price position in recent range
            high_20 = df['high'].rolling(20).max().iloc[-1]
            low_20 = df['low'].rolling(20).min().iloc[-1]
            price_position = ((current_price - low_20) / (high_20 - low_20)) * 100 if high_20 > low_20 else 50
            
            return {
                'symbol': latest.get('symbol', ''),
                'current_price': current_price,
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'short_ma': short_ma,
                'long_ma': long_ma,
                'trend_strength': trend_strength,
                'atr': atr,
                'atr_pct': atr_pct,
                'price_position': price_position,
                'volume': current_volume,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return None
            
    async def _evaluate_momentum_signal(self, symbol: str, momentum_data: Dict) -> Optional[Dict]:
        """Evaluate momentum data and generate signal"""
        
        try:
            # Momentum criteria
            price_momentum = momentum_data['price_momentum']
            volume_momentum = momentum_data['volume_momentum']
            rsi = momentum_data['rsi']
            trend_strength = momentum_data['trend_strength']
            price_position = momentum_data['price_position']
            
            # Signal strength components
            momentum_score = 0
            
            # 1. Price momentum score
            if price_momentum >= self.parameters['momentum_threshold']:
                momentum_score += 25
            elif price_momentum >= self.parameters['momentum_threshold'] / 2:
                momentum_score += 15
                
            # 2. Volume confirmation score
            if volume_momentum >= self.parameters['volume_threshold']:
                momentum_score += 20
            elif volume_momentum >= 1.2:
                momentum_score += 10
                
            # 3. Technical indicators score
            if trend_strength > 2:  # Strong uptrend
                momentum_score += 20
            elif trend_strength > 0:  # Uptrend
                momentum_score += 10
                
            # 4. RSI position score
            if 40 <= rsi <= 65:  # Sweet spot for momentum
                momentum_score += 15
            elif rsi < 30:  # Oversold, potential reversal
                momentum_score += 10
                
            # 5. Price position score
            if price_position >= 80:  # Near highs
                momentum_score += 10
            elif price_position >= 60:
                momentum_score += 5
                
            # 6. MACD confirmation
            if momentum_data['macd'] > momentum_data['macd_signal']:
                momentum_score += 10
                
            # Normalize to 0-1
            signal_strength = min(momentum_score / 100, 1.0)
            
            # Generate signal only if above threshold
            if signal_strength < 0.6:
                return None
                
            # Determine signal type
            signal_type = 'BUY'  # Momentum strategy is primarily long-only
            
            # Calculate position size
            base_position_size = self.max_position_size
            
            # Adjust for signal strength
            position_size = base_position_size * signal_strength
            
            # Adjust for volatility
            if self.parameters['volatility_adjustment']:
                volatility_adjustment = max(0.5, min(1.5, 2 / (momentum_data['atr_pct'] + 1)))
                position_size *= volatility_adjustment
                
            # Risk management levels
            current_price = momentum_data['current_price']
            atr = momentum_data['atr']
            
            stop_loss = current_price - (atr * self.parameters['stop_loss_atr_multiplier'])
            take_profit = current_price + (atr * self.parameters['stop_loss_atr_multiplier'] * self.parameters['take_profit_ratio'])
            
            # Generate rationale
            rationale = self._generate_momentum_rationale(momentum_data, signal_strength)
            
            signal = {
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'confidence': signal_strength,  # For momentum, strength = confidence
                'rationale': rationale,
                'target_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'metadata': {
                    'strategy': 'momentum',
                    'price_momentum': price_momentum,
                    'volume_momentum': volume_momentum,
                    'trend_strength': trend_strength,
                    'rsi': rsi,
                    'momentum_score': momentum_score,
                    'expected_hold_days': np.random.randint(
                        self.parameters['holding_period_min'],
                        self.parameters['holding_period_max']
                    )
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error evaluating momentum signal for {symbol}: {e}")
            return None
            
    def _generate_momentum_rationale(self, momentum_data: Dict, signal_strength: float) -> str:
        """Generate human-readable rationale for momentum signal"""
        
        try:
            rationale_parts = []
            
            price_momentum = momentum_data['price_momentum']
            volume_momentum = momentum_data['volume_momentum']
            trend_strength = momentum_data['trend_strength']
            rsi = momentum_data['rsi']
            
            # Price momentum
            if price_momentum >= 5:
                rationale_parts.append(f"Strong price momentum: {price_momentum:.1f}%")
            elif price_momentum >= 2:
                rationale_parts.append(f"Positive momentum: {price_momentum:.1f}%")
                
            # Volume confirmation
            if volume_momentum >= 2:
                rationale_parts.append(f"High volume confirmation: {volume_momentum:.1f}x")
            elif volume_momentum >= 1.5:
                rationale_parts.append(f"Volume support: {volume_momentum:.1f}x")
                
            # Trend
            if trend_strength > 3:
                rationale_parts.append("Strong uptrend")
            elif trend_strength > 1:
                rationale_parts.append("Uptrend")
                
            # RSI
            if 40 <= rsi <= 65:
                rationale_parts.append("RSI in momentum zone")
            elif rsi < 35:
                rationale_parts.append("Oversold with reversal potential")
                
            # Overall assessment
            if signal_strength >= 0.8:
                rationale_parts.insert(0, "🔥 Strong momentum signal:")
            elif signal_strength >= 0.7:
                rationale_parts.insert(0, "📈 Good momentum signal:")
            else:
                rationale_parts.insert(0, "⚡ Momentum signal:")
                
            return " ".join(rationale_parts)
            
        except Exception:
            return "Momentum-based trading signal"
            
    async def update_positions(self, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Update existing momentum positions"""
        
        try:
            position_updates = []
            
            for symbol, position in self.positions.items():
                if symbol not in current_data:
                    continue
                    
                data = current_data[symbol]
                if data.empty:
                    continue
                    
                current_price = data['close'].iloc[-1]
                entry_date = position.get('entry_date', datetime.now())
                hold_days = (datetime.now() - entry_date).days
                
                # Check exit conditions
                exit_signal = await self._check_momentum_exit(symbol, position, current_price, hold_days)
                
                if exit_signal:
                    position_updates.append(exit_signal)
                    
            return position_updates
            
        except Exception as e:
            logger.error(f"Error updating momentum positions: {e}")
            return []
            
    async def _check_momentum_exit(self, symbol: str, position: Dict, current_price: float, hold_days: int) -> Optional[Dict]:
        """Check if momentum position should be exited"""
        
        try:
            entry_price = position.get('avg_price', current_price)
            stop_loss = position.get('stop_loss', 0)
            take_profit = position.get('take_profit', 0)
            
            # Stop loss
            if stop_loss > 0 and current_price <= stop_loss:
                return {
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'price': current_price,
                    'reason': 'Stop loss triggered',
                    'exit_type': 'STOP_LOSS'
                }
                
            # Take profit
            if take_profit > 0 and current_price >= take_profit:
                return {
                    'symbol': symbol,
                    'action': 'TAKE_PROFIT',
                    'price': current_price,
                    'reason': 'Take profit target reached',
                    'exit_type': 'TAKE_PROFIT'
                }
                
            # Maximum holding period
            if hold_days >= self.parameters['holding_period_max']:
                return {
                    'symbol': symbol,
                    'action': 'TIME_EXIT',
                    'price': current_price,
                    'reason': 'Maximum holding period reached',
                    'exit_type': 'TIME_EXIT'
                }
                
            # Momentum deterioration check
            current_return = (current_price / entry_price - 1) * 100
            
            # If position is profitable but momentum is weakening
            if current_return > 2 and hold_days >= self.parameters['holding_period_min']:
                # Get recent momentum data
                try:
                    recent_data = await self._get_recent_momentum_data(symbol)
                    if recent_data and recent_data['price_momentum'] < 1:  # Momentum weakening
                        return {
                            'symbol': symbol,
                            'action': 'MOMENTUM_EXIT',
                            'price': current_price,
                            'reason': 'Momentum deteriorating, taking profits',
                            'exit_type': 'MOMENTUM_EXIT'
                        }
                except Exception:
                    pass
                    
            return None
            
        except Exception as e:
            logger.error(f"Error checking momentum exit for {symbol}: {e}")
            return None
            
    async def _get_recent_momentum_data(self, symbol: str) -> Optional[Dict]:
        """Get recent momentum data for a symbol"""
        
        try:
            query = """
            SELECT momentum_score, volume_score, technical_score, overall_score
            FROM momentum_signals 
            WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
            """
            
            result = await self.db_manager.fetch_one(query, (symbol,))
            
            if result:
                return {
                    'price_momentum': result['momentum_score'],
                    'volume_momentum': result['volume_score'],
                    'technical_score': result['technical_score'],
                    'overall_score': result['overall_score']
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting recent momentum data: {e}")
            return None
            
    def calculate_position_size(self, signal: Dict, portfolio_value: float) -> float:
        """Calculate position size for momentum signal"""
        
        try:
            base_size = signal.get('position_size', self.max_position_size)
            signal_strength = signal.get('signal_strength', 0.5)
            
            # Adjust for signal strength
            adjusted_size = base_size * signal_strength
            
            # Adjust for portfolio heat (number of positions)
            position_count = len(self.positions)
            heat_adjustment = max(0.5, 1 - (position_count / self.parameters['max_positions']) * 0.3)
            
            final_size = adjusted_size * heat_adjustment
            
            # Ensure within limits
            return min(final_size, self.max_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.max_position_size * 0.5
            
    async def _store_momentum_signals(self, signals: List[Dict]):
        """Store momentum signals in database"""
        
        try:
            if not signals:
                return
                
            today = datetime.now().date()
            records = []
            
            for signal in signals:
                metadata = signal.get('metadata', {})
                
                records.append((
                    today,
                    signal['symbol'],
                    metadata.get('price_momentum', 0),
                    metadata.get('volume_momentum', 0),
                    metadata.get('momentum_score', 0),
                    signal['signal_strength'],
                    signal['signal_strength'],
                    'UNKNOWN'  # regime
                ))
                
            if records:
                insert_query = """
                INSERT INTO momentum_signals 
                (date, symbol, momentum_score, volume_score, technical_score, 
                 overall_score, signal_strength, regime)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, symbol) DO UPDATE SET
                    momentum_score = EXCLUDED.momentum_score,
                    volume_score = EXCLUDED.volume_score,
                    technical_score = EXCLUDED.technical_score,
                    overall_score = EXCLUDED.overall_score,
                    signal_strength = EXCLUDED.signal_strength
                """
                
                await self.db_manager.execute_many(insert_query, records)
                
        except Exception as e:
            logger.error(f"Error storing momentum signals: {e}")
            
    async def get_momentum_analysis(self) -> Dict[str, Any]:
        """Get comprehensive momentum analysis"""
        
        try:
            analysis = {
                'strategy_name': self.name,
                'active_signals': len(self.current_momentum_stocks),
                'current_positions': len(self.positions),
                'universe_size': len(self.momentum_universe),
                'blacklist_size': len(self.blacklist),
                'parameters': self.parameters,
                'performance_summary': await self._get_momentum_performance_summary()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting momentum analysis: {e}")
            return {'error': str(e)}
            
    async def _get_momentum_performance_summary(self) -> Dict[str, Any]:
        """Get momentum strategy performance summary"""
        
        try:
            # Get recent momentum signals performance
            query = """
            SELECT 
                COUNT(*) as total_signals,
                AVG(signal_strength) as avg_signal_strength,
                MAX(signal_strength) as max_signal_strength
            FROM momentum_signals 
            WHERE date >= %s
            """
            
            cutoff_date = datetime.now() - timedelta(days=30)
            result = await self.db_manager.fetch_one(query, (cutoff_date,))
            
            if result:
                return {
                    'total_signals_30d': result['total_signals'],
                    'avg_signal_strength': float(result['avg_signal_strength']) if result['avg_signal_strength'] else 0,
                    'max_signal_strength': float(result['max_signal_strength']) if result['max_signal_strength'] else 0
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting momentum performance summary: {e}")
            return {}
            
    def add_to_blacklist(self, symbol: str, reason: str = ""):
        """Add symbol to blacklist"""
        
        self.blacklist.add(symbol)
        logger.info(f"Added {symbol} to momentum strategy blacklist: {reason}")
        
    def remove_from_blacklist(self, symbol: str):
        """Remove symbol from blacklist"""
        
        if symbol in self.blacklist:
            self.blacklist.remove(symbol)
            logger.info(f"Removed {symbol} from momentum strategy blacklist")
            
    async def scan_for_momentum_breakouts(self, min_volume_ratio: float = 2.0, min_price_change: float = 3.0) -> List[Dict]:
        """Scan for momentum breakout opportunities"""
        
        try:
            # Get stocks with significant price and volume moves
            query = """
            SELECT DISTINCT m1.symbol, m1.close as current_price,
                   m1.volume as current_volume,
                   m2.close as prev_price,
                   m2.volume as prev_volume,
                   ((m1.close / m2.close) - 1) * 100 as price_change,
                   (m1.volume / m2.volume) as volume_ratio
            FROM market_data m1
            JOIN market_data m2 ON m1.symbol = m2.symbol 
                AND m2.date = m1.date - INTERVAL '1 day'
            WHERE m1.date = CURRENT_DATE
            AND ((m1.close / m2.close) - 1) * 100 >= %s
            AND (m1.volume / m2.volume) >= %s
            ORDER BY price_change DESC, volume_ratio DESC
            LIMIT 20
            """
            
            breakouts = await self.db_manager.fetch_all(query, (min_price_change, min_volume_ratio))
            
            return [dict(breakout) for breakout in breakouts]
            
        except Exception as e:
            logger.error(f"Error scanning for momentum breakouts: {e}")
            return []
