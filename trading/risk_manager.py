"""
Advanced Risk Management System
Implements sophisticated risk controls and position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio

from config import Config
from database.db_manager import DatabaseManager
from utils.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Advanced risk management for trading operations
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        self.statistical_analysis = StatisticalAnalysis()
        
        # Risk parameters
        self.max_portfolio_risk = self.config.MAX_PORTFOLIO_RISK
        self.max_position_size = self.config.MAX_POSITION_SIZE
        self.max_daily_trades = self.config.MAX_DAILY_TRADES
        
        # Risk metrics
        self.current_portfolio_risk = 0
        self.current_positions = {}
        self.daily_trade_count = 0
        self.risk_budget_used = 0
        
        # Risk monitoring
        self.risk_alerts = []
        self.risk_history = []
        
    async def initialize(self):
        """Initialize risk management system"""
        logger.info("Initializing Risk Management System...")
        
        try:
            # Initialize risk tracking tables
            await self._initialize_risk_tables()
            
            # Load current portfolio state
            await self._load_portfolio_state()
            
            # Load risk parameters
            await self._load_risk_parameters()
            
            logger.info("Risk Management System initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Risk Manager: {e}")
            
    async def _initialize_risk_tables(self):
        """Initialize database tables for risk tracking"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                portfolio_value DECIMAL(15,2),
                total_risk DECIMAL(8,4),
                position_count INTEGER,
                daily_pnl DECIMAL(15,2),
                max_drawdown DECIMAL(8,4),
                sharpe_ratio DECIMAL(8,4),
                var_95 DECIMAL(15,2),
                var_99 DECIMAL(15,2),
                expected_shortfall DECIMAL(15,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                alert_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                symbol VARCHAR(20),
                message TEXT NOT NULL,
                risk_value DECIMAL(8,4),
                threshold_value DECIMAL(8,4),
                acknowledged BOOLEAN DEFAULT FALSE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS position_risks (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                position_size DECIMAL(15,2),
                market_value DECIMAL(15,2),
                var_contribution DECIMAL(15,2),
                correlation_risk DECIMAL(8,4),
                concentration_risk DECIMAL(8,4),
                liquidity_risk DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
            """
        ]
        
        for table_sql in tables:
            await self.db_manager.execute_query(table_sql)
            
    async def _load_portfolio_state(self):
        """Load current portfolio state"""
        
        try:
            # Load current positions
            query = """
            SELECT symbol, quantity, avg_price, market_value, unrealized_pnl
            FROM portfolio_positions 
            WHERE status = 'ACTIVE'
            """
            
            positions = await self.db_manager.fetch_all(query)
            
            for position in positions:
                self.current_positions[position['symbol']] = {
                    'quantity': float(position['quantity']),
                    'avg_price': float(position['avg_price']),
                    'market_value': float(position['market_value']),
                    'unrealized_pnl': float(position['unrealized_pnl'])
                }
                
            # Calculate current portfolio risk
            await self._calculate_portfolio_risk()
            
            logger.info(f"Loaded {len(self.current_positions)} active positions")
            
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            
    async def _load_risk_parameters(self):
        """Load risk parameters from configuration"""
        
        # These could be loaded from database or configuration
        # For now, using config values with potential dynamic adjustments
        
        # Adjust risk parameters based on market conditions
        current_volatility = await self._estimate_market_volatility()
        
        if current_volatility > 0.25:  # High volatility
            self.max_portfolio_risk *= 0.8  # Reduce risk tolerance
            self.max_position_size *= 0.8
        elif current_volatility < 0.15:  # Low volatility
            self.max_portfolio_risk *= 1.1  # Slightly increase risk tolerance
            
        logger.info(f"Risk parameters loaded: max_portfolio_risk={self.max_portfolio_risk:.3f}, "
                   f"max_position_size={self.max_position_size:.3f}")
                   
    async def _estimate_market_volatility(self) -> float:
        """Estimate current market volatility"""
        
        try:
            # Get recent market data for volatility estimation
            query = """
            SELECT close, LAG(close) OVER (ORDER BY date) as prev_close
            FROM market_data 
            WHERE symbol = %s AND date >= %s
            ORDER BY date
            """
            
            cutoff_date = datetime.now() - timedelta(days=30)
            data = await self.db_manager.fetch_dataframe(
                query, (self.config.BENCHMARK_SYMBOL, cutoff_date)
            )
            
            if not data.empty:
                data['returns'] = data['close'] / data['prev_close'] - 1
                volatility = data['returns'].std() * np.sqrt(252)  # Annualized
                return volatility
                
        except Exception as e:
            logger.error(f"Error estimating volatility: {e}")
            
        return 0.2  # Default volatility estimate
        
    async def assess_trade_risk(self, signal: Dict, regime_info: Dict = None) -> Dict[str, Any]:
        """Comprehensive trade risk assessment"""
        
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            current_price = signal['current_price']
            
            # Individual risk assessments
            position_risk = await self._assess_position_risk(signal)
            portfolio_risk = await self._assess_portfolio_impact(signal)
            market_risk = await self._assess_market_risk(signal, regime_info)
            liquidity_risk = await self._assess_liquidity_risk(signal)
            concentration_risk = await self._assess_concentration_risk(signal)
            
            # Combine risk scores
            risk_weights = {
                'position_risk': 0.25,
                'portfolio_risk': 0.25,
                'market_risk': 0.20,
                'liquidity_risk': 0.15,
                'concentration_risk': 0.15
            }
            
            composite_risk = (
                position_risk['risk_score'] * risk_weights['position_risk'] +
                portfolio_risk['risk_score'] * risk_weights['portfolio_risk'] +
                market_risk['risk_score'] * risk_weights['market_risk'] +
                liquidity_risk['risk_score'] * risk_weights['liquidity_risk'] +
                concentration_risk['risk_score'] * risk_weights['concentration_risk']
            )
            
            # Risk-adjusted position sizing
            recommended_size = self._calculate_position_size(signal, composite_risk)
            
            # Stop loss recommendation
            recommended_stop_loss = self._calculate_stop_loss(signal, composite_risk)
            
            # Risk limits check
            limit_violations = await self._check_risk_limits(signal, recommended_size)
            
            risk_assessment = {
                'risk_score': composite_risk,
                'position_risk': position_risk,
                'portfolio_risk': portfolio_risk,
                'market_risk': market_risk,
                'liquidity_risk': liquidity_risk,
                'concentration_risk': concentration_risk,
                'recommended_position_size': recommended_size,
                'recommended_stop_loss': recommended_stop_loss,
                'limit_violations': limit_violations,
                'risk_approved': len(limit_violations) == 0 and composite_risk <= 0.7,
                'risk_factors': self._identify_key_risk_factors(
                    position_risk, portfolio_risk, market_risk, 
                    liquidity_risk, concentration_risk
                )
            }
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing trade risk: {e}")
            return {
                'risk_score': 1.0,
                'risk_approved': False,
                'recommended_position_size': 0,
                'recommended_stop_loss': 0.05,
                'limit_violations': ['Risk assessment error'],
                'error': str(e)
            }
            
    async def _assess_position_risk(self, signal: Dict) -> Dict[str, Any]:
        """Assess individual position risk"""
        
        try:
            symbol = signal['symbol']
            current_price = signal['current_price']
            
            # Get historical volatility
            historical_vol = await self._get_historical_volatility(symbol)
            
            # Technical indicators risk
            tech_indicators = signal.get('technical_indicators', {})
            rsi = tech_indicators.get('rsi', 50)
            atr = tech_indicators.get('atr', 0)
            
            # Price risk based on volatility
            if historical_vol > 0.4:  # High volatility
                price_risk = 0.8
            elif historical_vol > 0.25:  # Medium volatility
                price_risk = 0.5
            else:  # Low volatility
                price_risk = 0.2
                
            # Technical risk
            tech_risk = 0.3  # Base technical risk
            if rsi > 80 or rsi < 20:  # Extreme RSI
                tech_risk += 0.3
            if atr / current_price > 0.05:  # High ATR
                tech_risk += 0.2
                
            # Time risk (intraday vs swing)
            time_risk = 0.2  # Base time risk
            
            overall_risk = min((price_risk + tech_risk + time_risk) / 3, 1.0)
            
            return {
                'risk_score': overall_risk,
                'price_risk': price_risk,
                'technical_risk': tech_risk,
                'time_risk': time_risk,
                'historical_volatility': historical_vol,
                'assessment': 'HIGH' if overall_risk > 0.7 else 'MEDIUM' if overall_risk > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error assessing position risk: {e}")
            return {'risk_score': 0.5, 'assessment': 'MEDIUM', 'error': str(e)}
            
    async def _assess_portfolio_impact(self, signal: Dict) -> Dict[str, Any]:
        """Assess impact on portfolio risk"""
        
        try:
            symbol = signal['symbol']
            
            # Current portfolio concentration
            total_portfolio_value = sum(pos['market_value'] for pos in self.current_positions.values())
            
            if total_portfolio_value == 0:
                concentration_risk = 0
            else:
                # Existing concentration in this symbol
                existing_value = self.current_positions.get(symbol, {}).get('market_value', 0)
                existing_concentration = existing_value / total_portfolio_value
                
                # Portfolio heat (number of positions)
                portfolio_heat = len(self.current_positions) / 20  # Normalize to 20 positions
                
                concentration_risk = max(existing_concentration, portfolio_heat)
                
            # Correlation risk (simplified)
            correlation_risk = await self._estimate_correlation_risk(symbol)
            
            # Daily trade limit impact
            trade_limit_risk = self.daily_trade_count / self.max_daily_trades
            
            overall_risk = min((concentration_risk + correlation_risk + trade_limit_risk) / 3, 1.0)
            
            return {
                'risk_score': overall_risk,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'trade_limit_risk': trade_limit_risk,
                'current_positions': len(self.current_positions),
                'daily_trades': self.daily_trade_count,
                'assessment': 'HIGH' if overall_risk > 0.7 else 'MEDIUM' if overall_risk > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio impact: {e}")
            return {'risk_score': 0.5, 'assessment': 'MEDIUM', 'error': str(e)}
            
    async def _assess_market_risk(self, signal: Dict, regime_info: Dict = None) -> Dict[str, Any]:
        """Assess market-wide risk factors"""
        
        try:
            # Market regime risk
            regime_risk = 0.3  # Base regime risk
            
            if regime_info:
                regime_type = regime_info.get('regime_type', 'SIDEWAYS')
                regime_confidence = regime_info.get('confidence', 0.5)
                
                if regime_type == 'BEAR':
                    regime_risk = 0.7
                elif regime_type == 'BULL':
                    regime_risk = 0.2
                    
                # Adjust for confidence
                regime_risk *= (2 - regime_confidence)  # Higher risk if low confidence
                
            # Market volatility risk
            market_vol = await self._estimate_market_volatility()
            vol_risk = min(market_vol / 0.3, 1.0)  # Normalize to 30% volatility
            
            # Time of day risk (simplified)
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 10 or 14 <= current_hour <= 15:  # Opening/closing hours
                time_risk = 0.6
            else:
                time_risk = 0.3
                
            overall_risk = min((regime_risk + vol_risk + time_risk) / 3, 1.0)
            
            return {
                'risk_score': overall_risk,
                'regime_risk': regime_risk,
                'volatility_risk': vol_risk,
                'time_risk': time_risk,
                'market_volatility': market_vol,
                'assessment': 'HIGH' if overall_risk > 0.7 else 'MEDIUM' if overall_risk > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error assessing market risk: {e}")
            return {'risk_score': 0.5, 'assessment': 'MEDIUM', 'error': str(e)}
            
    async def _assess_liquidity_risk(self, signal: Dict) -> Dict[str, Any]:
        """Assess liquidity risk"""
        
        try:
            symbol = signal['symbol']
            volume = signal.get('volume', 0)
            
            # Get average volume
            avg_volume = await self._get_average_volume(symbol)
            
            if avg_volume == 0:
                liquidity_risk = 0.8  # High risk if no volume data
            else:
                volume_ratio = volume / avg_volume
                
                if volume_ratio > 2:  # High volume
                    liquidity_risk = 0.1
                elif volume_ratio > 1:  # Normal volume
                    liquidity_risk = 0.3
                elif volume_ratio > 0.5:  # Low volume
                    liquidity_risk = 0.6
                else:  # Very low volume
                    liquidity_risk = 0.9
                    
            # Market cap consideration (simplified)
            # In practice, would use actual market cap data
            market_cap_risk = 0.3  # Assume medium-cap stocks
            
            overall_risk = min((liquidity_risk + market_cap_risk) / 2, 1.0)
            
            return {
                'risk_score': overall_risk,
                'volume_risk': liquidity_risk,
                'market_cap_risk': market_cap_risk,
                'current_volume': volume,
                'average_volume': avg_volume,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 0,
                'assessment': 'HIGH' if overall_risk > 0.7 else 'MEDIUM' if overall_risk > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error assessing liquidity risk: {e}")
            return {'risk_score': 0.5, 'assessment': 'MEDIUM', 'error': str(e)}
            
    async def _assess_concentration_risk(self, signal: Dict) -> Dict[str, Any]:
        """Assess concentration risk"""
        
        try:
            symbol = signal['symbol']
            
            # Calculate sector concentration (simplified)
            # In practice, would use actual sector classification
            sector_concentration = len(self.current_positions) / 10  # Assume 10 different sectors
            
            # Single stock concentration
            total_value = sum(pos['market_value'] for pos in self.current_positions.values())
            existing_value = self.current_positions.get(symbol, {}).get('market_value', 0)
            
            single_stock_concentration = existing_value / total_value if total_value > 0 else 0
            
            # Geographic concentration (NSE only, so always 1.0)
            geographic_concentration = 1.0
            
            overall_risk = min(
                (sector_concentration + single_stock_concentration + geographic_concentration * 0.2) / 2.2, 
                1.0
            )
            
            return {
                'risk_score': overall_risk,
                'sector_concentration': sector_concentration,
                'single_stock_concentration': single_stock_concentration,
                'geographic_concentration': geographic_concentration,
                'assessment': 'HIGH' if overall_risk > 0.7 else 'MEDIUM' if overall_risk > 0.4 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error assessing concentration risk: {e}")
            return {'risk_score': 0.5, 'assessment': 'MEDIUM', 'error': str(e)}
            
    async def _get_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """Get historical volatility for a symbol"""
        
        try:
            query = """
            SELECT close, LAG(close) OVER (ORDER BY date) as prev_close
            FROM market_data 
            WHERE symbol = %s AND date >= %s
            ORDER BY date
            """
            
            cutoff_date = datetime.now() - timedelta(days=days)
            data = await self.db_manager.fetch_dataframe(query, (symbol, cutoff_date))
            
            if not data.empty and len(data) > 10:
                data['returns'] = data['close'] / data['prev_close'] - 1
                volatility = data['returns'].std() * np.sqrt(252)  # Annualized
                return volatility
                
        except Exception as e:
            logger.error(f"Error getting historical volatility for {symbol}: {e}")
            
        return 0.25  # Default volatility
        
    async def _get_average_volume(self, symbol: str, days: int = 20) -> float:
        """Get average volume for a symbol"""
        
        try:
            query = """
            SELECT AVG(volume) as avg_volume
            FROM market_data 
            WHERE symbol = %s AND date >= %s
            """
            
            cutoff_date = datetime.now() - timedelta(days=days)
            result = await self.db_manager.fetch_one(query, (symbol, cutoff_date))
            
            if result and result['avg_volume']:
                return float(result['avg_volume'])
                
        except Exception as e:
            logger.error(f"Error getting average volume for {symbol}: {e}")
            
        return 100000  # Default average volume
        
    async def _estimate_correlation_risk(self, symbol: str) -> float:
        """Estimate correlation risk with existing positions"""
        
        try:
            if not self.current_positions:
                return 0  # No correlation risk if no positions
                
            # Simplified correlation risk
            # In practice, would calculate actual correlations
            
            # Assume moderate correlation between stocks
            base_correlation = 0.3
            
            # Higher correlation risk with more positions
            position_count_factor = min(len(self.current_positions) / 10, 1.0)
            
            correlation_risk = base_correlation * position_count_factor
            
            return correlation_risk
            
        except Exception as e:
            logger.error(f"Error estimating correlation risk: {e}")
            return 0.3  # Default correlation risk
            
    def _calculate_position_size(self, signal: Dict, risk_score: float) -> float:
        """Calculate risk-adjusted position size"""
        
        try:
            # Base position size
            base_size = self.max_position_size
            
            # Risk adjustment
            risk_multiplier = 1 - min(risk_score, 0.8)  # Reduce size for higher risk
            
            # Signal strength adjustment
            signal_strength = signal.get('signal_strength', 0.5)
            strength_multiplier = 0.5 + signal_strength * 0.5  # 0.5 to 1.0 range
            
            # Portfolio heat adjustment
            portfolio_positions = len(self.current_positions)
            heat_multiplier = max(0.3, 1 - portfolio_positions / 20)  # Reduce size with more positions
            
            adjusted_size = base_size * risk_multiplier * strength_multiplier * heat_multiplier
            
            return max(0.01, min(adjusted_size, self.max_position_size))  # Minimum 1%, max configured
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.02  # Default 2% position size
            
    def _calculate_stop_loss(self, signal: Dict, risk_score: float) -> float:
        """Calculate appropriate stop loss percentage"""
        
        try:
            # Base stop loss
            base_stop_loss = 0.02  # 2%
            
            # Adjust for risk
            risk_adjustment = 1 + risk_score * 0.5  # Increase stop loss for higher risk
            
            # Adjust for volatility
            tech_indicators = signal.get('technical_indicators', {})
            atr = tech_indicators.get('atr', 0)
            current_price = signal['current_price']
            
            if atr > 0 and current_price > 0:
                atr_stop = (atr / current_price) * self.config.ATR_MULTIPLIER
                volatility_stop = max(base_stop_loss, atr_stop)
            else:
                volatility_stop = base_stop_loss
                
            final_stop_loss = volatility_stop * risk_adjustment
            
            return min(final_stop_loss, 0.10)  # Maximum 10% stop loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return 0.03  # Default 3% stop loss
            
    async def _check_risk_limits(self, signal: Dict, position_size: float) -> List[str]:
        """Check for risk limit violations"""
        
        violations = []
        
        try:
            # Daily trade limit
            if self.daily_trade_count >= self.max_daily_trades:
                violations.append(f"Daily trade limit exceeded ({self.daily_trade_count}/{self.max_daily_trades})")
                
            # Portfolio risk limit
            if self.current_portfolio_risk >= self.max_portfolio_risk:
                violations.append(f"Portfolio risk limit exceeded ({self.current_portfolio_risk:.3f}/{self.max_portfolio_risk:.3f})")
                
            # Position size limit
            if position_size > self.max_position_size:
                violations.append(f"Position size limit exceeded ({position_size:.3f}/{self.max_position_size:.3f})")
                
            # Maximum positions limit
            max_positions = 20  # Configuration parameter
            if len(self.current_positions) >= max_positions:
                violations.append(f"Maximum positions limit reached ({len(self.current_positions)}/{max_positions})")
                
            # Risk budget check
            if self.risk_budget_used >= 1.0:
                violations.append(f"Risk budget fully utilized ({self.risk_budget_used:.3f})")
                
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            violations.append(f"Risk limit check error: {str(e)}")
            
        return violations
        
    def _identify_key_risk_factors(self, *risk_assessments) -> List[str]:
        """Identify key risk factors from assessments"""
        
        key_factors = []
        
        try:
            for assessment in risk_assessments:
                if assessment.get('risk_score', 0) > 0.6:
                    factor_type = assessment.get('assessment', 'UNKNOWN')
                    if factor_type == 'HIGH':
                        # Identify specific high-risk factors
                        for key, value in assessment.items():
                            if key.endswith('_risk') and isinstance(value, (int, float)) and value > 0.6:
                                key_factors.append(key.replace('_risk', '').replace('_', ' ').title())
                                
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            
        return list(set(key_factors))  # Remove duplicates
        
    async def _calculate_portfolio_risk(self):
        """Calculate current portfolio risk"""
        
        try:
            if not self.current_positions:
                self.current_portfolio_risk = 0
                return
                
            # Calculate position risks
            total_risk = 0
            
            for symbol, position in self.current_positions.items():
                # Position volatility contribution
                position_vol = await self._get_historical_volatility(symbol)
                position_weight = position['market_value'] / sum(p['market_value'] for p in self.current_positions.values())
                
                position_risk = position_vol * position_weight
                total_risk += position_risk ** 2  # Risk contribution
                
            # Portfolio risk (simplified, without correlations)
            self.current_portfolio_risk = np.sqrt(total_risk)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            self.current_portfolio_risk = 0.5  # Conservative estimate
            
    async def update_position(self, symbol: str, position_data: Dict):
        """Update position data for risk tracking"""
        
        try:
            self.current_positions[symbol] = position_data
            
            # Recalculate portfolio risk
            await self._calculate_portfolio_risk()
            
        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            
    async def close_position(self, symbol: str):
        """Remove position from risk tracking"""
        
        try:
            if symbol in self.current_positions:
                del self.current_positions[symbol]
                
            # Recalculate portfolio risk
            await self._calculate_portfolio_risk()
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            
    async def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        try:
            # Calculate metrics
            total_portfolio_value = sum(pos['market_value'] for pos in self.current_positions.values())
            total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.current_positions.values())
            
            # Position analysis
            largest_position = max(
                self.current_positions.values(), 
                key=lambda x: x['market_value'],
                default={'market_value': 0}
            )['market_value']
            
            concentration = largest_position / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Risk alerts
            active_alerts = len([alert for alert in self.risk_alerts if not alert.get('acknowledged', False)])
            
            report = {
                'timestamp': datetime.now(),
                'portfolio_summary': {
                    'total_value': total_portfolio_value,
                    'unrealized_pnl': total_unrealized_pnl,
                    'position_count': len(self.current_positions),
                    'largest_position_pct': concentration * 100
                },
                'risk_metrics': {
                    'portfolio_risk': self.current_portfolio_risk,
                    'risk_budget_used': self.risk_budget_used,
                    'daily_trades': self.daily_trade_count,
                    'max_daily_trades': self.max_daily_trades
                },
                'risk_limits': {
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_position_size': self.max_position_size,
                    'portfolio_risk_utilization': self.current_portfolio_risk / self.max_portfolio_risk
                },
                'alerts': {
                    'active_alerts': active_alerts,
                    'total_alerts': len(self.risk_alerts)
                },
                'positions': list(self.current_positions.keys())
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
            
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of risk management status"""
        
        return {
            'current_portfolio_risk': self.current_portfolio_risk,
            'max_portfolio_risk': self.max_portfolio_risk,
            'risk_utilization': self.current_portfolio_risk / self.max_portfolio_risk if self.max_portfolio_risk > 0 else 0,
            'position_count': len(self.current_positions),
            'daily_trades': self.daily_trade_count,
            'max_daily_trades': self.max_daily_trades,
            'risk_budget_used': self.risk_budget_used,
            'active_alerts': len([alert for alert in self.risk_alerts if not alert.get('acknowledged', False)])
        }
