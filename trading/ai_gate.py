"""
AI Gate - Central Decision Engine for Trade Approval
Uses ensemble ML models and reinforcement learning for trade decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio

from config import Config
from database.db_manager import DatabaseManager
from models.ml_models import EnsembleMLModel
from models.regime_detection import MarketRegimeDetector
from models.reinforcement_learning import RLTradingAgent
from data.feature_engineering import FeatureEngineer
from trading.risk_manager import RiskManager
from utils.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class AIGate:
    """
    AI-powered trade decision gate
    Central intelligence for approving/rejecting trades
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        
        # AI/ML Components
        self.ensemble_model = EnsembleMLModel(self.config)
        self.regime_detector = MarketRegimeDetector(self.config)
        self.rl_agent = RLTradingAgent(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.statistical_analysis = StatisticalAnalysis()
        
        # Decision tracking
        self.decision_history = []
        self.model_performance = {}
        self.confidence_threshold = self.config.MIN_CONFIDENCE_THRESHOLD
        
        # Feature importance tracking
        self.feature_weights = {}
        self.decision_factors = {}
        
    async def initialize(self):
        """Initialize AI Gate components"""
        logger.info("Initializing AI Gate...")
        
        try:
            # Initialize all AI components
            await self.regime_detector.initialize()
            await self.rl_agent.initialize()
            
            # Try to load existing models
            models_loaded = await self.ensemble_model.load_models()
            
            if not models_loaded:
                logger.info("No existing models found, will train on first run")
                
            # Initialize decision tracking tables
            await self._initialize_decision_tables()
            
            # Load recent decision history
            await self._load_decision_history()
            
            logger.info("AI Gate initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing AI Gate: {e}")
            
    async def _initialize_decision_tables(self):
        """Initialize database tables for decision tracking"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(20) NOT NULL,
                signal_type VARCHAR(10) NOT NULL,
                signal_strength DECIMAL(8,4),
                ai_decision BOOLEAN NOT NULL,
                confidence DECIMAL(8,4),
                ensemble_probability DECIMAL(8,4),
                rl_recommendation VARCHAR(10),
                rl_confidence DECIMAL(8,4),
                regime_type VARCHAR(20),
                regime_confidence DECIMAL(8,4),
                risk_score DECIMAL(8,4),
                features JSON,
                rationale TEXT,
                actual_outcome DECIMAL(8,4),
                outcome_updated_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                model_name VARCHAR(50) NOT NULL,
                accuracy DECIMAL(8,4),
                precision_score DECIMAL(8,4),
                recall DECIMAL(8,4),
                f1_score DECIMAL(8,4),
                auc_score DECIMAL(8,4),
                total_predictions INTEGER,
                correct_predictions INTEGER,
                UNIQUE(date, model_name)
            )
            """
        ]
        
        for table_sql in tables:
            await self.db_manager.execute_query(table_sql)
            
    async def _load_decision_history(self):
        """Load recent decision history"""
        
        try:
            query = """
            SELECT * FROM ai_decisions 
            WHERE timestamp >= %s 
            ORDER BY timestamp DESC 
            LIMIT 1000
            """
            
            cutoff_date = datetime.now() - timedelta(days=30)
            decisions = await self.db_manager.fetch_all(query, (cutoff_date,))
            
            self.decision_history = [dict(decision) for decision in decisions]
            
            logger.info(f"Loaded {len(self.decision_history)} recent decisions")
            
        except Exception as e:
            logger.error(f"Error loading decision history: {e}")
            
    async def update_market_regime(self):
        """Update market regime detection"""
        
        try:
            regime_info = await self.regime_detector.detect_current_regime()
            
            # Get regime-adjusted parameters
            regime_params = await self.regime_detector.get_regime_adjusted_parameters()
            
            # Update confidence threshold based on regime
            self.confidence_threshold = (
                self.config.MIN_CONFIDENCE_THRESHOLD * 
                regime_params.get('confidence_threshold', 1.0)
            )
            
            logger.info(f"Market regime updated: {regime_info.get('regime_type', 'UNKNOWN')} "
                       f"(confidence: {regime_info.get('confidence', 0):.3f})")
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error updating market regime: {e}")
            return {}
            
    async def generate_watchlist(self, candidate_stocks: List[Dict]) -> List[Dict]:
        """Generate AI-filtered watchlist from candidate stocks"""
        
        logger.info(f"Generating watchlist from {len(candidate_stocks)} candidates...")
        
        watchlist = []
        
        try:
            for stock in candidate_stocks:
                # Extract features for AI evaluation
                features = await self._extract_features_for_evaluation(stock)
                
                if not features:
                    continue
                    
                # Get AI recommendation
                ai_decision = await self._evaluate_for_watchlist(stock, features)
                
                if ai_decision['add_to_watchlist']:
                    stock_info = {
                        'symbol': stock['symbol'],
                        'current_price': stock['current_price'],
                        'ai_score': ai_decision['score'],
                        'confidence': ai_decision['confidence'],
                        'rationale': ai_decision['rationale'],
                        'risk_level': ai_decision['risk_level'],
                        'expected_return': ai_decision['expected_return'],
                        'time_horizon': ai_decision['time_horizon']
                    }
                    watchlist.append(stock_info)
                    
            # Sort by AI score
            watchlist.sort(key=lambda x: x['ai_score'], reverse=True)
            
            logger.info(f"Generated watchlist with {len(watchlist)} stocks")
            
        except Exception as e:
            logger.error(f"Error generating watchlist: {e}")
            
        return watchlist
        
    async def _extract_features_for_evaluation(self, stock: Dict) -> Dict[str, float]:
        """Extract features for AI evaluation"""
        
        try:
            symbol = stock['symbol']
            
            # Get recent market data for feature extraction
            from data.market_data import MarketDataManager
            market_data_manager = MarketDataManager(self.config)
            
            symbol_data = await market_data_manager.get_symbol_data(symbol)
            
            if symbol_data.empty:
                return {}
                
            # Engineer features
            featured_data = await self.feature_engineer.engineer_features(symbol_data)
            
            if featured_data.empty:
                return {}
                
            # Get latest features
            latest_features = featured_data.iloc[-1]
            
            # Extract feature dictionary
            features = {}
            for col in featured_data.columns:
                if col.startswith('feature_'):
                    value = latest_features[col]
                    features[col] = float(value) if pd.notna(value) else 0.0
                    
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {stock.get('symbol', 'unknown')}: {e}")
            return {}
            
    async def _evaluate_for_watchlist(self, stock: Dict, features: Dict) -> Dict:
        """Evaluate stock for watchlist inclusion"""
        
        try:
            # Get ensemble model prediction
            ensemble_prediction = await self.ensemble_model.predict_trade_outcome(features)
            
            # Get RL agent recommendation
            rl_recommendation = await self.rl_agent.get_action_recommendation(features)
            
            # Get current market regime
            regime_info = await self.regime_detector.detect_current_regime()
            
            # Calculate composite score
            ensemble_weight = 0.4
            rl_weight = 0.3
            regime_weight = 0.2
            technical_weight = 0.1
            
            # Technical score from screening
            technical_score = stock.get('technical_score', 50) / 100
            
            # Regime adjustment
            regime_multiplier = 1.0
            if regime_info.get('regime_type') == 'BULL':
                regime_multiplier = 1.2
            elif regime_info.get('regime_type') == 'BEAR':
                regime_multiplier = 0.8
                
            composite_score = (
                ensemble_prediction['probability'] * ensemble_weight +
                rl_recommendation['confidence'] * rl_weight +
                regime_info.get('confidence', 0.5) * regime_weight +
                technical_score * technical_weight
            ) * regime_multiplier
            
            # Calculate confidence
            individual_confidences = [
                ensemble_prediction['confidence'],
                rl_recommendation['confidence'],
                regime_info.get('confidence', 0.5)
            ]
            
            confidence = np.mean(individual_confidences)
            
            # Decision logic
            add_to_watchlist = (
                composite_score > 0.6 and
                confidence > 0.5 and
                ensemble_prediction['prediction'] == 1
            )
            
            # Generate rationale
            rationale = self._generate_watchlist_rationale(
                ensemble_prediction, rl_recommendation, regime_info, stock
            )
            
            return {
                'add_to_watchlist': add_to_watchlist,
                'score': composite_score,
                'confidence': confidence,
                'rationale': rationale,
                'risk_level': self._assess_risk_level(features, regime_info),
                'expected_return': self._estimate_expected_return(ensemble_prediction, regime_info),
                'time_horizon': self._suggest_time_horizon(rl_recommendation, regime_info)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating stock for watchlist: {e}")
            return {
                'add_to_watchlist': False,
                'score': 0,
                'confidence': 0,
                'rationale': f"Evaluation error: {str(e)}",
                'risk_level': 'HIGH',
                'expected_return': 0,
                'time_horizon': 'UNKNOWN'
            }
            
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate trading signals from market data"""
        
        logger.info(f"Generating signals for {len(market_data)} symbols...")
        
        signals = []
        
        try:
            for symbol, data in market_data.items():
                if data.empty:
                    continue
                    
                # Generate signal for this symbol
                signal = await self._generate_symbol_signal(symbol, data)
                
                if signal:
                    signals.append(signal)
                    
            logger.info(f"Generated {len(signals)} trading signals")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            
        return signals
        
    async def _generate_symbol_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal for a specific symbol"""
        
        try:
            # Engineer features
            featured_data = await self.feature_engineer.engineer_features(data)
            
            if featured_data.empty:
                return None
                
            latest_data = featured_data.iloc[-1]
            
            # Extract features
            features = {}
            for col in featured_data.columns:
                if col.startswith('feature_'):
                    value = latest_data[col]
                    features[col] = float(value) if pd.notna(value) else 0.0
                    
            # Calculate signal strength
            signal_strength = await self._calculate_signal_strength(features, latest_data)
            
            if signal_strength < 0.3:  # Minimum signal strength
                return None
                
            # Determine signal type
            signal_type = await self._determine_signal_type(features, latest_data)
            
            if signal_type == 'HOLD':
                return None
                
            # Create signal
            signal = {
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'current_price': latest_data['close'],
                'volume': latest_data['volume'],
                'features': features,
                'timestamp': datetime.now(),
                'technical_indicators': {
                    'rsi': latest_data.get('rsi', 50),
                    'macd': latest_data.get('macd', 0),
                    'bb_position': latest_data.get('bb_position', 0.5),
                    'atr': latest_data.get('atr', 0)
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
            
    async def _calculate_signal_strength(self, features: Dict, latest_data: pd.Series) -> float:
        """Calculate signal strength based on multiple factors"""
        
        try:
            # Technical momentum
            momentum_5d = features.get('feature_momentum_5d', 0)
            momentum_20d = features.get('feature_momentum_20d', 0)
            
            # Volatility
            volatility = features.get('feature_volatility_20d', 0)
            
            # Volume
            volume_ratio = features.get('feature_volume_ratio', 1)
            
            # RSI divergence from extremes
            rsi = latest_data.get('rsi', 50)
            rsi_strength = 1 - abs(rsi - 50) / 50  # Stronger when not at extremes
            
            # Combine factors
            signal_strength = (
                abs(momentum_5d) * 0.3 +
                abs(momentum_20d) * 0.2 +
                min(volatility * 10, 1) * 0.2 +  # Cap volatility contribution
                min(volume_ratio / 2, 1) * 0.2 +  # Cap volume contribution
                rsi_strength * 0.1
            )
            
            return min(signal_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0
            
    async def _determine_signal_type(self, features: Dict, latest_data: pd.Series) -> str:
        """Determine signal type (BUY, SELL, HOLD)"""
        
        try:
            # Get multiple indicators
            momentum_5d = features.get('feature_momentum_5d', 0)
            rsi = latest_data.get('rsi', 50)
            macd = latest_data.get('macd', 0)
            macd_signal = latest_data.get('macd_signal', 0)
            price_vs_sma20 = features.get('feature_price_vs_sma20', 0)
            
            # Buy signals
            buy_signals = 0
            if momentum_5d > 1:  # Positive momentum
                buy_signals += 1
            if rsi < 70 and rsi > 30:  # RSI in tradeable range
                buy_signals += 1
            if macd > macd_signal:  # MACD bullish
                buy_signals += 1
            if price_vs_sma20 > 0:  # Price above moving average
                buy_signals += 1
                
            # Sell signals (opposite conditions)
            sell_signals = 0
            if momentum_5d < -1:  # Negative momentum
                sell_signals += 1
            if rsi > 70:  # Overbought
                sell_signals += 1
            if macd < macd_signal:  # MACD bearish
                sell_signals += 1
            if price_vs_sma20 < -0.02:  # Price significantly below MA
                sell_signals += 1
                
            # Decision logic
            if buy_signals >= 3:
                return 'BUY'
            elif sell_signals >= 3:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error determining signal type: {e}")
            return 'HOLD'
            
    async def evaluate_trade(self, signal: Dict) -> Dict[str, Any]:
        """Main AI Gate evaluation for trade approval"""
        
        logger.info(f"Evaluating trade signal for {signal['symbol']}")
        
        try:
            # Extract information
            symbol = signal['symbol']
            features = signal['features']
            signal_type = signal['signal_type']
            signal_strength = signal['signal_strength']
            
            # Get AI model predictions
            ensemble_prediction = await self.ensemble_model.predict_trade_outcome(features)
            rl_recommendation = await self.rl_agent.get_action_recommendation(features)
            regime_info = await self.regime_detector.detect_current_regime()
            
            # Risk assessment
            risk_assessment = await self.risk_manager.assess_trade_risk(signal, regime_info)
            
            # Make final decision
            decision = await self._make_final_decision(
                signal, ensemble_prediction, rl_recommendation, 
                regime_info, risk_assessment
            )
            
            # Store decision
            await self._store_decision(
                signal, decision, ensemble_prediction, 
                rl_recommendation, regime_info, risk_assessment
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating trade: {e}")
            return {
                'approved': False,
                'confidence': 0,
                'rationale': f"Evaluation error: {str(e)}",
                'risk_score': 1.0,
                'position_size': 0,
                'stop_loss': 0,
                'take_profit': 0
            }
            
    async def _make_final_decision(
        self, signal: Dict, ensemble_pred: Dict, rl_rec: Dict, 
        regime_info: Dict, risk_assessment: Dict
    ) -> Dict[str, Any]:
        """Make final trade approval decision"""
        
        try:
            # Decision factors
            factors = {
                'ensemble_confidence': ensemble_pred['confidence'],
                'ensemble_probability': ensemble_pred['probability'],
                'rl_confidence': rl_rec['confidence'],
                'regime_confidence': regime_info.get('confidence', 0.5),
                'risk_score': risk_assessment['risk_score'],
                'signal_strength': signal['signal_strength']
            }
            
            # Calculate composite confidence
            confidence_weights = {
                'ensemble_confidence': 0.35,
                'rl_confidence': 0.25,
                'regime_confidence': 0.20,
                'signal_strength': 0.20
            }
            
            composite_confidence = sum(
                factors[factor] * weight 
                for factor, weight in confidence_weights.items()
            )
            
            # Apply risk penalty
            risk_penalty = min(risk_assessment['risk_score'], 0.5)
            adjusted_confidence = composite_confidence * (1 - risk_penalty)
            
            # Decision logic
            approved = (
                adjusted_confidence >= self.confidence_threshold and
                ensemble_pred['prediction'] == 1 and
                risk_assessment['risk_score'] <= 0.7 and
                signal['signal_strength'] >= 0.3
            )
            
            # Position sizing
            position_size = 0
            if approved:
                base_size = self.config.MAX_POSITION_SIZE
                confidence_multiplier = min(adjusted_confidence / self.confidence_threshold, 1.5)
                risk_multiplier = 1 - risk_assessment['risk_score'] * 0.5
                
                position_size = base_size * confidence_multiplier * risk_multiplier
                
            # Stop loss and take profit
            stop_loss_pct = risk_assessment.get('recommended_stop_loss', 0.02)
            take_profit_pct = stop_loss_pct * self.config.TAKE_PROFIT_RATIO
            
            current_price = signal['current_price']
            
            if signal['signal_type'] == 'BUY':
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            else:  # SELL
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
                
            # Generate rationale
            rationale = self._generate_decision_rationale(
                approved, factors, ensemble_pred, rl_rec, regime_info, risk_assessment
            )
            
            decision = {
                'approved': approved,
                'confidence': adjusted_confidence,
                'rationale': rationale,
                'risk_score': risk_assessment['risk_score'],
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'decision_factors': factors,
                'ensemble_prediction': ensemble_pred,
                'rl_recommendation': rl_rec,
                'regime_info': regime_info,
                'risk_assessment': risk_assessment
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making final decision: {e}")
            return {
                'approved': False,
                'confidence': 0,
                'rationale': f"Decision error: {str(e)}",
                'risk_score': 1.0,
                'position_size': 0,
                'stop_loss': 0,
                'take_profit': 0
            }
            
    def _generate_decision_rationale(
        self, approved: bool, factors: Dict, ensemble_pred: Dict, 
        rl_rec: Dict, regime_info: Dict, risk_assessment: Dict
    ) -> str:
        """Generate human-readable rationale for the decision"""
        
        try:
            rationale_parts = []
            
            # Decision outcome
            if approved:
                rationale_parts.append("✅ TRADE APPROVED")
            else:
                rationale_parts.append("❌ TRADE REJECTED")
                
            # Key factors
            rationale_parts.append(f"Ensemble Model: {ensemble_pred['probability']:.3f} probability, {ensemble_pred['confidence']:.3f} confidence")
            rationale_parts.append(f"RL Agent: {rl_rec['action_name']} with {rl_rec['confidence']:.3f} confidence")
            rationale_parts.append(f"Market Regime: {regime_info.get('regime_type', 'UNKNOWN')} ({regime_info.get('confidence', 0):.3f})")
            rationale_parts.append(f"Risk Score: {risk_assessment['risk_score']:.3f}")
            
            # Specific reasons
            if not approved:
                reasons = []
                if factors['ensemble_confidence'] < 0.6:
                    reasons.append("low ensemble confidence")
                if risk_assessment['risk_score'] > 0.7:
                    reasons.append("high risk score")
                if factors['signal_strength'] < 0.3:
                    reasons.append("weak signal strength")
                    
                if reasons:
                    rationale_parts.append(f"Rejection reasons: {', '.join(reasons)}")
                    
            return " | ".join(rationale_parts)
            
        except Exception as e:
            return f"Rationale generation error: {str(e)}"
            
    def _generate_watchlist_rationale(
        self, ensemble_pred: Dict, rl_rec: Dict, regime_info: Dict, stock: Dict
    ) -> str:
        """Generate rationale for watchlist decisions"""
        
        try:
            parts = []
            parts.append(f"AI Score: {ensemble_pred['probability']:.3f}")
            parts.append(f"RL: {rl_rec['action_name']}")
            parts.append(f"Regime: {regime_info.get('regime_type', 'UNKNOWN')}")
            parts.append(f"Technical: {stock.get('technical_score', 0):.1f}")
            
            return " | ".join(parts)
            
        except Exception as e:
            return f"Rationale error: {str(e)}"
            
    def _assess_risk_level(self, features: Dict, regime_info: Dict) -> str:
        """Assess risk level for watchlist stocks"""
        
        try:
            volatility = features.get('feature_volatility_20d', 0)
            atr = features.get('feature_atr_normalized', 0)
            
            # Base risk from volatility
            if volatility > 0.04 or atr > 0.03:
                risk_level = 'HIGH'
            elif volatility > 0.02 or atr > 0.015:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
                
            # Adjust for regime
            if regime_info.get('regime_type') == 'BEAR':
                if risk_level == 'LOW':
                    risk_level = 'MEDIUM'
                elif risk_level == 'MEDIUM':
                    risk_level = 'HIGH'
                    
            return risk_level
            
        except Exception:
            return 'MEDIUM'
            
    def _estimate_expected_return(self, ensemble_pred: Dict, regime_info: Dict) -> float:
        """Estimate expected return"""
        
        try:
            base_return = ensemble_pred['probability'] * 0.05  # 5% max expected
            
            # Adjust for regime
            if regime_info.get('regime_type') == 'BULL':
                base_return *= 1.2
            elif regime_info.get('regime_type') == 'BEAR':
                base_return *= 0.8
                
            return round(base_return * 100, 2)  # Return as percentage
            
        except Exception:
            return 0
            
    def _suggest_time_horizon(self, rl_rec: Dict, regime_info: Dict) -> str:
        """Suggest appropriate time horizon"""
        
        try:
            # Base on RL recommendation strength
            confidence = rl_rec.get('confidence', 0.5)
            
            if confidence > 0.8:
                horizon = 'SHORT'  # 1-5 days
            elif confidence > 0.6:
                horizon = 'MEDIUM'  # 1-2 weeks
            else:
                horizon = 'LONG'  # 2-4 weeks
                
            return horizon
            
        except Exception:
            return 'MEDIUM'
            
    async def _store_decision(
        self, signal: Dict, decision: Dict, ensemble_pred: Dict, 
        rl_rec: Dict, regime_info: Dict, risk_assessment: Dict
    ):
        """Store AI decision in database"""
        
        try:
            insert_query = """
            INSERT INTO ai_decisions 
            (symbol, signal_type, signal_strength, ai_decision, confidence,
             ensemble_probability, rl_recommendation, rl_confidence,
             regime_type, regime_confidence, risk_score, features, rationale)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                signal['symbol'],
                signal['signal_type'],
                signal['signal_strength'],
                decision['approved'],
                decision['confidence'],
                ensemble_pred['probability'],
                rl_rec['action_name'],
                rl_rec['confidence'],
                regime_info.get('regime_type'),
                regime_info.get('confidence'),
                risk_assessment['risk_score'],
                str(signal['features']),  # JSON string
                decision['rationale']
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing decision: {e}")
            
    async def log_decision(self, signal: Dict, decision: Dict):
        """Log decision for learning"""
        
        # Add to decision history for analysis
        self.decision_history.append({
            'timestamp': datetime.now(),
            'symbol': signal['symbol'],
            'signal_type': signal['signal_type'],
            'approved': decision['approved'],
            'confidence': decision['confidence'],
            'signal_strength': signal['signal_strength']
        })
        
        # Keep only recent history
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
            
    async def retrain_models(self):
        """Retrain AI models with new data"""
        
        logger.info("Starting AI model retraining...")
        
        try:
            # Retrain ensemble model
            X, y = await self.ensemble_model.prepare_training_data()
            
            if len(X) > 100:  # Minimum data requirement
                ensemble_results = await self.ensemble_model.train_models(X, y)
                logger.info(f"Ensemble model retrained: {ensemble_results}")
                
            # Retrain RL agent
            rl_results = await self.rl_agent.train_agent(total_timesteps=50000)
            logger.info(f"RL agent retrained: {rl_results}")
            
            # Update market regime
            await self.update_market_regime()
            
            logger.info("AI model retraining completed")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            
    async def save_models(self):
        """Save all AI models"""
        
        try:
            await self.ensemble_model.save_models()
            await self.rl_agent.save_model()
            logger.info("AI models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            
    def get_ai_summary(self) -> Dict[str, Any]:
        """Get comprehensive AI Gate summary"""
        
        return {
            'confidence_threshold': self.confidence_threshold,
            'decision_history_size': len(self.decision_history),
            'recent_approval_rate': self._calculate_recent_approval_rate(),
            'model_performance': self.model_performance,
            'ensemble_model_summary': self.ensemble_model.get_feature_importance_summary(),
            'rl_agent_summary': self.rl_agent.get_model_summary(),
            'regime_summary': self.regime_detector.get_regime_summary(),
            'last_retrain': getattr(self.ensemble_model, 'last_training_date', None)
        }
        
    def _calculate_recent_approval_rate(self) -> float:
        """Calculate recent approval rate"""
        
        if not self.decision_history:
            return 0
            
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        approvals = sum(1 for d in recent_decisions if d.get('approved', False))
        
        return approvals / len(recent_decisions) if recent_decisions else 0
