"""
Advanced Market Regime Detection using Hidden Markov Models
Identifies bull, bear, and sideways market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as stats

from config import Config
from database.db_manager import DatabaseManager
from utils.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Advanced market regime detection using Hidden Markov Models
    and clustering techniques
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager()
        self.statistical_analysis = StatisticalAnalysis()
        
        # Model components
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.current_regime = None
        self.regime_probabilities = None
        self.regime_history = []
        
        # Regime characteristics
        self.regime_stats = {}
        self.transition_matrix = None
        
        # Feature extractors
        self.volatility_estimator = None
        self.trend_estimator = None
        
    async def initialize(self):
        """Initialize regime detection models"""
        logger.info("Initializing market regime detection...")
        
        # Load historical data for training
        historical_data = await self._load_historical_data()
        
        if len(historical_data) > 0:
            await self._train_regime_models(historical_data)
            logger.info("Regime detection models initialized successfully")
        else:
            logger.warning("Insufficient historical data for regime detection")
            
    async def _load_historical_data(self, days_back: int = 500) -> pd.DataFrame:
        """Load historical market data for regime training"""
        
        query = """
        SELECT date, open, high, low, close, volume
        FROM market_data 
        WHERE symbol = %s 
        AND date >= %s
        ORDER BY date
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        df = await self.db_manager.fetch_dataframe(
            query, (self.config.BENCHMARK_SYMBOL, cutoff_date)
        )
        
        return df
        
    async def _train_regime_models(self, data: pd.DataFrame):
        """Train Hidden Markov Model for regime detection"""
        
        # Extract regime features
        features = self._extract_regime_features(data)
        
        if features.empty:
            logger.error("No features extracted for regime detection")
            return
            
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train HMM model
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.config.REGIME_STATES,
            covariance_type="full",
            random_state=42
        )
        
        try:
            self.hmm_model.fit(features_scaled)
            
            # Predict regimes for historical data
            regime_sequence = self.hmm_model.predict(features_scaled)
            
            # Analyze regime characteristics
            self._analyze_regime_characteristics(data, regime_sequence)
            
            # Get transition matrix
            self.transition_matrix = self.hmm_model.transmat_
            
            logger.info("HMM regime detection model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training HMM model: {e}")
            
    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection"""
        
        if data.empty or len(data) < 30:
            return pd.DataFrame()
            
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Trend features
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend_20_50'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Range features
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['true_range'].rolling(14).mean()
        df['normalized_range'] = df['true_range'] / df['close']
        
        # Skewness and kurtosis of returns
        df['returns_skew'] = df['returns'].rolling(20).apply(lambda x: stats.skew(x.dropna()))
        df['returns_kurtosis'] = df['returns'].rolling(20).apply(lambda x: stats.kurtosis(x.dropna()))
        
        # VIX-like fear index (using rolling volatility)
        df['fear_index'] = df['volatility_20'] * np.sqrt(252) * 100
        
        # Market microstructure
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Select features for regime detection
        regime_features = [
            'returns', 'volatility_20', 'trend_20_50', 'momentum_20',
            'volume_ratio', 'normalized_range', 'returns_skew',
            'returns_kurtosis', 'fear_index', 'close_position'
        ]
        
        feature_df = df[regime_features].dropna()
        
        return feature_df
        
    def _analyze_regime_characteristics(self, data: pd.DataFrame, regime_sequence: np.ndarray):
        """Analyze characteristics of each detected regime"""
        
        df = data.copy()
        df['regime'] = regime_sequence
        df['returns'] = df['close'].pct_change()
        
        self.regime_stats = {}
        
        for regime in range(self.config.REGIME_STATES):
            regime_data = df[df['regime'] == regime]
            
            if len(regime_data) > 10:
                stats_dict = {
                    'mean_return': regime_data['returns'].mean(),
                    'volatility': regime_data['returns'].std(),
                    'sharpe_ratio': regime_data['returns'].mean() / regime_data['returns'].std() if regime_data['returns'].std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(regime_data['close']),
                    'avg_duration': len(regime_data),
                    'win_rate': (regime_data['returns'] > 0).mean(),
                    'skewness': stats.skew(regime_data['returns'].dropna()),
                    'kurtosis': stats.kurtosis(regime_data['returns'].dropna())
                }
                
                # Classify regime type
                if stats_dict['mean_return'] > 0.001 and stats_dict['volatility'] < 0.02:
                    regime_type = 'BULL'
                elif stats_dict['mean_return'] < -0.001 and stats_dict['volatility'] > 0.025:
                    regime_type = 'BEAR'
                else:
                    regime_type = 'SIDEWAYS'
                    
                stats_dict['type'] = regime_type
                self.regime_stats[regime] = stats_dict
                
                logger.info(f"Regime {regime} ({regime_type}): Return={stats_dict['mean_return']:.4f}, Vol={stats_dict['volatility']:.4f}")
                
    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown for a price series"""
        cumulative = (1 + price_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
        
    async def detect_current_regime(self, recent_data: pd.DataFrame = None) -> Dict:
        """Detect current market regime"""
        
        if self.hmm_model is None:
            logger.warning("HMM model not trained")
            return {'regime': None, 'confidence': 0, 'probabilities': []}
            
        if recent_data is None:
            # Get recent market data
            recent_data = await self._load_historical_data(days_back=60)
            
        if recent_data.empty:
            logger.error("No recent data available for regime detection")
            return {'regime': None, 'confidence': 0, 'probabilities': []}
            
        # Extract features
        features = self._extract_regime_features(recent_data)
        
        if features.empty:
            return {'regime': None, 'confidence': 0, 'probabilities': []}
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get most recent regime probabilities
        log_prob, regime_sequence = self.hmm_model.decode(features_scaled)
        regime_probs = self.hmm_model.predict_proba(features_scaled)
        
        # Current regime is the last prediction
        current_regime = regime_sequence[-1]
        current_probs = regime_probs[-1]
        
        # Calculate confidence as max probability
        confidence = np.max(current_probs)
        
        # Update internal state
        self.current_regime = current_regime
        self.regime_probabilities = current_probs
        
        # Add to history
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': current_regime,
            'probabilities': current_probs.tolist(),
            'confidence': confidence
        })
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
            
        regime_info = {
            'regime': current_regime,
            'regime_type': self.regime_stats.get(current_regime, {}).get('type', 'UNKNOWN'),
            'confidence': confidence,
            'probabilities': current_probs.tolist(),
            'regime_stats': self.regime_stats.get(current_regime, {}),
            'transition_probabilities': self.transition_matrix[current_regime].tolist() if self.transition_matrix is not None else []
        }
        
        logger.info(f"Current regime: {current_regime} ({regime_info['regime_type']}) with confidence {confidence:.3f}")
        
        return regime_info
        
    async def get_regime_adjusted_parameters(self) -> Dict:
        """Get trading parameters adjusted for current market regime"""
        
        if self.current_regime is None:
            return self._get_default_parameters()
            
        regime_stats = self.regime_stats.get(self.current_regime, {})
        regime_type = regime_stats.get('type', 'SIDEWAYS')
        
        # Adjust parameters based on regime
        if regime_type == 'BULL':
            parameters = {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.3,
                'confidence_threshold': 0.6,
                'max_trades_per_day': 4,
                'trend_following_weight': 1.2,
                'mean_reversion_weight': 0.8
            }
        elif regime_type == 'BEAR':
            parameters = {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 0.8,
                'confidence_threshold': 0.8,
                'max_trades_per_day': 2,
                'trend_following_weight': 0.8,
                'mean_reversion_weight': 1.2
            }
        else:  # SIDEWAYS
            parameters = {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'confidence_threshold': 0.7,
                'max_trades_per_day': 3,
                'trend_following_weight': 0.9,
                'mean_reversion_weight': 1.1
            }
            
        # Adjust for regime confidence
        confidence_factor = self.regime_probabilities[self.current_regime] if self.regime_probabilities is not None else 0.5
        
        parameters['position_size_multiplier'] *= (0.5 + 0.5 * confidence_factor)
        parameters['confidence_threshold'] += (1 - confidence_factor) * 0.1
        
        return parameters
        
    def _get_default_parameters(self) -> Dict:
        """Get default trading parameters when regime is unknown"""
        return {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'confidence_threshold': 0.75,
            'max_trades_per_day': 2,
            'trend_following_weight': 1.0,
            'mean_reversion_weight': 1.0
        }
        
    async def get_regime_forecast(self, steps_ahead: int = 5) -> Dict:
        """Forecast regime transitions for next few periods"""
        
        if self.hmm_model is None or self.current_regime is None:
            return {'forecast': [], 'confidence': 0}
            
        # Current state probabilities
        current_state_prob = np.zeros(self.config.REGIME_STATES)
        current_state_prob[self.current_regime] = 1.0
        
        forecasts = []
        state_prob = current_state_prob.copy()
        
        for step in range(1, steps_ahead + 1):
            # Transition to next period
            state_prob = state_prob @ self.transition_matrix
            
            # Most likely regime
            most_likely_regime = np.argmax(state_prob)
            confidence = np.max(state_prob)
            
            forecasts.append({
                'step': step,
                'regime': most_likely_regime,
                'regime_type': self.regime_stats.get(most_likely_regime, {}).get('type', 'UNKNOWN'),
                'confidence': confidence,
                'probabilities': state_prob.tolist()
            })
            
        avg_confidence = np.mean([f['confidence'] for f in forecasts])
        
        return {
            'forecast': forecasts,
            'avg_confidence': avg_confidence,
            'current_regime': self.current_regime,
            'transition_matrix': self.transition_matrix.tolist()
        }
        
    def get_regime_summary(self) -> Dict:
        """Get comprehensive regime analysis summary"""
        
        return {
            'current_regime': self.current_regime,
            'current_regime_type': self.regime_stats.get(self.current_regime, {}).get('type', 'UNKNOWN') if self.current_regime is not None else None,
            'regime_confidence': np.max(self.regime_probabilities) if self.regime_probabilities is not None else 0,
            'regime_probabilities': self.regime_probabilities.tolist() if self.regime_probabilities is not None else [],
            'regime_statistics': self.regime_stats,
            'transition_matrix': self.transition_matrix.tolist() if self.transition_matrix is not None else [],
            'regime_history': self.regime_history[-10:],  # Last 10 regime detections
            'model_trained': self.hmm_model is not None
        }
