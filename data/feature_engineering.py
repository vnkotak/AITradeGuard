"""
Advanced Feature Engineering for Trading Signals
Creates sophisticated features from market data and microstructure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import talib as ta
from scipy import stats
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import Config, FEATURE_CONFIG
from utils.technical_indicators import TechnicalIndicators
from utils.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for trading signals
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.technical_indicators = TechnicalIndicators(config)
        self.statistical_analysis = StatisticalAnalysis()
        
        # Feature storage
        self.feature_cache = {}
        self.feature_importance = {}
        
        # PCA for dimensionality reduction
        self.pca_model = None
        self.scaler = StandardScaler()
        
    async def engineer_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        """
        logger.info(f"Engineering features for {len(price_data)} data points")
        
        try:
            df = price_data.copy()
            
            # Basic price features
            df = await self._add_price_features(df)
            
            # Technical indicator features
            df = await self._add_technical_features(df)
            
            # Market microstructure features
            df = await self._add_microstructure_features(df, volume_data)
            
            # Statistical features
            df = await self._add_statistical_features(df)
            
            # Time-based features
            df = await self._add_temporal_features(df)
            
            # Relative strength features
            df = await self._add_relative_strength_features(df)
            
            # Volatility regime features
            df = await self._add_volatility_features(df)
            
            # Pattern recognition features
            df = await self._add_pattern_features(df)
            
            # Cross-asset features
            df = await self._add_cross_asset_features(df)
            
            # Interaction features
            df = await self._add_interaction_features(df)
            
            # Clean and validate features
            df = await self._clean_features(df)
            
            logger.info(f"Feature engineering completed: {len([c for c in df.columns if c.startswith('feature_')])} features created")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return price_data
            
    async def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        
        # Returns
        df['feature_return_1d'] = df['close'].pct_change()
        df['feature_return_3d'] = df['close'].pct_change(3)
        df['feature_return_5d'] = df['close'].pct_change(5)
        df['feature_return_10d'] = df['close'].pct_change(10)
        
        # Log returns
        df['feature_log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['feature_high_low_ratio'] = df['high'] / df['low']
        df['feature_close_open_ratio'] = df['close'] / df['open']
        df['feature_close_high_ratio'] = df['close'] / df['high']
        df['feature_close_low_ratio'] = df['close'] / df['low']
        
        # Price position within range
        df['feature_close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap analysis
        df['feature_gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['feature_gap_magnitude'] = abs(df['feature_gap_up'])
        
        # Price acceleration
        df['feature_price_acceleration'] = df['feature_return_1d'] - df['feature_return_1d'].shift(1)
        
        # Cumulative returns
        df['feature_cumret_5d'] = (1 + df['feature_return_1d']).rolling(5).apply(lambda x: x.prod() - 1)
        df['feature_cumret_20d'] = (1 + df['feature_return_1d']).rolling(20).apply(lambda x: x.prod() - 1)
        
        return df
        
    async def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        
        # RSI
        df['feature_rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['feature_rsi_ma'] = df['feature_rsi'].rolling(5).mean()
        df['feature_rsi_std'] = df['feature_rsi'].rolling(10).std()
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(df['close'])
        df['feature_macd'] = macd
        df['feature_macd_signal'] = macd_signal
        df['feature_macd_histogram'] = macd_hist
        df['feature_macd_cross'] = np.where(macd > macd_signal, 1, 0)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'])
        df['feature_bb_upper'] = bb_upper
        df['feature_bb_middle'] = bb_middle
        df['feature_bb_lower'] = bb_lower
        df['feature_bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['feature_bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Moving Averages
        df['feature_sma_5'] = ta.SMA(df['close'], timeperiod=5)
        df['feature_sma_20'] = ta.SMA(df['close'], timeperiod=20)
        df['feature_sma_50'] = ta.SMA(df['close'], timeperiod=50)
        df['feature_ema_12'] = ta.EMA(df['close'], timeperiod=12)
        df['feature_ema_26'] = ta.EMA(df['close'], timeperiod=26)
        
        # MA relationships
        df['feature_price_vs_sma20'] = df['close'] / df['feature_sma_20'] - 1
        df['feature_sma5_vs_sma20'] = df['feature_sma_5'] / df['feature_sma_20'] - 1
        df['feature_sma20_vs_sma50'] = df['feature_sma_20'] / df['feature_sma_50'] - 1
        
        # ATR
        df['feature_atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['feature_atr_normalized'] = df['feature_atr'] / df['close']
        
        # ADX (trend strength)
        df['feature_adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Stochastic
        stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'])
        df['feature_stoch_k'] = stoch_k
        df['feature_stoch_d'] = stoch_d
        
        # Williams %R
        df['feature_williams_r'] = ta.WILLR(df['high'], df['low'], df['close'])
        
        # CCI
        df['feature_cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Money Flow Index
        df['feature_mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        return df
        
    async def _add_microstructure_features(self, df: pd.DataFrame, volume_data: pd.DataFrame = None) -> pd.DataFrame:
        """Add market microstructure features"""
        
        # Volume features
        if 'volume' in df.columns:
            df['feature_volume_ma'] = df['volume'].rolling(20).mean()
            df['feature_volume_ratio'] = df['volume'] / df['feature_volume_ma']
            df['feature_volume_std'] = df['volume'].rolling(10).std()
            
            # Volume Price Trend
            df['feature_vpt'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
            
            # On Balance Volume
            df['feature_obv'] = ta.OBV(df['close'], df['volume'])
            
            # Volume oscillator
            df['feature_volume_oscillator'] = (df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean() - 1) * 100
            
        # VWAP
        df['feature_vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['feature_vwap_deviation'] = (df['close'] - df['feature_vwap']) / df['feature_vwap']
        
        # True Range features
        df['feature_true_range'] = ta.TRANGE(df['high'], df['low'], df['close'])
        df['feature_tr_normalized'] = df['feature_true_range'] / df['close']
        
        # Efficiency Ratio
        df['feature_efficiency_ratio'] = abs(df['close'] - df['close'].shift(10)) / df['feature_true_range'].rolling(10).sum()
        
        # Tick data approximation (using OHLC)
        df['feature_tick_direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['feature_tick_momentum'] = df['feature_tick_direction'].rolling(5).sum()
        
        return df
        
    async def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        
        # Rolling statistics
        for window in [5, 10, 20]:
            # Volatility
            df[f'feature_volatility_{window}d'] = df['feature_return_1d'].rolling(window).std()
            
            # Skewness
            df[f'feature_skewness_{window}d'] = df['feature_return_1d'].rolling(window).skew()
            
            # Kurtosis
            df[f'feature_kurtosis_{window}d'] = df['feature_return_1d'].rolling(window).kurt()
            
            # Min/Max
            df[f'feature_min_{window}d'] = df['close'].rolling(window).min()
            df[f'feature_max_{window}d'] = df['close'].rolling(window).max()
            
            # Percentiles
            df[f'feature_q25_{window}d'] = df['close'].rolling(window).quantile(0.25)
            df[f'feature_q75_{window}d'] = df['close'].rolling(window).quantile(0.75)
            
        # Z-scores
        df['feature_price_zscore_20d'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['feature_volume_zscore_20d'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        # Autocorrelation
        df['feature_autocorr_1d'] = df['feature_return_1d'].rolling(20).apply(lambda x: x.autocorr(lag=1))
        df['feature_autocorr_5d'] = df['feature_return_1d'].rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        # Hurst Exponent (simplified calculation)
        def hurst_exponent(ts):
            try:
                lags = range(2, min(20, len(ts)))
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
                
        df['feature_hurst_20d'] = df['close'].rolling(20).apply(hurst_exponent)
        
        return df
        
    async def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Day of week
            df['feature_day_of_week'] = df['date'].dt.dayofweek
            df['feature_is_monday'] = (df['feature_day_of_week'] == 0).astype(int)
            df['feature_is_friday'] = (df['feature_day_of_week'] == 4).astype(int)
            
            # Month
            df['feature_month'] = df['date'].dt.month
            df['feature_quarter'] = df['date'].dt.quarter
            
            # Week of year
            df['feature_week_of_year'] = df['date'].dt.isocalendar().week
            
            # Days from start/end of month
            df['feature_days_from_month_start'] = df['date'].dt.day
            df['feature_days_to_month_end'] = df['date'].dt.days_in_month - df['date'].dt.day
            
        # Time-based momentum
        df['feature_momentum_5d'] = (df['close'] / df['close'].shift(5) - 1) * 100
        df['feature_momentum_20d'] = (df['close'] / df['close'].shift(20) - 1) * 100
        
        # Seasonal patterns
        df['feature_seasonal_strength'] = np.sin(2 * np.pi * df.index / 252)  # Annual cycle
        df['feature_monthly_cycle'] = np.sin(2 * np.pi * df.index / 21)  # Monthly cycle
        
        return df
        
    async def _add_relative_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative strength features"""
        
        # Note: In a real implementation, you would compare against benchmark
        # For now, we'll create proxy features
        
        # Relative strength vs moving average
        df['feature_rs_vs_ma20'] = df['close'] / df['feature_sma_20'] - 1
        df['feature_rs_vs_ma50'] = df['close'] / df['feature_sma_50'] - 1
        
        # Price rank (percentile rank over rolling window)
        df['feature_price_rank_20d'] = df['close'].rolling(20).rank(pct=True)
        df['feature_price_rank_50d'] = df['close'].rolling(50).rank(pct=True)
        
        # Volume rank
        df['feature_volume_rank_20d'] = df['volume'].rolling(20).rank(pct=True)
        
        # Return rank
        df['feature_return_rank_20d'] = df['feature_return_1d'].rolling(20).rank(pct=True)
        
        return df
        
    async def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime features"""
        
        # Different volatility measures
        df['feature_parkinson_vol'] = np.sqrt(252) * np.sqrt(
            np.log(df['high'] / df['low']).rolling(20).mean()
        )
        
        df['feature_garman_klass_vol'] = np.sqrt(252) * np.sqrt(
            0.5 * (np.log(df['high'] / df['low']))**2 - 
            (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']))**2
        ).rolling(20).mean()
        
        # Volatility regimes
        df['feature_vol_regime'] = pd.cut(
            df['feature_volatility_20d'], 
            bins=3, 
            labels=[0, 1, 2]
        ).astype(float)
        
        # Volatility breakouts
        vol_ma = df['feature_volatility_20d'].rolling(50).mean()
        vol_std = df['feature_volatility_20d'].rolling(50).std()
        df['feature_vol_breakout'] = (df['feature_volatility_20d'] - vol_ma) / vol_std
        
        # GARCH-like features
        df['feature_vol_persistence'] = df['feature_volatility_20d'].rolling(5).std()
        
        return df
        
    async def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        
        # Candlestick patterns (simplified)
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        
        df['feature_body_size'] = body_size / df['close']
        df['feature_upper_shadow'] = upper_shadow / df['close']
        df['feature_lower_shadow'] = lower_shadow / df['close']
        
        # Doji detection
        df['feature_is_doji'] = (body_size < (upper_shadow + lower_shadow) * 0.1).astype(int)
        
        # Hammer/Hanging man
        df['feature_is_hammer'] = (
            (lower_shadow > 2 * body_size) & 
            (upper_shadow < 0.5 * body_size)
        ).astype(int)
        
        # Support/Resistance levels
        highs = df['high'].rolling(10, center=True).max()
        lows = df['low'].rolling(10, center=True).min()
        
        df['feature_near_resistance'] = (df['close'] / highs > 0.98).astype(int)
        df['feature_near_support'] = (df['close'] / lows < 1.02).astype(int)
        
        # Breakout detection
        df['feature_breakout_up'] = (df['close'] > df['high'].rolling(20).max().shift(1)).astype(int)
        df['feature_breakout_down'] = (df['close'] < df['low'].rolling(20).min().shift(1)).astype(int)
        
        # Trend detection
        trend_window = 20
        df['feature_uptrend'] = (
            df['close'] > df['close'].rolling(trend_window).mean()
        ).astype(int)
        
        return df
        
    async def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset features (simplified for single asset)"""
        
        # Market correlation proxy (using autocorrelation)
        df['feature_market_corr'] = df['feature_return_1d'].rolling(20).apply(
            lambda x: x.corr(df['feature_return_1d'].shift(1).rolling(20).mean())
        )
        
        # Beta proxy (volatility relative to historical average)
        hist_vol = df['feature_volatility_20d'].rolling(100).mean()
        df['feature_beta_proxy'] = df['feature_volatility_20d'] / hist_vol
        
        # Sector rotation proxy
        df['feature_sector_momentum'] = df['feature_momentum_20d'].rolling(5).rank(pct=True)
        
        return df
        
    async def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction and derived features"""
        
        # Price-Volume interactions
        df['feature_price_volume_trend'] = df['feature_return_1d'] * df['feature_volume_ratio']
        
        # Volatility-Momentum interactions
        df['feature_vol_momentum'] = df['feature_volatility_20d'] * df['feature_momentum_5d']
        
        # RSI-Volume interaction
        df['feature_rsi_volume'] = df['feature_rsi'] * df['feature_volume_ratio']
        
        # Multiple timeframe features
        df['feature_short_long_ma_ratio'] = df['feature_sma_5'] / df['feature_sma_20']
        df['feature_ma_convergence'] = abs(df['feature_sma_5'] - df['feature_sma_20']) / df['feature_sma_20']
        
        # Momentum-Volatility ratio
        df['feature_momentum_vol_ratio'] = abs(df['feature_momentum_5d']) / df['feature_volatility_5d']
        
        return df
        
    async def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        # Handle infinite values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values for continuity
        df[feature_cols] = df[feature_cols].fillna(method='ffill')
        
        # Backward fill remaining NaN values
        df[feature_cols] = df[feature_cols].fillna(method='bfill')
        
        # Final cleanup - replace any remaining NaN with 0
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Cap extreme values
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32']:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = np.clip(df[col], q01, q99)
                
        logger.info(f"Cleaned {len(feature_cols)} features")
        
        return df
        
    async def select_features(self, df: pd.DataFrame, target_col: str, n_features: int = 50) -> List[str]:
        """Select most important features"""
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return feature_cols[:n_features]
            
        # Remove features with low variance
        from sklearn.feature_selection import VarianceThreshold
        
        variance_selector = VarianceThreshold(threshold=0.01)
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        try:
            X_var_selected = variance_selector.fit_transform(X)
            selected_features = [feature_cols[i] for i in range(len(feature_cols)) 
                               if variance_selector.get_support()[i]]
            
            # Mutual information feature selection
            from sklearn.feature_selection import mutual_info_regression, SelectKBest
            
            mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, len(selected_features)))
            mi_selector.fit(X[selected_features], y)
            
            final_features = [selected_features[i] for i in range(len(selected_features)) 
                            if mi_selector.get_support()[i]]
            
            # Store feature importance
            feature_scores = dict(zip(selected_features, mi_selector.scores_))
            self.feature_importance = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
            
            logger.info(f"Selected {len(final_features)} features from {len(feature_cols)} original features")
            
            return final_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return feature_cols[:n_features]
            
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of engineered features"""
        
        return {
            'feature_importance': self.feature_importance,
            'feature_categories': {
                'price_features': len([f for f in self.feature_importance.keys() if 'return' in f or 'price' in f]),
                'technical_features': len([f for f in self.feature_importance.keys() if any(x in f for x in ['rsi', 'macd', 'bb', 'sma', 'ema'])]),
                'volume_features': len([f for f in self.feature_importance.keys() if 'volume' in f]),
                'volatility_features': len([f for f in self.feature_importance.keys() if 'vol' in f or 'atr' in f]),
                'statistical_features': len([f for f in self.feature_importance.keys() if any(x in f for x in ['skew', 'kurt', 'zscore'])]),
                'pattern_features': len([f for f in self.feature_importance.keys() if any(x in f for x in ['breakout', 'trend', 'doji', 'hammer'])])
            },
            'total_features': len(self.feature_importance)
        }
