"""
Advanced Feature Engineering for Trading Signals
Creates sophisticated features from market data and microstructure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import pandas_ta as pta
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
        df['feature_rsi'] = pta.rsi(df['close'], length=14)
        df['feature_rsi_ma'] = df['feature_rsi'].rolling(5).mean()
        df['feature_rsi_std'] = df['feature_rsi'].rolling(10).std()
        
        # MACD
        macd_df = pta.macd(df['close'])
        df['feature_macd'] = macd_df['MACD_12_26_9']
        df['feature_macd_signal'] = macd_df['MACDs_12_26_9']
        df['feature_macd_histogram'] = macd_df['MACDh_12_26_9']
        df['feature_macd_cross'] = np.where(df['feature_macd'] > df['feature_macd_signal'], 1, 0)
        
        # Bollinger Bands
        bb_df = pta.bbands(df['close'])
        df['feature_bb_upper'] = bb_df['BBU_20_2.0']
        df['feature_bb_middle'] = bb_df['BBM_20_2.0']
        df['feature_bb_lower'] = bb_df['BBL_20_2.0']
        df['feature_bb_width'] = (df['feature_bb_upper'] - df['feature_bb_lower']) / df['feature_bb_middle']
        df['feature_bb_position'] = (df['close'] - df['feature_bb_lower']) / (df['feature_bb_upper'] - df['feature_bb_lower'])
        
        # Moving Averages
        df['feature_sma_5'] = pta.sma(df['close'], length=5)
        df['feature_sma_20'] = pta.sma(df['close'], length=20)
        df['feature_sma_50'] = pta.sma(df['close'], length=50)
        df['feature_ema_12'] = pta.ema(df['close'], length=12)
        df['feature_ema_26'] = pta.ema(df['close'], length=26)
        
        # MA relationships
        df['feature_price_vs_sma20'] = df['close'] / df['feature_sma_20'] - 1
        df['feature_sma5_vs_sma20'] = df['feature_sma_5'] / df['feature_sma_20'] - 1
        df['feature_sma20_vs_sma50'] = df['feature_sma_20'] / df['feature_sma_50'] - 1
        
        # ATR
        df['feature_atr'] = pta.atr(df['high'], df['low'], df['close'], length=14)
        df['feature_atr_normalized'] = df['feature_atr'] / df['close']
        
        # ADX (trend strength)
        df['feature_adx'] = pta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        
        # Stochastic
        stoch_df = pta.stoch(df['high'], df['low'], df['close'])
        df['feature_stoch_k'] = stoch_df['STOCHk_14_3_3']
        df['feature_stoch_d'] = stoch_df['STOCHd_14_3_3']
        
        # Williams %R
        df['feature_williams_r'] = pta.willr(df['high'], df['low'], df['close'], length=14)
        
        # CCI
        df['feature_cci'] = pta.cci(df['high'], df['low'], df['close'], length=14)
        
        # Money Flow Index
        df['feature_mfi'] = pta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        
        return df
        
    async def _add_microstructure_features(self, df: pd.DataFrame, volume_data: pd.DataFrame = None) -> pd.DataFrame:
        """Add market microstructure features"""
        
        # Volume features
        if 'volume' in df.columns:
            df['feature_volume_ma'] = df['volume'].rolling(20).mean()
            df['feature_volume_ratio'] = df['volume'] / df['feature_volume_ma']
            df['feature_volume_std'] = df['volume'].rolling(10).std()
            
            # Volume Price Trend
            df['feature_vpt'] = pta.ad(df['high'], df['low'], df['close'], df['volume'])
            
            # On Balance Volume
            df['feature_obv'] = pta.obv(df['close'], df['volume'])
            
            # Volume oscillator
            df['feature_volume_oscillator'] = (df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean() - 1) * 100
            
        # VWAP
        df['feature_vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['feature_vwap_deviation'] = (df['close'] - df['feature_vwap']) / df['feature_vwap']
        
        # True Range features
        df['feature_true_range'] = pta.trange(df['high'], df['low'], df['close'])
        df['feature_tr_normalized'] = df['feature_true_range'] / df['close']
        
        # Efficiency Ratio
        df['feature_efficiency_ratio'] = abs(df['close'] - df['close'].shift(10)) / df['feature_true_range'].rolling(10).sum()
        
        # Tick data approximation (using OHLC)
        df['feature_tick_direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['feature_tick_momentum'] = df['feature_tick_direction'].rolling(5).sum()
        
        return df
        
    # ... rest of the file unchanged ...
