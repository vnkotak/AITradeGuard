"""
Technical Indicators Module
Comprehensive technical analysis indicators for trading signals
"""

import numpy as np
import pandas as pd
import pandas_ta as pta
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Indicator parameters from config
        self.rsi_period = self.config.RSI_PERIOD
        self.macd_fast = self.config.MACD_FAST
        self.macd_slow = self.config.MACD_SLOW
        self.macd_signal = self.config.MACD_SIGNAL
        self.atr_period = self.config.ATR_PERIOD
        
    async def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given data"""
        
        logger.debug(f"Calculating technical indicators for {len(data)} data points")
        
        try:
            df = data.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in data")
                    return df
                    
            # Convert to numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Remove rows with NaN values in OHLCV
            df = df.dropna(subset=required_columns)
            
            if len(df) < 20:  # Need minimum data points
                logger.warning("Insufficient data for technical indicators")
                return df
                
            # Calculate indicators
            df = await self._calculate_trend_indicators(df)
            df = await self._calculate_momentum_indicators(df)
            df = await self._calculate_volatility_indicators(df)
            df = await self._calculate_volume_indicators(df)
            df = await self._calculate_support_resistance(df)
            df = await self._calculate_pattern_indicators(df)
            df = await self._calculate_custom_indicators(df)
            
            logger.debug(f"Technical indicators calculated successfully")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data
            
    async def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        
        try:
            # Simple Moving Averages
            df['sma_5'] = pta.sma(df['close'], length=5)
            df['sma_10'] = pta.sma(df['close'], length=10)
            df['sma_20'] = pta.sma(df['close'], length=20)
            df['sma_50'] = pta.sma(df['close'], length=50)
            df['sma_100'] = pta.sma(df['close'], length=100)
            df['sma_200'] = pta.sma(df['close'], length=200)
            
            # Exponential Moving Averages
            df['ema_5'] = pta.ema(df['close'], length=5)
            df['ema_10'] = pta.ema(df['close'], length=10)
            df['ema_20'] = pta.ema(df['close'], length=20)
            df['ema_50'] = pta.ema(df['close'], length=50)
            
            # MACD
            macd_df = pta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            df['macd'] = macd_df['MACD_12_26_9']
            df['macd_signal'] = macd_df['MACDs_12_26_9']
            df['macd_histogram'] = macd_df['MACDh_12_26_9']
            
            # Parabolic SAR (no direct pandas-ta, but can use ta.trend.psar if ta-lib is not present)
            # For now, skip or implement with custom logic if needed
            
            # Average Directional Index (ADX)
            adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx_df['ADX_14']
            df['plus_di'] = adx_df['DMP_14']
            df['minus_di'] = adx_df['DMN_14']
            
            # Aroon
            aroon_df = pta.aroon(df['high'], df['low'], length=14)
            df['aroon_up'] = aroon_df['AROONU_14']
            df['aroon_down'] = aroon_df['AROOND_14']
            df['aroon_osc'] = aroon_df['AROONOSC_14']
            
            # Linear Regression
            df['linear_reg'] = pta.linreg(df['close'], length=14)
            df['linear_reg_slope'] = pta.linreg_slope(df['close'], length=14)
            df['linear_reg_angle'] = pta.linreg_angle(df['close'], length=14)
            
            # Price channels
            df['highest_high_20'] = df['high'].rolling(20).max()
            df['lowest_low_20'] = df['low'].rolling(20).min()
            df['middle_channel'] = (df['highest_high_20'] + df['lowest_low_20']) / 2
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return df
            
    async def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum oscillators"""
        
        try:
            # RSI
            df['rsi'] = pta.rsi(df['close'], length=self.rsi_period)
            df['rsi_ma'] = df['rsi'].rolling(5).mean()
            
            # Stochastic
            stoch_df = pta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            df['stoch_k'] = stoch_df['STOCHk_14_3_3']
            df['stoch_d'] = stoch_df['STOCHd_14_3_3']
            
            # Williams %R
            df['williams_r'] = pta.willr(df['high'], df['low'], df['close'], length=14)
            
            # Commodity Channel Index (CCI)
            df['cci'] = pta.cci(df['high'], df['low'], df['close'], length=14)
            
            # Rate of Change (ROC)
            df['roc'] = pta.roc(df['close'], length=10)
            df['roc_ma'] = df['roc'].rolling(5).mean()
            
            # Momentum
            df['momentum'] = pta.mom(df['close'], length=10)
            
            # Price Oscillator (no direct pandas-ta, but PPO is similar to MACD; can skip if not needed)
            # Ultimate Oscillator (not available in pandas-ta directly, can use alternate calculation or skip)
            
            # Custom momentum indicators
            df['momentum_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
            df['momentum_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
            df['momentum_20'] = (df['close'] / df['close'].shift(20) - 1) * 100
            
            # Relative Momentum Index
            df['rmi'] = self._calculate_rmi(df['close'], 20, 5)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return df
            
    async def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators"""
        
        try:
            # Average True Range (ATR)
            df['atr'] = pta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
            df['atr_percent'] = (df['atr'] / df['close']) * 100
            
            # Bollinger Bands
            bb_df = pta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb_df['BBU_20_2.0']
            df['bb_middle'] = bb_df['BBM_20_2.0']
            df['bb_lower'] = bb_df['BBL_20_2.0']
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100
            df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])) * 100
            
            # Keltner Channels
            kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(df)
            df['kc_upper'] = kc_upper
            df['kc_middle'] = kc_middle
            df['kc_lower'] = kc_lower
            
            # Donchian Channels
            df['dc_upper'] = df['high'].rolling(20).max()
            df['dc_lower'] = df['low'].rolling(20).min()
            df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
            
            # Historical Volatility
            df['hist_vol'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            
            # Chaikin Volatility (use pandas-ta AD as proxy)
            df['chaikin_vol'] = pta.ad(df['high'], df['low'], df['close'], df['volume'])
            
            # Price channels and volatility bands
            df['volatility_bands_upper'] = df['sma_20'] + (df['atr'] * 2)
            df['volatility_bands_lower'] = df['sma_20'] - (df['atr'] * 2)
            
            # Normalized volatility
            df['normalized_vol'] = df['atr'] / df['close']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return df
            
    async def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        
        try:
            # Volume Moving Averages
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # On-Balance Volume (OBV)
            df['obv'] = pta.obv(df['close'], df['volume'])
            df['obv_ma'] = df['obv'].rolling(10).mean()
            
            # Accumulation/Distribution Line
            df['ad_line'] = pta.ad(df['high'], df['low'], df['close'], df['volume'])
            
            # Chaikin Money Flow
            df['cmf'] = self._calculate_cmf(df, 20)
            
            # Money Flow Index
            df['mfi'] = pta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            
            # Volume Price Trend (VPT)
            df['vpt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
            df['vpt'] = df['vpt'].cumsum()
            
            # Ease of Movement (not available in pandas-ta directly, can skip or implement)
            
            # Volume Weighted Average Price (VWAP)
            df['vwap'] = self._calculate_vwap(df)
            df['vwap_deviation'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
            
            # Volume Oscillator
            df['volume_osc'] = ((df['volume_sma_10'] - df['volume_sma_20']) / df['volume_sma_20']) * 100
            
            # Price Volume Trend
            df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return df
            
    async def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels"""
        
        try:
            # Pivot Points (Traditional)
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = (2 * df['pivot']) - df['low']
            df['s1'] = (2 * df['pivot']) - df['high']
            df['r2'] = df['pivot'] + (df['high'] - df['low'])
            df['s2'] = df['pivot'] - (df['high'] - df['low'])
            df['r3'] = df['high'] + 2 * (df['pivot'] - df['low'])
            df['s3'] = df['low'] - 2 * (df['high'] - df['pivot'])
            
            # Fibonacci Retracements
            df = self._calculate_fibonacci_levels(df)
            
            # Support and Resistance based on local extremes
            df['resistance_level'] = df['high'].rolling(20, center=True).max()
            df['support_level'] = df['low'].rolling(20, center=True).min()
            
            # Distance from support/resistance
            df['dist_from_resistance'] = ((df['resistance_level'] - df['close']) / df['close']) * 100
            df['dist_from_support'] = ((df['close'] - df['support_level']) / df['close']) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return df
            
    async def _calculate_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition indicators"""
        
        try:
            # Candlestick pattern recognition
            # pandas-ta does not offer direct candlestick pattern functions, so you may need to implement these manually or use simple feature engineering.
            df['doji'] = ((abs(df['open'] - df['close']) <= ((df['high'] - df['low']) * 0.1))).astype(int)
            df['hammer'] = (((df['close'] - df['low']) > 2 * abs(df['close'] - df['open'])) & ((df['high'] - df['close']) < 0.5 * abs(df['close'] - df['open']))).astype(int)
            df['hanging_man'] = (((df['open'] - df['low']) > 2 * abs(df['open'] - df['close'])) & ((df['high'] - df['open']) < 0.5 * abs(df['open'] - df['close']))).astype(int)
            # For other candlestick patterns, you can add custom logic
            
            # Price patterns
            df['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_lows'] = (df['low'] < df['low'].shift(1)).astype(int)
            df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
            df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
            
            # Breakout patterns
            df['breakout_up'] = (df['close'] > df['high'].rolling(20).max().shift(1)).astype(int)
            df['breakout_down'] = (df['close'] < df['low'].rolling(20).min().shift(1)).astype(int)
            
            # Gap analysis
            df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
            df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
            df['gap_fill'] = self._detect_gap_fill(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating pattern indicators: {e}")
            return df
            
    async def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom proprietary indicators"""
        
        try:
            # Custom Strength Index
            df['custom_strength'] = self._calculate_custom_strength_index(df)
            
            # Trend Quality Index
            df['trend_quality'] = self._calculate_trend_quality(df)
            
            # Volatility Adjusted Momentum
            df['vol_adj_momentum'] = (df['roc'] / df['hist_vol']) * 100
            
            # Price Efficiency Ratio
            df['efficiency_ratio'] = abs(df['close'] - df['close'].shift(10)) / df['atr'].rolling(10).sum()
            
            # Market Structure Index
            df['market_structure'] = self._calculate_market_structure(df)
            
            # Composite Momentum Score
            df['momentum_composite'] = (
                (df['rsi'] - 50) * 0.3 +
                np.sign(df['macd']) * 0.3 +
                (df['roc'] / 10) * 0.2 +
                np.sign(df['momentum']) * 0.2
            )
            
            # Volatility Regime
            df['vol_regime'] = pd.cut(
                df['hist_vol'], 
                bins=3, 
                labels=['LOW', 'MEDIUM', 'HIGH']
            )
            
            # Trend Regime
            trend_score = (
                (df['close'] > df['sma_20']).astype(int) +
                (df['sma_20'] > df['sma_50']).astype(int) +
                (df['adx'] > 25).astype(int)
            )
            df['trend_regime'] = pd.cut(
                trend_score,
                bins=[-1, 0, 1, 3],
                labels=['BEARISH', 'SIDEWAYS', 'BULLISH']
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating custom indicators: {e}")
            return df
            
    def _calculate_rmi(self, close: pd.Series, length: int, momentum_period: int) -> pd.Series:
        """Calculate Relative Momentum Index"""
        
        try:
            momentum = close.diff(momentum_period)
            
            positive_momentum = momentum.where(momentum > 0, 0)
            negative_momentum = momentum.where(momentum < 0, 0).abs()
            
            positive_avg = positive_momentum.rolling(length).mean()
            negative_avg = negative_momentum.rolling(length).mean()
            
            rmi = 100 - (100 / (1 + positive_avg / negative_avg))
            
            return rmi
            
        except Exception:
            return pd.Series(index=close.index, dtype=float)
            
    def _calculate_keltner_channels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        
        try:
            middle = df['close'].rolling(20).mean()
            atr = df['atr']
            
            upper = middle + (atr * 2)
            lower = middle - (atr * 2)
            
            return upper, middle, lower
            
        except Exception:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
            
    def _calculate_cmf(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        
        try:
            money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            
            cmf = money_flow_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
            
            return cmf
            
        except Exception:
            return pd.Series(index=df.index, dtype=float)
            
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            return vwap
            
        except Exception:
            return pd.Series(index=df.index, dtype=float)
            
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels"""
        
        try:
            # Calculate over rolling 50-period window
            high_50 = df['high'].rolling(50).max()
            low_50 = df['low'].rolling(50).min()
            
            diff = high_50 - low_50
            
            df['fib_23.6'] = high_50 - (diff * 0.236)
            df['fib_38.2'] = high_50 - (diff * 0.382)
            df['fib_50.0'] = high_50 - (diff * 0.500)
            df['fib_61.8'] = high_50 - (diff * 0.618)
            df['fib_78.6'] = high_50 - (diff * 0.786)
            
            return df
            
        except Exception:
            return df
            
    def _detect_gap_fill(self, df: pd.DataFrame) -> pd.Series:
        """Detect gap fill patterns"""
        
        try:
            gap_fill = pd.Series(0, index=df.index)
            
            for i in range(1, len(df)):
                # Check for gap up and subsequent fill
                if df.iloc[i-1]['gap_up'] == 1:
                    gap_high = df.iloc[i-1]['high']
                    # Look for fill in next few bars
                    for j in range(i, min(i+5, len(df))):
                        if df.iloc[j]['low'] <= gap_high:
                            gap_fill.iloc[j] = 1
                            break
                            
                # Check for gap down and subsequent fill
                elif df.iloc[i-1]['gap_down'] == 1:
                    gap_low = df.iloc[i-1]['low']
                    # Look for fill in next few bars
                    for j in range(i, min(i+5, len(df))):
                        if df.iloc[j]['high'] >= gap_low:
                            gap_fill.iloc[j] = -1
                            break
                            
            return gap_fill
            
        except Exception:
            return pd.Series(0, index=df.index)
            
    def _calculate_custom_strength_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate custom strength index"""
        
        try:
            # Combine multiple strength measures
            price_strength = (df['close'] - df['sma_20']) / df['sma_20'] * 100
            volume_strength = (df['volume'] - df['volume_sma_20']) / df['volume_sma_20'] * 100
            momentum_strength = df['rsi'] - 50
            
            strength_index = (price_strength * 0.4 + volume_strength * 0.3 + momentum_strength * 0.3)
            
            return strength_index
            
        except Exception:
            return pd.Series(0, index=df.index)
            
    def _calculate_trend_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality indicator"""
        
        try:
            # Measure consistency of trend
            price_above_ma = (df['close'] > df['sma_20']).astype(int)
            ma_slope = df['sma_20'].diff()
            
            trend_consistency = price_above_ma.rolling(10).mean()
            trend_strength = abs(ma_slope) / df['atr']
            
            trend_quality = trend_consistency * trend_strength
            
            return trend_quality
            
        except Exception:
            return pd.Series(0, index=df.index)
            
    def _calculate_market_structure(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market structure indicator"""
        
        try:
            # Identify swing highs and lows
            swing_high = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
            swing_low = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
            
            # Count structure breaks
            structure_breaks = swing_high.astype(int) - swing_low.astype(int)
            market_structure = structure_breaks.rolling(20).sum()
            
            return market_structure
            
        except Exception:
            return pd.Series(0, index=df.index)
            
    def get_signal_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of technical signals"""
        
        try:
            if df.empty:
                return {}
                
            latest = df.iloc[-1]
            
            signals = {
                'trend_signals': {
                    'sma_cross': 'BULLISH' if latest['close'] > latest['sma_20'] else 'BEARISH',
                    'macd': 'BULLISH' if latest['macd'] > latest['macd_signal'] else 'BEARISH',
                    'adx_strength': 'STRONG' if latest['adx'] > 25 else 'WEAK'
                },
                'momentum_signals': {
                    'rsi': 'OVERBOUGHT' if latest['rsi'] > 70 else 'OVERSOLD' if latest['rsi'] < 30 else 'NEUTRAL',
                    'stochastic': 'OVERBOUGHT' if latest['stoch_k'] > 80 else 'OVERSOLD' if latest['stoch_k'] < 20 else 'NEUTRAL',
                    'momentum': 'POSITIVE' if latest['momentum'] > 0 else 'NEGATIVE'
                },
                'volatility_signals': {
                    'bollinger_position': latest['bb_position'],
                    'atr_trend': 'INCREASING' if latest['atr'] > df['atr'].iloc[-5:].mean() else 'DECREASING',
                    'volatility_regime': latest.get('vol_regime', 'UNKNOWN')
                },
                'volume_signals': {
                    'volume_trend': 'ABOVE_AVERAGE' if latest['volume_ratio'] > 1.2 else 'BELOW_AVERAGE',
                    'obv_trend': 'BULLISH' if latest['obv'] > latest['obv_ma'] else 'BEARISH',
                    'mfi': 'OVERBOUGHT' if latest['mfi'] > 80 else 'OVERSOLD' if latest['mfi'] < 20 else 'NEUTRAL'
                },
                'composite_score': latest.get('momentum_composite', 0),
                'trend_regime': latest.get('trend_regime', 'UNKNOWN'),
                'custom_strength': latest.get('custom_strength', 0)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {}
