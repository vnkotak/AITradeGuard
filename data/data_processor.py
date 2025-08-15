"""
Advanced Data Processing Pipeline
Handles data cleaning, validation, and preprocessing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import ta

from config import Config
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Advanced data processing pipeline for market data
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager()
        
        # Data quality metrics
        self.quality_metrics = {}
        self.outlier_stats = {}
        
        # Preprocessing components
        self.scalers = {}
        self.imputers = {}
        
        # Data validation rules
        self.validation_rules = self._setup_validation_rules()
        
    def _setup_validation_rules(self) -> Dict:
        """Setup data validation rules"""
        return {
            'price_rules': {
                'min_price': 0.01,
                'max_price': 50000,
                'max_price_change': 0.20,  # 20% max daily change
                'min_volume': 1
            },
            'technical_rules': {
                'rsi_range': (0, 100),
                'macd_max_abs': 1000,
                'bb_width_max': 5.0,
                'atr_min': 0
            },
            'data_completeness': {
                'min_completion_rate': 0.95,
                'max_missing_consecutive': 3
            }
        }
        
    async def process_raw_data(self, raw_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Main data processing pipeline
        """
        logger.info(f"Processing raw data for {symbol}: {len(raw_data)} records")
        
        try:
            # Step 1: Data validation and cleaning
            cleaned_data = await self._clean_data(raw_data, symbol)
            
            # Step 2: Handle missing values
            imputed_data = await self._handle_missing_values(cleaned_data, symbol)
            
            # Step 3: Outlier detection and treatment
            outlier_treated_data = await self._handle_outliers(imputed_data, symbol)
            
            # Step 4: Data normalization
            normalized_data = await self._normalize_data(outlier_treated_data, symbol)
            
            # Step 5: Calculate quality metrics
            await self._calculate_quality_metrics(normalized_data, symbol)
            
            logger.info(f"Data processing completed for {symbol}: {len(normalized_data)} clean records")
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            return pd.DataFrame()
            
    async def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate raw data"""
        
        df = data.copy()
        original_len = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Validate price data
        price_cols = ['open', 'high', 'low', 'close']
        rules = self.validation_rules['price_rules']
        
        # Remove invalid prices
        for col in price_cols:
            if col in df.columns:
                df = df[
                    (df[col] >= rules['min_price']) & 
                    (df[col] <= rules['max_price'])
                ]
                
        # Check OHLC consistency
        if all(col in df.columns for col in price_cols):
            df = df[
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ]
            
        # Check for reasonable daily price changes
        if 'close' in df.columns:
            df['daily_return'] = df['close'].pct_change().abs()
            df = df[
                (df['daily_return'].isna()) | 
                (df['daily_return'] <= rules['max_price_change'])
            ]
            df = df.drop('daily_return', axis=1)
            
        # Validate volume
        if 'volume' in df.columns:
            df = df[df['volume'] >= rules['min_volume']]
            
        cleaned_len = len(df)
        logger.info(f"Data cleaning for {symbol}: {original_len} -> {cleaned_len} records")
        
        return df
        
    async def _handle_missing_values(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle missing values using advanced imputation"""
        
        df = data.copy()
        
        # Calculate missing value statistics
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df)) * 100
        
        # Check data completeness
        completion_rate = 1 - (missing_stats.sum() / (len(df) * len(df.columns)))
        
        if completion_rate < self.validation_rules['data_completeness']['min_completion_rate']:
            logger.warning(f"Low data completion rate for {symbol}: {completion_rate:.2%}")
            
        # Handle different types of missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0 and missing_stats[numeric_cols].sum() > 0:
            # Use KNN imputation for numeric columns
            imputer_key = f"{symbol}_numeric"
            
            if imputer_key not in self.imputers:
                self.imputers[imputer_key] = KNNImputer(n_neighbors=5)
                
            # Fit and transform
            df[numeric_cols] = self.imputers[imputer_key].fit_transform(df[numeric_cols])
            
        # Forward fill for time series continuity
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    async def _handle_outliers(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Detect and handle outliers using multiple methods"""
        
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_counts = {}
        
        for col in numeric_cols:
            if col == 'date' or col.startswith('date'):
                continue
                
            original_values = df[col].copy()
            
            # Method 1: IQR-based outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Method 2: Z-score based detection (for normal distributions)
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_threshold = 3
            
            # Method 3: Modified Z-score (robust to outliers)
            median = np.median(df[col].dropna())
            mad = np.median(np.abs(df[col].dropna() - median))
            modified_z_scores = 0.6745 * (df[col] - median) / mad
            modified_z_threshold = 3.5
            
            # Combine outlier detection methods
            iqr_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            z_outliers = z_scores > z_threshold
            modified_z_outliers = np.abs(modified_z_scores) > modified_z_threshold
            
            # Mark as outlier if detected by multiple methods
            outliers = iqr_outliers & (z_outliers | modified_z_outliers)
            
            outlier_count = outliers.sum()
            outlier_counts[col] = outlier_count
            
            if outlier_count > 0:
                # Winsorize outliers instead of removing them
                df.loc[outliers, col] = np.where(
                    df.loc[outliers, col] > upper_bound,
                    upper_bound,
                    lower_bound
                )
                
        self.outlier_stats[symbol] = outlier_counts
        
        total_outliers = sum(outlier_counts.values())
        if total_outliers > 0:
            logger.info(f"Handled {total_outliers} outliers for {symbol}")
            
        return df
        
    async def _normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize data using robust scaling"""
        
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Exclude date columns and already normalized features
        cols_to_normalize = [
            col for col in numeric_cols 
            if not col.startswith('date') and not col.endswith('_normalized')
        ]
        
        if len(cols_to_normalize) > 0:
            scaler_key = f"{symbol}_robust"
            
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = RobustScaler()
                
            # Fit and transform
            df[cols_to_normalize] = self.scalers[scaler_key].fit_transform(
                df[cols_to_normalize]
            )
            
        return df
        
    async def _calculate_quality_metrics(self, data: pd.DataFrame, symbol: str):
        """Calculate data quality metrics"""
        
        metrics = {
            'total_records': len(data),
            'completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'outliers_handled': sum(self.outlier_stats.get(symbol, {}).values()),
            'date_range': {
                'start': data['date'].min() if 'date' in data.columns else None,
                'end': data['date'].max() if 'date' in data.columns else None
            },
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'data_types': dict(data.dtypes.astype(str))
        }
        
        # Calculate statistical properties
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            metrics['statistics'] = {
                'mean_values': numeric_data.mean().to_dict(),
                'std_values': numeric_data.std().to_dict(),
                'skewness': numeric_data.skew().to_dict(),
                'kurtosis': numeric_data.kurtosis().to_dict()
            }
            
        self.quality_metrics[symbol] = metrics
        
    async def batch_process_symbols(self, symbols: List[str], lookback_days: int = 252) -> Dict[str, pd.DataFrame]:
        """Process multiple symbols in parallel"""
        
        logger.info(f"Batch processing {len(symbols)} symbols...")
        
        async def process_symbol(symbol: str) -> Tuple[str, pd.DataFrame]:
            try:
                # Get raw data
                raw_data = await self._get_raw_data(symbol, lookback_days)
                
                if raw_data.empty:
                    return symbol, pd.DataFrame()
                    
                # Process data
                processed_data = await self.process_raw_data(raw_data, symbol)
                return symbol, processed_data
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return symbol, pd.DataFrame()
                
        # Process symbols in parallel
        tasks = [process_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        processed_data = {symbol: data for symbol, data in results}
        
        successful_symbols = [s for s, d in processed_data.items() if not d.empty]
        logger.info(f"Successfully processed {len(successful_symbols)}/{len(symbols)} symbols")
        
        return processed_data
        
    async def _get_raw_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Get raw market data for a symbol"""
        
        query = """
        SELECT date, open, high, low, close, volume
        FROM market_data 
        WHERE symbol = %s 
        AND date >= %s
        ORDER BY date
        """
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        df = await self.db_manager.fetch_dataframe(query, (symbol, cutoff_date))
        return df
        
    async def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        
        validation_results = {
            'symbol': symbol,
            'passed': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check data completeness
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        
        if completeness < 95:
            validation_results['warnings'].append(f"Low data completeness: {completeness:.1f}%")
            
        if completeness < 80:
            validation_results['errors'].append(f"Critically low data completeness: {completeness:.1f}%")
            validation_results['passed'] = False
            
        # Check data recency
        if 'date' in data.columns:
            latest_date = pd.to_datetime(data['date'].max())
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 7:
                validation_results['warnings'].append(f"Data is {days_old} days old")
                
            if days_old > 30:
                validation_results['errors'].append(f"Data is critically outdated: {days_old} days")
                validation_results['passed'] = False
                
        # Check for sufficient data points
        if len(data) < 100:
            validation_results['warnings'].append(f"Limited data points: {len(data)}")
            
        if len(data) < 50:
            validation_results['errors'].append(f"Insufficient data points: {len(data)}")
            validation_results['passed'] = False
            
        # Check data distribution
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith('date'):
                continue
                
            # Check for constant values
            if data[col].nunique() <= 1:
                validation_results['errors'].append(f"Column {col} has constant values")
                validation_results['passed'] = False
                
            # Check for extreme skewness
            skewness = abs(data[col].skew())
            if skewness > 5:
                validation_results['warnings'].append(f"High skewness in {col}: {skewness:.2f}")
                
        # Generate recommendations
        if completeness < 95:
            validation_results['recommendations'].append("Consider improving data collection process")
            
        if len(validation_results['warnings']) > 5:
            validation_results['recommendations'].append("Multiple data quality issues detected - review data pipeline")
            
        return validation_results
        
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of data processing activities"""
        
        return {
            'symbols_processed': list(self.quality_metrics.keys()),
            'total_symbols': len(self.quality_metrics),
            'quality_metrics': self.quality_metrics,
            'outlier_statistics': self.outlier_stats,
            'scalers_fitted': list(self.scalers.keys()),
            'imputers_fitted': list(self.imputers.keys())
        }
        
    async def export_processed_data(self, symbol: str, data: pd.DataFrame, format: str = 'csv'):
        """Export processed data to various formats"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'csv':
            filename = f"processed_data/{symbol}_{timestamp}.csv"
            data.to_csv(filename, index=False)
            
        elif format.lower() == 'parquet':
            filename = f"processed_data/{symbol}_{timestamp}.parquet"
            data.to_parquet(filename, index=False)
            
        elif format.lower() == 'hdf':
            filename = f"processed_data/{symbol}_{timestamp}.h5"
            data.to_hdf(filename, key='data', mode='w')
            
        logger.info(f"Exported processed data for {symbol} to {filename}")
        return filename
