"""
Advanced Trade Analysis Module
Comprehensive analysis of trading patterns, performance, and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from config import Config
from database.db_manager import DatabaseManager
from utils.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class TradeAnalysis:
    """
    Advanced trade analysis and pattern recognition
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        self.statistical_analysis = StatisticalAnalysis()
        
        # Analysis cache
        self.trade_cache = {}
        self.pattern_cache = {}
        self.optimization_cache = {}
        
    async def initialize(self):
        """Initialize trade analysis module"""
        logger.info("Initializing Trade Analysis...")
        
        try:
            # Initialize analysis tables
            await self._initialize_analysis_tables()
            
            # Load recent trade data
            await self._load_trade_data()
            
            logger.info("Trade Analysis initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Trade Analysis: {e}")
            
    async def _initialize_analysis_tables(self):
        """Initialize database tables for trade analysis"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS trade_patterns (
                id SERIAL PRIMARY KEY,
                pattern_id VARCHAR(50) UNIQUE NOT NULL,
                pattern_type VARCHAR(50) NOT NULL,
                pattern_name VARCHAR(100) NOT NULL,
                description TEXT,
                frequency INTEGER DEFAULT 0,
                avg_return DECIMAL(8,4),
                win_rate DECIMAL(8,4),
                confidence_score DECIMAL(8,4),
                parameters JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trade_clusters (
                id SERIAL PRIMARY KEY,
                cluster_id VARCHAR(50) NOT NULL,
                trade_id VARCHAR(50) NOT NULL,
                cluster_type VARCHAR(50) NOT NULL,
                cluster_label INTEGER,
                distance_to_center DECIMAL(8,4),
                cluster_performance DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS optimization_results (
                id SERIAL PRIMARY KEY,
                optimization_id VARCHAR(50) UNIQUE NOT NULL,
                strategy_name VARCHAR(100) NOT NULL,
                parameter_set JSON NOT NULL,
                backtest_return DECIMAL(8,4),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(8,4),
                win_rate DECIMAL(8,4),
                total_trades INTEGER,
                optimization_score DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            await self.db_manager.execute_query(table_sql)
            
    async def _load_trade_data(self, days_back: int = 90):
        """Load recent trade data for analysis"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = """
            SELECT t.*, ai.confidence as ai_confidence, ai.rationale
            FROM trades t
            LEFT JOIN ai_decisions ai ON t.trade_id = ai.symbol
            WHERE t.entry_date >= %s
            ORDER BY t.entry_date DESC
            """
            
            self.trade_cache = await self.db_manager.fetch_dataframe(query, (cutoff_date,))
            
            logger.info(f"Loaded {len(self.trade_cache)} trades for analysis")
            
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
            
    async def analyze_trade_patterns(self) -> Dict[str, Any]:
        """Analyze trading patterns and identify successful strategies"""
        
        logger.info("Analyzing trade patterns...")
        
        try:
            if self.trade_cache.empty:
                return {'error': 'No trade data available'}
                
            # Pattern analysis
            patterns = {}
            
            # 1. Time-based patterns
            patterns['time_patterns'] = await self._analyze_time_patterns()
            
            # 2. Entry/Exit patterns
            patterns['entry_exit_patterns'] = await self._analyze_entry_exit_patterns()
            
            # 3. Market condition patterns
            patterns['market_condition_patterns'] = await self._analyze_market_condition_patterns()
            
            # 4. Performance clustering
            patterns['performance_clusters'] = await self._analyze_performance_clusters()
            
            # 5. AI confidence patterns
            patterns['ai_confidence_patterns'] = await self._analyze_ai_confidence_patterns()
            
            # 6. Symbol-specific patterns
            patterns['symbol_patterns'] = await self._analyze_symbol_patterns()
            
            # Store patterns
            await self._store_patterns(patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {e}")
            return {'error': str(e)}
            
    async def _analyze_time_patterns(self) -> Dict[str, Any]:
        """Analyze time-based trading patterns"""
        
        try:
            df = self.trade_cache.copy()
            
            if df.empty:
                return {}
                
            # Convert dates
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])
            
            # Extract time features
            df['entry_hour'] = df['entry_date'].dt.hour
            df['entry_day_of_week'] = df['entry_date'].dt.dayofweek
            df['entry_month'] = df['entry_date'].dt.month
            df['hold_days'] = (df['exit_date'] - df['entry_date']).dt.days
            
            patterns = {}
            
            # Hour of day analysis
            hourly_performance = df.groupby('entry_hour').agg({
                'pnl': ['mean', 'count'],
                'pnl_pct': 'mean'
            }).round(4)
            
            patterns['best_entry_hours'] = hourly_performance.sort_values(
                ('pnl_pct', 'mean'), ascending=False
            ).head(3).to_dict()
            
            # Day of week analysis
            daily_performance = df.groupby('entry_day_of_week').agg({
                'pnl': ['mean', 'count'],
                'pnl_pct': 'mean'
            }).round(4)
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            patterns['best_entry_days'] = {
                day_names[i]: daily_performance.loc[i].to_dict() 
                for i in range(min(5, len(daily_performance)))
            }
            
            # Hold time analysis
            hold_time_performance = df.groupby('hold_days').agg({
                'pnl': ['mean', 'count'],
                'pnl_pct': 'mean'
            }).round(4)
            
            patterns['optimal_hold_times'] = hold_time_performance.sort_values(
                ('pnl_pct', 'mean'), ascending=False
            ).head(5).to_dict()
            
            # Month analysis
            monthly_performance = df.groupby('entry_month').agg({
                'pnl': ['mean', 'count'],
                'pnl_pct': 'mean'
            }).round(4)
            
            patterns['seasonal_patterns'] = monthly_performance.to_dict()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {}
            
    async def _analyze_entry_exit_patterns(self) -> Dict[str, Any]:
        """Analyze entry and exit patterns"""
        
        try:
            df = self.trade_cache.copy()
            
            if df.empty or 'exit_reason' not in df.columns:
                return {}
                
            patterns = {}
            
            # Exit reason analysis
            exit_performance = df.groupby('exit_reason').agg({
                'pnl': ['mean', 'count'],
                'pnl_pct': ['mean', 'std'],
                'hold_days': 'mean'
            }).round(4)
            
            patterns['exit_reason_performance'] = exit_performance.to_dict()
            
            # Price movement at entry
            if 'entry_price' in df.columns:
                # Calculate price momentum at entry (simplified)
                df['entry_momentum'] = df['pnl_pct'].apply(
                    lambda x: 'POSITIVE' if x > 1 else 'NEGATIVE' if x < -1 else 'NEUTRAL'
                )
                
                momentum_performance = df.groupby('entry_momentum').agg({
                    'pnl': ['mean', 'count'],
                    'pnl_pct': 'mean'
                }).round(4)
                
                patterns['entry_momentum_patterns'] = momentum_performance.to_dict()
                
            # Stop loss vs take profit effectiveness
            stop_loss_trades = df[df['exit_reason'] == 'STOP_LOSS']
            take_profit_trades = df[df['exit_reason'] == 'TAKE_PROFIT']
            
            if len(stop_loss_trades) > 0 and len(take_profit_trades) > 0:
                patterns['risk_reward_effectiveness'] = {
                    'stop_loss_avg_loss': stop_loss_trades['pnl_pct'].mean(),
                    'take_profit_avg_gain': take_profit_trades['pnl_pct'].mean(),
                    'stop_loss_frequency': len(stop_loss_trades) / len(df),
                    'take_profit_frequency': len(take_profit_trades) / len(df),
                    'risk_reward_ratio': abs(take_profit_trades['pnl_pct'].mean() / stop_loss_trades['pnl_pct'].mean()) if stop_loss_trades['pnl_pct'].mean() != 0 else 0
                }
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing entry/exit patterns: {e}")
            return {}
            
    async def _analyze_market_condition_patterns(self) -> Dict[str, Any]:
        """Analyze performance under different market conditions"""
        
        try:
            df = self.trade_cache.copy()
            
            if df.empty:
                return {}
                
            patterns = {}
            
            # Volatility-based market conditions
            if 'pnl_pct' in df.columns:
                # Calculate market volatility proxy
                df['volatility_proxy'] = df['pnl_pct'].rolling(20, min_periods=5).std()
                df['volatility_regime'] = pd.cut(
                    df['volatility_proxy'], 
                    bins=3, 
                    labels=['LOW_VOL', 'MEDIUM_VOL', 'HIGH_VOL']
                )
                
                vol_performance = df.groupby('volatility_regime').agg({
                    'pnl': ['mean', 'count'],
                    'pnl_pct': ['mean', 'std']
                }).round(4)
                
                patterns['volatility_regime_performance'] = vol_performance.to_dict()
                
            # AI confidence in different market conditions
            if 'ai_confidence' in df.columns:
                df['confidence_level'] = pd.cut(
                    df['ai_confidence'],
                    bins=[0, 0.6, 0.8, 1.0],
                    labels=['LOW_CONFIDENCE', 'MEDIUM_CONFIDENCE', 'HIGH_CONFIDENCE']
                )
                
                confidence_performance = df.groupby('confidence_level').agg({
                    'pnl': ['mean', 'count'],
                    'pnl_pct': 'mean'
                }).round(4)
                
                patterns['ai_confidence_performance'] = confidence_performance.to_dict()
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing market condition patterns: {e}")
            return {}
            
    async def _analyze_performance_clusters(self) -> Dict[str, Any]:
        """Cluster trades by performance characteristics"""
        
        try:
            df = self.trade_cache.copy()
            
            if df.empty or len(df) < 10:
                return {}
                
            # Prepare features for clustering
            feature_columns = []
            
            if 'pnl_pct' in df.columns:
                feature_columns.append('pnl_pct')
            if 'hold_days' in df.columns:
                feature_columns.append('hold_days')
            if 'ai_confidence' in df.columns:
                feature_columns.append('ai_confidence')
                
            if not feature_columns:
                return {}
                
            # Clean data
            cluster_data = df[feature_columns].fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform clustering
            n_clusters = min(5, len(df) // 5)  # Ensure reasonable cluster size
            
            if n_clusters < 2:
                return {}
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Analyze clusters
            df['cluster'] = cluster_labels
            
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_trades = df[df['cluster'] == cluster_id]
                
                if len(cluster_trades) > 0:
                    cluster_analysis[f'cluster_{cluster_id}'] = {
                        'size': len(cluster_trades),
                        'avg_pnl_pct': cluster_trades['pnl_pct'].mean() if 'pnl_pct' in cluster_trades.columns else 0,
                        'avg_hold_days': cluster_trades['hold_days'].mean() if 'hold_days' in cluster_trades.columns else 0,
                        'avg_ai_confidence': cluster_trades['ai_confidence'].mean() if 'ai_confidence' in cluster_trades.columns else 0,
                        'win_rate': (cluster_trades['pnl'] > 0).mean() * 100 if 'pnl' in cluster_trades.columns else 0,
                        'characteristics': self._describe_cluster_characteristics(cluster_trades)
                    }
                    
            # Store clustering results
            await self._store_clustering_results(df, 'performance_clustering')
            
            return {
                'n_clusters': n_clusters,
                'cluster_analysis': cluster_analysis,
                'silhouette_score': self._calculate_silhouette_score(scaled_data, cluster_labels)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance clusters: {e}")
            return {}
            
    def _describe_cluster_characteristics(self, cluster_data: pd.DataFrame) -> str:
        """Describe characteristics of a trade cluster"""
        
        try:
            characteristics = []
            
            if 'pnl_pct' in cluster_data.columns:
                avg_return = cluster_data['pnl_pct'].mean()
                if avg_return > 2:
                    characteristics.append("High-return trades")
                elif avg_return < -1:
                    characteristics.append("Loss-making trades")
                else:
                    characteristics.append("Moderate-return trades")
                    
            if 'hold_days' in cluster_data.columns:
                avg_hold = cluster_data['hold_days'].mean()
                if avg_hold < 2:
                    characteristics.append("Short-term holds")
                elif avg_hold > 5:
                    characteristics.append("Long-term holds")
                else:
                    characteristics.append("Medium-term holds")
                    
            if 'ai_confidence' in cluster_data.columns:
                avg_confidence = cluster_data['ai_confidence'].mean()
                if avg_confidence > 0.8:
                    characteristics.append("High AI confidence")
                elif avg_confidence < 0.6:
                    characteristics.append("Low AI confidence")
                else:
                    characteristics.append("Medium AI confidence")
                    
            return ", ".join(characteristics) if characteristics else "Mixed characteristics"
            
        except Exception:
            return "Unable to characterize"
            
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(data, labels)
        except Exception:
            return 0.0
            
    async def _analyze_ai_confidence_patterns(self) -> Dict[str, Any]:
        """Analyze AI confidence patterns and their relationship to performance"""
        
        try:
            df = self.trade_cache.copy()
            
            if df.empty or 'ai_confidence' not in df.columns:
                return {}
                
            patterns = {}
            
            # Confidence distribution
            confidence_bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
            df['confidence_bin'] = pd.cut(df['ai_confidence'], bins=confidence_bins)
            
            confidence_performance = df.groupby('confidence_bin').agg({
                'pnl': ['mean', 'count'],
                'pnl_pct': ['mean', 'std']
            }).round(4)
            
            patterns['confidence_vs_performance'] = confidence_performance.to_dict()
            
            # Confidence calibration
            actual_success = (df['pnl'] > 0).astype(int)
            
            calibration_data = []
            for bin_start in np.arange(0.5, 1.0, 0.1):
                bin_end = bin_start + 0.1
                bin_mask = (df['ai_confidence'] >= bin_start) & (df['ai_confidence'] < bin_end)
                
                if bin_mask.sum() > 0:
                    predicted_prob = df[bin_mask]['ai_confidence'].mean()
                    actual_prob = actual_success[bin_mask].mean()
                    
                    calibration_data.append({
                        'predicted_prob': predicted_prob,
                        'actual_prob': actual_prob,
                        'count': bin_mask.sum(),
                        'calibration_error': abs(predicted_prob - actual_prob)
                    })
                    
            patterns['confidence_calibration'] = calibration_data
            
            # Optimal confidence threshold
            thresholds = np.arange(0.5, 1.0, 0.05)
            threshold_analysis = []
            
            for threshold in thresholds:
                high_conf_trades = df[df['ai_confidence'] >= threshold]
                
                if len(high_conf_trades) > 0:
                    threshold_analysis.append({
                        'threshold': threshold,
                        'trade_count': len(high_conf_trades),
                        'win_rate': (high_conf_trades['pnl'] > 0).mean(),
                        'avg_return': high_conf_trades['pnl_pct'].mean(),
                        'sharpe_ratio': high_conf_trades['pnl_pct'].mean() / high_conf_trades['pnl_pct'].std() if high_conf_trades['pnl_pct'].std() > 0 else 0
                    })
                    
            patterns['optimal_threshold_analysis'] = threshold_analysis
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing AI confidence patterns: {e}")
            return {}
            
    async def _analyze_symbol_patterns(self) -> Dict[str, Any]:
        """Analyze symbol-specific trading patterns"""
        
        try:
            df = self.trade_cache.copy()
            
            if df.empty or 'symbol' not in df.columns:
                return {}
                
            patterns = {}
            
            # Symbol performance
            symbol_performance = df.groupby('symbol').agg({
                'pnl': ['sum', 'mean', 'count'],
                'pnl_pct': ['mean', 'std'],
                'hold_days': 'mean'
            }).round(4)
            
            # Filter symbols with significant activity
            min_trades = 3
            active_symbols = symbol_performance[symbol_performance[('pnl', 'count')] >= min_trades]
            
            patterns['top_performing_symbols'] = active_symbols.sort_values(
                ('pnl', 'sum'), ascending=False
            ).head(10).to_dict()
            
            patterns['most_traded_symbols'] = symbol_performance.sort_values(
                ('pnl', 'count'), ascending=False
            ).head(10).to_dict()
            
            # Symbol-specific patterns
            symbol_patterns = {}
            
            for symbol in active_symbols.index:
                symbol_trades = df[df['symbol'] == symbol]
                
                symbol_patterns[symbol] = {
                    'total_trades': len(symbol_trades),
                    'win_rate': (symbol_trades['pnl'] > 0).mean() * 100,
                    'avg_return': symbol_trades['pnl_pct'].mean(),
                    'volatility': symbol_trades['pnl_pct'].std(),
                    'avg_hold_time': symbol_trades['hold_days'].mean() if 'hold_days' in symbol_trades.columns else 0,
                    'best_trade': symbol_trades['pnl_pct'].max(),
                    'worst_trade': symbol_trades['pnl_pct'].min(),
                    'consistency_score': self._calculate_consistency_score(symbol_trades)
                }
                
            patterns['symbol_specific_patterns'] = symbol_patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing symbol patterns: {e}")
            return {}
            
    def _calculate_consistency_score(self, trades: pd.DataFrame) -> float:
        """Calculate consistency score for symbol trading"""
        
        try:
            if len(trades) < 3 or 'pnl_pct' not in trades.columns:
                return 0.0
                
            returns = trades['pnl_pct']
            
            # Consistency based on low volatility and positive skew
            volatility_score = 1 / (1 + returns.std()) if returns.std() > 0 else 1
            
            win_rate = (returns > 0).mean()
            
            # Penalize large losses
            max_loss = abs(returns.min()) if returns.min() < 0 else 0
            loss_penalty = 1 / (1 + max_loss / 10)  # Normalize by 10%
            
            consistency = (volatility_score * 0.4 + win_rate * 0.4 + loss_penalty * 0.2) * 100
            
            return min(consistency, 100)
            
        except Exception:
            return 0.0
            
    async def _store_patterns(self, patterns: Dict[str, Any]):
        """Store identified patterns in database"""
        
        try:
            # Store significant patterns
            pattern_records = []
            
            for pattern_type, pattern_data in patterns.items():
                if isinstance(pattern_data, dict):
                    pattern_id = f"{pattern_type}_{datetime.now().strftime('%Y%m%d')}"
                    
                    # Calculate pattern significance
                    significance = self._calculate_pattern_significance(pattern_data)
                    
                    pattern_records.append((
                        pattern_id,
                        pattern_type,
                        pattern_type.replace('_', ' ').title(),
                        f"Analysis of {pattern_type}",
                        1,  # frequency
                        significance.get('avg_return', 0),
                        significance.get('win_rate', 0),
                        significance.get('confidence', 0),
                        str(pattern_data)
                    ))
                    
            if pattern_records:
                insert_query = """
                INSERT INTO trade_patterns 
                (pattern_id, pattern_type, pattern_name, description, frequency,
                 avg_return, win_rate, confidence_score, parameters)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pattern_id) DO UPDATE SET
                    frequency = EXCLUDED.frequency,
                    avg_return = EXCLUDED.avg_return,
                    win_rate = EXCLUDED.win_rate,
                    confidence_score = EXCLUDED.confidence_score,
                    parameters = EXCLUDED.parameters,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                await self.db_manager.execute_many(insert_query, pattern_records)
                
        except Exception as e:
            logger.error(f"Error storing patterns: {e}")
            
    def _calculate_pattern_significance(self, pattern_data: Any) -> Dict[str, float]:
        """Calculate significance metrics for a pattern"""
        
        try:
            # Default values
            significance = {'avg_return': 0, 'win_rate': 0, 'confidence': 0}
            
            if isinstance(pattern_data, dict):
                # Try to extract meaningful metrics
                for key, value in pattern_data.items():
                    if isinstance(value, dict):
                        if 'mean' in value:
                            if 'avg_return' not in significance or significance['avg_return'] == 0:
                                significance['avg_return'] = value['mean']
                        if 'win_rate' in value:
                            significance['win_rate'] = value['win_rate']
                            
            # Confidence based on data quality
            significance['confidence'] = min(significance['avg_return'] / 10 + significance['win_rate'] / 100, 1.0)
            
            return significance
            
        except Exception:
            return {'avg_return': 0, 'win_rate': 0, 'confidence': 0}
            
    async def _store_clustering_results(self, df: pd.DataFrame, cluster_type: str):
        """Store clustering results in database"""
        
        try:
            cluster_records = []
            
            for _, row in df.iterrows():
                if 'trade_id' in row and 'cluster' in row:
                    cluster_records.append((
                        f"{cluster_type}_{datetime.now().strftime('%Y%m%d')}",
                        row['trade_id'],
                        cluster_type,
                        int(row['cluster']),
                        0.0,  # distance_to_center placeholder
                        row.get('pnl_pct', 0)
                    ))
                    
            if cluster_records:
                insert_query = """
                INSERT INTO trade_clusters 
                (cluster_id, trade_id, cluster_type, cluster_label, distance_to_center, cluster_performance)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """
                
                await self.db_manager.execute_many(insert_query, cluster_records)
                
        except Exception as e:
            logger.error(f"Error storing clustering results: {e}")
            
    async def optimize_strategy_parameters(self, strategy_name: str, parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search"""
        
        logger.info(f"Optimizing parameters for {strategy_name}...")
        
        try:
            from itertools import product
            
            # Generate parameter combinations
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            combinations = list(product(*param_values))
            
            optimization_results = []
            
            # Test each combination (simplified - would use actual backtesting)
            for combination in combinations[:20]:  # Limit combinations for performance
                params = dict(zip(param_names, combination))
                
                # Simulate backtest results
                backtest_result = await self._simulate_parameter_backtest(params)
                
                optimization_results.append({
                    'parameters': params,
                    'performance': backtest_result
                })
                
            # Find best parameters
            best_result = max(optimization_results, key=lambda x: x['performance']['optimization_score'])
            
            # Store optimization results
            await self._store_optimization_results(strategy_name, optimization_results)
            
            return {
                'strategy_name': strategy_name,
                'best_parameters': best_result['parameters'],
                'best_performance': best_result['performance'],
                'total_combinations_tested': len(optimization_results),
                'optimization_summary': self._summarize_optimization(optimization_results)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return {'error': str(e)}
            
    async def _simulate_parameter_backtest(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Simulate backtest results for parameter set"""
        
        try:
            # Simplified simulation based on historical trade performance
            df = self.trade_cache.copy()
            
            if df.empty:
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'optimization_score': 0
                }
                
            # Apply parameter effects (simplified)
            confidence_threshold = params.get('confidence_threshold', 0.7)
            position_size = params.get('position_size', 0.05)
            stop_loss = params.get('stop_loss', 0.02)
            
            # Filter trades by confidence
            if 'ai_confidence' in df.columns:
                filtered_trades = df[df['ai_confidence'] >= confidence_threshold]
            else:
                filtered_trades = df
                
            if len(filtered_trades) == 0:
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'optimization_score': 0
                }
                
            # Calculate performance metrics
            returns = filtered_trades['pnl_pct'] * position_size
            
            total_return = returns.sum()
            win_rate = (returns > 0).mean() * 100
            volatility = returns.std() if len(returns) > 1 else 0
            sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative = (1 + returns / 100).cumprod()
            max_drawdown = ((cumulative.expanding().max() - cumulative) / cumulative.expanding().max()).max() * 100
            
            # Optimization score (weighted combination)
            optimization_score = (
                total_return * 0.3 +
                sharpe_ratio * 10 * 0.3 +
                win_rate * 0.2 +
                (100 - max_drawdown) * 0.2
            )
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(filtered_trades),
                'optimization_score': optimization_score
            }
            
        except Exception as e:
            logger.error(f"Error simulating parameter backtest: {e}")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'optimization_score': 0
            }
            
    async def _store_optimization_results(self, strategy_name: str, results: List[Dict]):
        """Store optimization results in database"""
        
        try:
            optimization_records = []
            
            for i, result in enumerate(results):
                optimization_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d')}_{i}"
                params = result['parameters']
                performance = result['performance']
                
                optimization_records.append((
                    optimization_id,
                    strategy_name,
                    str(params),
                    performance['total_return'],
                    performance['sharpe_ratio'],
                    performance['max_drawdown'],
                    performance['win_rate'],
                    performance['total_trades'],
                    performance['optimization_score']
                ))
                
            if optimization_records:
                insert_query = """
                INSERT INTO optimization_results 
                (optimization_id, strategy_name, parameter_set, backtest_return,
                 sharpe_ratio, max_drawdown, win_rate, total_trades, optimization_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                await self.db_manager.execute_many(insert_query, optimization_records)
                
        except Exception as e:
            logger.error(f"Error storing optimization results: {e}")
            
    def _summarize_optimization(self, results: List[Dict]) -> Dict[str, Any]:
        """Summarize optimization results"""
        
        try:
            scores = [r['performance']['optimization_score'] for r in results]
            
            return {
                'total_combinations': len(results),
                'best_score': max(scores),
                'worst_score': min(scores),
                'avg_score': np.mean(scores),
                'score_std': np.std(scores),
                'improvement_potential': (max(scores) - min(scores)) / min(scores) * 100 if min(scores) > 0 else 0
            }
            
        except Exception:
            return {}
            
    async def generate_trade_insights(self) -> List[str]:
        """Generate actionable insights from trade analysis"""
        
        try:
            insights = []
            
            # Analyze recent patterns
            patterns = await self.analyze_trade_patterns()
            
            if 'time_patterns' in patterns:
                time_patterns = patterns['time_patterns']
                
                # Best entry hours
                if 'best_entry_hours' in time_patterns:
                    best_hours = time_patterns['best_entry_hours']
                    insights.append(f"💡 Consider focusing trades during optimal hours for better performance")
                    
            if 'ai_confidence_patterns' in patterns:
                confidence_patterns = patterns['ai_confidence_patterns']
                
                if 'optimal_threshold_analysis' in confidence_patterns:
                    threshold_data = confidence_patterns['optimal_threshold_analysis']
                    
                    if threshold_data:
                        # Find best threshold
                        best_threshold = max(threshold_data, key=lambda x: x.get('sharpe_ratio', 0))
                        insights.append(f"🎯 Consider raising AI confidence threshold to {best_threshold['threshold']:.2f} for better risk-adjusted returns")
                        
            if 'performance_clusters' in patterns:
                cluster_analysis = patterns['performance_clusters']
                
                if 'cluster_analysis' in cluster_analysis:
                    clusters = cluster_analysis['cluster_analysis']
                    
                    # Find best performing cluster characteristics
                    best_cluster = max(clusters.values(), key=lambda x: x.get('avg_pnl_pct', 0))
                    insights.append(f"🏆 Focus on trades with characteristics: {best_cluster.get('characteristics', 'N/A')}")
                    
            # Portfolio-level insights
            df = self.trade_cache.copy()
            
            if not df.empty:
                recent_performance = df.tail(20)
                
                if len(recent_performance) > 0:
                    recent_win_rate = (recent_performance['pnl'] > 0).mean() * 100
                    
                    if recent_win_rate < 50:
                        insights.append("⚠️ Recent win rate below 50% - consider reviewing strategy parameters")
                    elif recent_win_rate > 70:
                        insights.append("✅ Strong recent performance - current strategy is working well")
                        
            if not insights:
                insights.append("📊 Continue monitoring trade patterns for optimization opportunities")
                
            return insights
            
        except Exception as e:
            logger.error(f"Error generating trade insights: {e}")
            return ["❌ Error generating insights - check system logs"]
            
    async def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        
        try:
            df = self.trade_cache.copy()
            
            if df.empty:
                return {'status': 'No trade data available'}
                
            summary = {
                'data_summary': {
                    'total_trades': len(df),
                    'date_range': {
                        'start': df['entry_date'].min() if 'entry_date' in df.columns else None,
                        'end': df['entry_date'].max() if 'entry_date' in df.columns else None
                    },
                    'symbols_traded': df['symbol'].nunique() if 'symbol' in df.columns else 0
                },
                'performance_summary': {
                    'win_rate': (df['pnl'] > 0).mean() * 100 if 'pnl' in df.columns else 0,
                    'avg_return': df['pnl_pct'].mean() if 'pnl_pct' in df.columns else 0,
                    'total_pnl': df['pnl'].sum() if 'pnl' in df.columns else 0,
                    'best_trade': df['pnl_pct'].max() if 'pnl_pct' in df.columns else 0,
                    'worst_trade': df['pnl_pct'].min() if 'pnl_pct' in df.columns else 0
                },
                'analysis_capabilities': {
                    'pattern_analysis': True,
                    'clustering_analysis': True,
                    'optimization': True,
                    'insights_generation': True
                },
                'last_analysis': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {'error': str(e)}
