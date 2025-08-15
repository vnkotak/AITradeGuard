"""
Statistical Analysis Utilities
Advanced statistical functions for trading analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import scipy.cluster.hierarchy as hierarchy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StatisticalAnalysis:
    """
    Advanced statistical analysis tools for trading
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_returns(self, prices: pd.Series, method: str = 'simple') -> pd.Series:
        """Calculate returns using different methods"""
        
        try:
            if method == 'simple':
                return prices.pct_change()
            elif method == 'log':
                return np.log(prices / prices.shift(1))
            elif method == 'continuous':
                return np.log(prices).diff()
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series(dtype=float)
            
    def calculate_volatility(self, returns: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
        """Calculate rolling volatility"""
        
        try:
            vol = returns.rolling(window).std()
            
            if annualize:
                vol = vol * np.sqrt(252)  # Annualize assuming 252 trading days
                
            return vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return pd.Series(dtype=float)
            
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio"""
        
        try:
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            return excess_returns.mean() / returns.std() * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
            
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sortino ratio"""
        
        try:
            excess_returns = returns - risk_free_rate / 252
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0:
                return np.inf
                
            downside_std = downside_returns.std()
            return excess_returns.mean() / downside_std * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
            
    def calculate_maximum_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics"""
        
        try:
            # Calculate cumulative returns
            cumulative = prices / prices.iloc[0]
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            max_drawdown = drawdown.min()
            max_drawdown_idx = drawdown.idxmin()
            
            # Find peak before max drawdown
            peak_idx = cumulative[:max_drawdown_idx].idxmax()
            
            # Find recovery point after max drawdown (if any)
            recovery_idx = None
            peak_value = running_max.loc[max_drawdown_idx]
            
            post_drawdown = cumulative[max_drawdown_idx:]
            recovery_points = post_drawdown[post_drawdown >= peak_value]
            
            if len(recovery_points) > 0:
                recovery_idx = recovery_points.index[0]
                
            # Calculate duration
            drawdown_duration = (max_drawdown_idx - peak_idx).days if hasattr(max_drawdown_idx - peak_idx, 'days') else 0
            
            recovery_duration = None
            if recovery_idx:
                recovery_duration = (recovery_idx - max_drawdown_idx).days if hasattr(recovery_idx - max_drawdown_idx, 'days') else 0
                
            return {
                'max_drawdown': abs(max_drawdown),
                'max_drawdown_pct': abs(max_drawdown) * 100,
                'peak_date': peak_idx,
                'trough_date': max_drawdown_idx,
                'recovery_date': recovery_idx,
                'drawdown_duration': drawdown_duration,
                'recovery_duration': recovery_duration,
                'current_drawdown': abs(drawdown.iloc[-1]) * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            return {'max_drawdown': 0, 'max_drawdown_pct': 0}
            
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        
        try:
            return np.percentile(returns, confidence_level * 100)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
            
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        
        try:
            var = self.calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0
            
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        
        try:
            # Align series
            aligned_data = pd.DataFrame({
                'asset': asset_returns,
                'market': market_returns
            }).dropna()
            
            if len(aligned_data) < 2:
                return 1.0
                
            covariance = aligned_data['asset'].cov(aligned_data['market'])
            market_variance = aligned_data['market'].var()
            
            return covariance / market_variance if market_variance > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0
            
    def calculate_alpha(self, asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Jensen's alpha"""
        
        try:
            beta = self.calculate_beta(asset_returns, market_returns)
            
            asset_excess = asset_returns.mean() - risk_free_rate / 252
            market_excess = market_returns.mean() - risk_free_rate / 252
            
            alpha = asset_excess - (beta * market_excess)
            
            return alpha * 252  # Annualize
            
        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return 0.0
            
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        
        try:
            return returns_df.corr()
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
            
    def calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling correlation between two series"""
        
        try:
            return series1.rolling(window).corr(series2)
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {e}")
            return pd.Series(dtype=float)
            
    def detect_outliers(self, data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """Detect outliers using various methods"""
        
        try:
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data.dropna()))
                outliers = pd.Series(z_scores > threshold, index=data.index)
                
            elif method == 'modified_zscore':
                median = data.median()
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
                
            else:
                raise ValueError(f"Unknown method: {method}")
                
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return pd.Series(False, index=data.index)
            
    def perform_normality_test(self, data: pd.Series) -> Dict[str, Any]:
        """Perform normality tests on data"""
        
        try:
            results = {}
            
            # Shapiro-Wilk test (for small samples)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data.dropna())
                results['shapiro'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
                
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(data.dropna(), 'norm')
            results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > 0.05
            }
            
            # Anderson-Darling test
            ad_result = stats.anderson(data.dropna(), dist='norm')
            results['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_level
            }
            
            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(data.dropna())
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > 0.05
            }
            
            # Summary statistics
            results['summary'] = {
                'mean': data.mean(),
                'std': data.std(),
                'skewness': stats.skew(data.dropna()),
                'kurtosis': stats.kurtosis(data.dropna()),
                'median': data.median(),
                'min': data.min(),
                'max': data.max()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in normality test: {e}")
            return {}
            
    def calculate_hurst_exponent(self, series: pd.Series) -> float:
        """Calculate Hurst exponent for mean reversion/trend following"""
        
        try:
            lags = range(2, min(100, len(series) // 2))
            
            # Calculate the variance of the differences
            variances = []
            for lag in lags:
                differences = series.diff(lag).dropna()
                variances.append(differences.var())
                
            # Fit log-log relationship
            log_lags = np.log(lags)
            log_variances = np.log(variances)
            
            slope, intercept = np.polyfit(log_lags, log_variances, 1)
            
            hurst = slope / 2
            
            return hurst
            
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5
            
    def perform_stationarity_test(self, series: pd.Series) -> Dict[str, Any]:
        """Perform Augmented Dickey-Fuller test for stationarity"""
        
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(series.dropna())
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error in stationarity test: {e}")
            return {'is_stationary': False}
            
    def calculate_autocorrelation(self, series: pd.Series, max_lags: int = 20) -> pd.Series:
        """Calculate autocorrelation function"""
        
        try:
            from statsmodels.tsa.stattools import acf
            
            autocorr = acf(series.dropna(), nlags=max_lags)
            return pd.Series(autocorr, index=range(len(autocorr)))
            
        except Exception as e:
            logger.error(f"Error calculating autocorrelation: {e}")
            return pd.Series(dtype=float)
            
    def perform_pca(self, data: pd.DataFrame, n_components: int = None) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data.fillna(0))
            
            if n_components is None:
                n_components = min(data.shape[0], data.shape[1])
                
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(scaled_data)
            
            return {
                'components': pd.DataFrame(
                    components, 
                    index=data.index,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                ),
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
                'feature_loadings': pd.DataFrame(
                    pca.components_.T,
                    index=data.columns,
                    columns=[f'PC{i+1}' for i in range(n_components)]
                )
            }
            
        except Exception as e:
            logger.error(f"Error in PCA: {e}")
            return {}
            
    def perform_clustering(self, data: pd.DataFrame, n_clusters: int = None, method: str = 'kmeans') -> Dict[str, Any]:
        """Perform clustering analysis"""
        
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data.fillna(0))
            
            if method == 'kmeans':
                if n_clusters is None:
                    # Use elbow method to find optimal clusters
                    n_clusters = self._find_optimal_clusters(scaled_data)
                    
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(scaled_data)
                
                return {
                    'labels': pd.Series(labels, index=data.index),
                    'centers': kmeans.cluster_centers_,
                    'n_clusters': n_clusters,
                    'inertia': kmeans.inertia_
                }
                
            elif method == 'hierarchical':
                linkage_matrix = hierarchy.linkage(scaled_data, method='ward')
                
                if n_clusters is None:
                    n_clusters = 3
                    
                labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                return {
                    'labels': pd.Series(labels - 1, index=data.index),  # Convert to 0-based
                    'linkage_matrix': linkage_matrix,
                    'n_clusters': n_clusters
                }
                
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {}
            
    def _find_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        
        try:
            inertias = []
            k_range = range(1, min(max_clusters + 1, len(data)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
                
            # Find elbow point
            if len(inertias) < 3:
                return 2
                
            # Calculate second derivative
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_derivatives.append(inertias[i-1] - 2*inertias[i] + inertias[i+1])
                
            # Find maximum second derivative (elbow)
            elbow_idx = np.argmax(second_derivatives) + 1
            
            return k_range[elbow_idx]
            
        except Exception:
            return 3  # Default
            
    def calculate_regime_probabilities(self, returns: pd.Series, n_regimes: int = 3) -> pd.DataFrame:
        """Calculate regime probabilities using simple volatility-based approach"""
        
        try:
            # Calculate rolling volatility
            vol = self.calculate_volatility(returns, window=20, annualize=False)
            
            # Define regime thresholds
            vol_low = vol.quantile(0.33)
            vol_high = vol.quantile(0.67)
            
            # Assign regimes
            regimes = pd.DataFrame(index=returns.index)
            regimes['low_vol'] = (vol <= vol_low).astype(float)
            regimes['medium_vol'] = ((vol > vol_low) & (vol <= vol_high)).astype(float)
            regimes['high_vol'] = (vol > vol_high).astype(float)
            
            # Smooth probabilities
            for col in regimes.columns:
                regimes[col] = regimes[col].rolling(5).mean().fillna(method='bfill')
                
            return regimes
            
        except Exception as e:
            logger.error(f"Error calculating regime probabilities: {e}")
            return pd.DataFrame()
            
    def calculate_risk_metrics(self, returns: pd.Series, confidence_levels: List[float] = [0.01, 0.05, 0.10]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        
        try:
            metrics = {}
            
            # Basic statistics
            metrics['mean'] = returns.mean()
            metrics['std'] = returns.std()
            metrics['skewness'] = stats.skew(returns.dropna())
            metrics['kurtosis'] = stats.kurtosis(returns.dropna())
            
            # VaR and CVaR for different confidence levels
            for conf_level in confidence_levels:
                var = self.calculate_var(returns, conf_level)
                cvar = self.calculate_cvar(returns, conf_level)
                
                metrics[f'var_{int(conf_level*100)}'] = var
                metrics[f'cvar_{int(conf_level*100)}'] = cvar
                
            # Semi-deviation (downside risk)
            negative_returns = returns[returns < 0]
            metrics['semi_deviation'] = negative_returns.std() if len(negative_returns) > 0 else 0
            
            # Maximum daily loss
            metrics['max_daily_loss'] = returns.min()
            metrics['max_daily_gain'] = returns.max()
            
            # Tail ratio
            metrics['tail_ratio'] = abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
            
            # Risk-return metrics
            metrics['return_to_risk_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Calmar ratio (if we have enough data)
            if len(returns) >= 252:
                annual_return = returns.mean() * 252
                max_dd = self.calculate_maximum_drawdown(returns.cumsum())['max_drawdown']
                metrics['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
            
    def monte_carlo_simulation(self, returns: pd.Series, n_simulations: int = 1000, n_periods: int = 252) -> Dict[str, Any]:
        """Perform Monte Carlo simulation on returns"""
        
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random scenarios
            random_returns = np.random.normal(mean_return, std_return, (n_simulations, n_periods))
            
            # Calculate cumulative returns for each simulation
            cumulative_returns = np.cumprod(1 + random_returns, axis=1)
            
            # Calculate final values
            final_values = cumulative_returns[:, -1]
            
            # Calculate statistics
            results = {
                'mean_final_value': np.mean(final_values),
                'median_final_value': np.median(final_values),
                'std_final_value': np.std(final_values),
                'percentiles': {
                    '5th': np.percentile(final_values, 5),
                    '25th': np.percentile(final_values, 25),
                    '75th': np.percentile(final_values, 75),
                    '95th': np.percentile(final_values, 95)
                },
                'probability_of_loss': (final_values < 1).mean(),
                'max_simulated_return': final_values.max(),
                'min_simulated_return': final_values.min(),
                'simulation_paths': cumulative_returns
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {}
            
    def calculate_information_coefficient(self, predictions: pd.Series, actual_returns: pd.Series) -> float:
        """Calculate Information Coefficient (correlation between predictions and actual returns)"""
        
        try:
            # Align series
            aligned_data = pd.DataFrame({
                'predictions': predictions,
                'actual': actual_returns
            }).dropna()
            
            if len(aligned_data) < 2:
                return 0.0
                
            correlation = aligned_data['predictions'].corr(aligned_data['actual'])
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating information coefficient: {e}")
            return 0.0
