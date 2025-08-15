"""
Performance Analytics Module
Advanced performance analysis and reporting for trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from config import Config
from database.db_manager import DatabaseManager
from utils.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """
    Advanced performance analytics for trading strategies
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        self.statistical_analysis = StatisticalAnalysis()
        
        # Performance data
        self.portfolio_data = pd.DataFrame()
        self.trade_data = pd.DataFrame()
        self.benchmark_data = pd.DataFrame()
        
        # Metrics cache
        self.metrics_cache = {}
        self.analysis_cache = {}
        
    async def initialize(self):
        """Initialize performance analytics"""
        logger.info("Initializing Performance Analytics...")
        
        try:
            # Load recent performance data
            await self._load_performance_data()
            
            # Initialize analytics tables
            await self._initialize_analytics_tables()
            
            logger.info("Performance Analytics initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Performance Analytics: {e}")
            
    async def _initialize_analytics_tables(self):
        """Initialize database tables for analytics"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS performance_reports (
                id SERIAL PRIMARY KEY,
                report_id VARCHAR(50) UNIQUE NOT NULL,
                report_type VARCHAR(50) NOT NULL,
                period_start DATE NOT NULL,
                period_end DATE NOT NULL,
                total_return DECIMAL(8,4),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(8,4),
                win_rate DECIMAL(8,4),
                profit_factor DECIMAL(8,4),
                calmar_ratio DECIMAL(8,4),
                sortino_ratio DECIMAL(8,4),
                alpha DECIMAL(8,4),
                beta DECIMAL(8,4),
                information_ratio DECIMAL(8,4),
                treynor_ratio DECIMAL(8,4),
                total_trades INTEGER,
                report_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS attribution_analysis (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                strategy VARCHAR(50),
                symbol VARCHAR(20),
                contribution DECIMAL(8,4),
                weight DECIMAL(8,4),
                return_contribution DECIMAL(8,4),
                risk_contribution DECIMAL(8,4),
                alpha_contribution DECIMAL(8,4),
                beta_contribution DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            await self.db_manager.execute_query(table_sql)
            
    async def _load_performance_data(self, days_back: int = 365):
        """Load performance data for analysis"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Load portfolio snapshots
            portfolio_query = """
            SELECT * FROM portfolio_snapshots 
            WHERE date >= %s 
            ORDER BY date
            """
            
            self.portfolio_data = await self.db_manager.fetch_dataframe(
                portfolio_query, (cutoff_date,)
            )
            
            # Load trade data
            trade_query = """
            SELECT * FROM trades 
            WHERE entry_date >= %s 
            ORDER BY entry_date
            """
            
            self.trade_data = await self.db_manager.fetch_dataframe(
                trade_query, (cutoff_date,)
            )
            
            # Load benchmark data (Nifty)
            benchmark_query = """
            SELECT date, close as benchmark_price
            FROM market_data 
            WHERE symbol = %s AND date >= %s
            ORDER BY date
            """
            
            self.benchmark_data = await self.db_manager.fetch_dataframe(
                benchmark_query, (self.config.BENCHMARK_SYMBOL, cutoff_date)
            )
            
            logger.info(f"Loaded performance data: {len(self.portfolio_data)} portfolio snapshots, "
                       f"{len(self.trade_data)} trades")
                       
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            
    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily performance report"""
        
        logger.info("Generating daily performance report...")
        
        try:
            today = datetime.now().date()
            
            # Basic performance metrics
            basic_metrics = await self._calculate_basic_metrics()
            
            # Risk metrics
            risk_metrics = await self._calculate_risk_metrics()
            
            # Trade analysis
            trade_analysis = await self._analyze_recent_trades()
            
            # Attribution analysis
            attribution = await self._calculate_attribution()
            
            # Market comparison
            market_comparison = await self._compare_to_market()
            
            # Performance trends
            trends = await self._analyze_performance_trends()
            
            # Generate insights
            insights = await self._generate_insights(basic_metrics, risk_metrics, trade_analysis)
            
            report = {
                'report_date': today,
                'report_type': 'DAILY',
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'trade_analysis': trade_analysis,
                'attribution': attribution,
                'market_comparison': market_comparison,
                'trends': trends,
                'insights': insights,
                'charts': await self._generate_charts()
            }
            
            # Store report
            await self._store_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return {'error': str(e)}
            
    async def _calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        
        try:
            if self.portfolio_data.empty:
                return {}
                
            # Latest values
            latest = self.portfolio_data.iloc[-1] if len(self.portfolio_data) > 0 else None
            if latest is None:
                return {}
                
            # Calculate returns
            portfolio_values = self.portfolio_data['total_value'].values
            
            if len(portfolio_values) < 2:
                return {'total_value': latest['total_value']}
                
            # Total return
            initial_value = portfolio_values[0]
            current_value = portfolio_values[-1]
            total_return = (current_value / initial_value - 1) * 100
            
            # Calculate daily returns
            daily_returns = self.portfolio_data['total_value'].pct_change().dropna()
            
            # Performance metrics
            metrics = {
                'total_value': current_value,
                'total_return': total_return,
                'daily_return': daily_returns.iloc[-1] * 100 if len(daily_returns) > 0 else 0,
                'weekly_return': ((current_value / portfolio_values[-min(7, len(portfolio_values))]) - 1) * 100,
                'monthly_return': ((current_value / portfolio_values[-min(30, len(portfolio_values))]) - 1) * 100,
                'ytd_return': total_return,  # Simplified
                'volatility': daily_returns.std() * np.sqrt(252) * 100,
                'avg_daily_return': daily_returns.mean() * 100,
                'best_day': daily_returns.max() * 100,
                'worst_day': daily_returns.min() * 100,
                'positive_days': (daily_returns > 0).sum(),
                'total_days': len(daily_returns),
                'win_rate_daily': (daily_returns > 0).mean() * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {}
            
    async def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        
        try:
            if self.portfolio_data.empty:
                return {}
                
            # Calculate daily returns
            daily_returns = self.portfolio_data['total_value'].pct_change().dropna()
            
            if len(daily_returns) < 30:  # Need minimum data
                return {}
                
            # Risk-free rate (assume 6% annually for Indian markets)
            risk_free_rate = 0.06 / 252  # Daily risk-free rate
            
            # Excess returns
            excess_returns = daily_returns - risk_free_rate
            
            # Sharpe Ratio
            sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            # Sortino Ratio
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino_ratio = (excess_returns.mean() / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Current Drawdown
            current_drawdown = drawdown.iloc[-1] * 100
            
            # Calmar Ratio
            annualized_return = (cumulative_returns.iloc[-1] ** (252 / len(daily_returns))) - 1
            calmar_ratio = annualized_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(daily_returns, 5) * 100
            var_99 = np.percentile(daily_returns, 1) * 100
            
            # Expected Shortfall (Conditional VaR)
            es_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
            
            # Beta (if benchmark data available)
            beta = await self._calculate_beta(daily_returns)
            
            # Alpha
            alpha = await self._calculate_alpha(daily_returns, beta)
            
            # Information Ratio
            information_ratio = await self._calculate_information_ratio(daily_returns)
            
            # Treynor Ratio
            treynor_ratio = annualized_return / beta if beta != 0 else 0
            
            metrics = {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': es_95,
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'treynor_ratio': treynor_ratio,
                'tracking_error': daily_returns.std() * np.sqrt(252) * 100,
                'ulcer_index': self._calculate_ulcer_index(cumulative_returns),
                'pain_index': abs(drawdown.mean()) * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
            
    def _calculate_ulcer_index(self, cumulative_returns: pd.Series) -> float:
        """Calculate Ulcer Index"""
        
        try:
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            ulcer_index = np.sqrt((drawdown ** 2).mean()) * 100
            return ulcer_index
            
        except Exception:
            return 0
            
    async def _calculate_beta(self, portfolio_returns: pd.Series) -> float:
        """Calculate portfolio beta relative to benchmark"""
        
        try:
            if self.benchmark_data.empty:
                return 1.0
                
            # Calculate benchmark returns
            benchmark_returns = self.benchmark_data['benchmark_price'].pct_change().dropna()
            
            # Align dates
            portfolio_df = pd.DataFrame({'portfolio': portfolio_returns})
            benchmark_df = pd.DataFrame({'benchmark': benchmark_returns})
            
            # Merge on index (dates)
            combined = portfolio_df.join(benchmark_df, how='inner')
            
            if len(combined) < 30:
                return 1.0
                
            # Calculate beta using covariance
            covariance = combined['portfolio'].cov(combined['benchmark'])
            benchmark_variance = combined['benchmark'].var()
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0
            
    async def _calculate_alpha(self, portfolio_returns: pd.Series, beta: float) -> float:
        """Calculate Jensen's alpha"""
        
        try:
            if self.benchmark_data.empty:
                return 0
                
            # Calculate benchmark returns
            benchmark_returns = self.benchmark_data['benchmark_price'].pct_change().dropna()
            
            # Risk-free rate
            risk_free_rate = 0.06 / 252
            
            # Portfolio excess return
            portfolio_excess = portfolio_returns.mean() - risk_free_rate
            
            # Benchmark excess return
            benchmark_excess = benchmark_returns.mean() - risk_free_rate
            
            # Alpha = Portfolio excess return - Beta * Benchmark excess return
            alpha = portfolio_excess - (beta * benchmark_excess)
            
            # Annualize
            alpha_annualized = alpha * 252 * 100
            
            return alpha_annualized
            
        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return 0
            
    async def _calculate_information_ratio(self, portfolio_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        
        try:
            if self.benchmark_data.empty:
                return 0
                
            # Calculate benchmark returns
            benchmark_returns = self.benchmark_data['benchmark_price'].pct_change().dropna()
            
            # Align returns
            portfolio_df = pd.DataFrame({'portfolio': portfolio_returns})
            benchmark_df = pd.DataFrame({'benchmark': benchmark_returns})
            combined = portfolio_df.join(benchmark_df, how='inner')
            
            if len(combined) < 30:
                return 0
                
            # Active returns
            active_returns = combined['portfolio'] - combined['benchmark']
            
            # Information ratio
            info_ratio = (active_returns.mean() / active_returns.std()) * np.sqrt(252) if active_returns.std() > 0 else 0
            
            return info_ratio
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0
            
    async def _analyze_recent_trades(self, days: int = 30) -> Dict[str, Any]:
        """Analyze recent trading performance"""
        
        try:
            if self.trade_data.empty:
                return {}
                
            # Filter recent trades
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = self.trade_data[
                pd.to_datetime(self.trade_data['entry_date']) >= cutoff_date
            ].copy()
            
            if recent_trades.empty:
                return {'message': 'No recent trades'}
                
            # Completed trades
            completed_trades = recent_trades[recent_trades['status'] == 'CLOSED'].copy()
            
            if completed_trades.empty:
                return {
                    'total_trades': len(recent_trades),
                    'open_trades': len(recent_trades),
                    'completed_trades': 0
                }
                
            # Basic trade metrics
            winning_trades = completed_trades[completed_trades['pnl'] > 0]
            losing_trades = completed_trades[completed_trades['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(completed_trades) * 100 if len(completed_trades) > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Hold time analysis
            avg_hold_time = completed_trades['hold_days'].mean() if 'hold_days' in completed_trades.columns else 0
            
            # Best and worst trades
            best_trade = completed_trades.loc[completed_trades['pnl'].idxmax()] if len(completed_trades) > 0 else None
            worst_trade = completed_trades.loc[completed_trades['pnl'].idxmin()] if len(completed_trades) > 0 else None
            
            # Strategy performance
            strategy_performance = {}
            if 'strategy' in completed_trades.columns:
                for strategy in completed_trades['strategy'].unique():
                    strategy_trades = completed_trades[completed_trades['strategy'] == strategy]
                    strategy_performance[strategy] = {
                        'trades': len(strategy_trades),
                        'total_pnl': strategy_trades['pnl'].sum(),
                        'win_rate': (strategy_trades['pnl'] > 0).mean() * 100,
                        'avg_pnl': strategy_trades['pnl'].mean()
                    }
                    
            # Symbol performance
            symbol_performance = {}
            for symbol in completed_trades['symbol'].unique():
                symbol_trades = completed_trades[completed_trades['symbol'] == symbol]
                symbol_performance[symbol] = {
                    'trades': len(symbol_trades),
                    'total_pnl': symbol_trades['pnl'].sum(),
                    'win_rate': (symbol_trades['pnl'] > 0).mean() * 100,
                    'avg_pnl': symbol_trades['pnl'].mean()
                }
                
            analysis = {
                'total_trades': len(recent_trades),
                'completed_trades': len(completed_trades),
                'open_trades': len(recent_trades) - len(completed_trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_hold_time': avg_hold_time,
                'total_pnl': completed_trades['pnl'].sum(),
                'largest_win': completed_trades['pnl'].max(),
                'largest_loss': completed_trades['pnl'].min(),
                'best_trade': {
                    'symbol': best_trade['symbol'],
                    'pnl': best_trade['pnl'],
                    'pnl_pct': best_trade.get('pnl_pct', 0)
                } if best_trade is not None else None,
                'worst_trade': {
                    'symbol': worst_trade['symbol'],
                    'pnl': worst_trade['pnl'],
                    'pnl_pct': worst_trade.get('pnl_pct', 0)
                } if worst_trade is not None else None,
                'strategy_performance': strategy_performance,
                'top_symbols': dict(sorted(symbol_performance.items(), key=lambda x: x[1]['total_pnl'], reverse=True)[:5])
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing recent trades: {e}")
            return {'error': str(e)}
            
    async def _calculate_attribution(self) -> Dict[str, Any]:
        """Calculate performance attribution"""
        
        try:
            if self.trade_data.empty:
                return {}
                
            # Recent completed trades
            recent_completed = self.trade_data[
                (self.trade_data['status'] == 'CLOSED') &
                (pd.to_datetime(self.trade_data['exit_date']) >= datetime.now() - timedelta(days=30))
            ].copy()
            
            if recent_completed.empty:
                return {}
                
            total_pnl = recent_completed['pnl'].sum()
            
            attribution = {}
            
            # Attribution by strategy
            if 'strategy' in recent_completed.columns:
                strategy_attribution = recent_completed.groupby('strategy')['pnl'].sum()
                for strategy, pnl in strategy_attribution.items():
                    attribution[f'strategy_{strategy}'] = {
                        'contribution': pnl / total_pnl * 100 if total_pnl != 0 else 0,
                        'absolute_pnl': pnl
                    }
                    
            # Attribution by symbol
            symbol_attribution = recent_completed.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
            top_contributors = symbol_attribution.head(5)
            bottom_contributors = symbol_attribution.tail(5)
            
            attribution['top_contributors'] = {
                symbol: {
                    'contribution': pnl / total_pnl * 100 if total_pnl != 0 else 0,
                    'absolute_pnl': pnl
                }
                for symbol, pnl in top_contributors.items()
            }
            
            attribution['bottom_contributors'] = {
                symbol: {
                    'contribution': pnl / total_pnl * 100 if total_pnl != 0 else 0,
                    'absolute_pnl': pnl
                }
                for symbol, pnl in bottom_contributors.items()
            }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error calculating attribution: {e}")
            return {}
            
    async def _compare_to_market(self) -> Dict[str, Any]:
        """Compare performance to market benchmark"""
        
        try:
            if self.portfolio_data.empty or self.benchmark_data.empty:
                return {}
                
            # Align data
            portfolio_df = self.portfolio_data[['date', 'total_value']].set_index('date')
            benchmark_df = self.benchmark_data.set_index('date')
            
            combined = portfolio_df.join(benchmark_df, how='inner')
            
            if len(combined) < 2:
                return {}
                
            # Calculate returns
            portfolio_returns = combined['total_value'].pct_change().dropna()
            benchmark_returns = combined['benchmark_price'].pct_change().dropna()
            
            # Performance comparison
            portfolio_total_return = (combined['total_value'].iloc[-1] / combined['total_value'].iloc[0] - 1) * 100
            benchmark_total_return = (combined['benchmark_price'].iloc[-1] / combined['benchmark_price'].iloc[0] - 1) * 100
            
            outperformance = portfolio_total_return - benchmark_total_return
            
            # Correlation
            correlation = portfolio_returns.corr(benchmark_returns)
            
            # Tracking error
            tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252) * 100
            
            # Up/Down capture ratios
            up_periods = benchmark_returns > 0
            down_periods = benchmark_returns < 0
            
            up_capture = (portfolio_returns[up_periods].mean() / benchmark_returns[up_periods].mean()) * 100 if len(benchmark_returns[up_periods]) > 0 else 0
            down_capture = (portfolio_returns[down_periods].mean() / benchmark_returns[down_periods].mean()) * 100 if len(benchmark_returns[down_periods]) > 0 else 0
            
            comparison = {
                'portfolio_return': portfolio_total_return,
                'benchmark_return': benchmark_total_return,
                'outperformance': outperformance,
                'correlation': correlation,
                'tracking_error': tracking_error,
                'up_capture_ratio': up_capture,
                'down_capture_ratio': down_capture,
                'information_ratio': (outperformance / 100) / (tracking_error / 100) if tracking_error > 0 else 0
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing to market: {e}")
            return {}
            
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends and patterns"""
        
        try:
            if self.portfolio_data.empty:
                return {}
                
            # Rolling performance
            portfolio_values = self.portfolio_data['total_value']
            
            # 7-day rolling return
            rolling_7d = portfolio_values.pct_change(7).tail(30) * 100
            
            # 30-day rolling return
            rolling_30d = portfolio_values.pct_change(30).tail(60) * 100
            
            # Volatility trends
            daily_returns = portfolio_values.pct_change()
            rolling_vol_30d = daily_returns.rolling(30).std() * np.sqrt(252) * 100
            
            # Sharpe ratio trend
            rolling_sharpe = daily_returns.rolling(30).mean() / daily_returns.rolling(30).std() * np.sqrt(252)
            
            # Trend analysis
            recent_trend = 'NEUTRAL'
            if len(rolling_7d) > 0:
                avg_7d_return = rolling_7d.tail(7).mean()
                if avg_7d_return > 1:
                    recent_trend = 'POSITIVE'
                elif avg_7d_return < -1:
                    recent_trend = 'NEGATIVE'
                    
            trends = {
                'recent_trend': recent_trend,
                'avg_7d_return': rolling_7d.tail(7).mean() if len(rolling_7d) > 0 else 0,
                'avg_30d_return': rolling_30d.tail(30).mean() if len(rolling_30d) > 0 else 0,
                'current_volatility': rolling_vol_30d.iloc[-1] if len(rolling_vol_30d) > 0 else 0,
                'volatility_trend': 'INCREASING' if len(rolling_vol_30d) > 1 and rolling_vol_30d.iloc[-1] > rolling_vol_30d.iloc[-2] else 'DECREASING',
                'current_sharpe': rolling_sharpe.iloc[-1] if len(rolling_sharpe) > 0 else 0,
                'improving_sharpe': len(rolling_sharpe) > 1 and rolling_sharpe.iloc[-1] > rolling_sharpe.iloc[-2]
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {}
            
    async def _generate_insights(self, basic_metrics: Dict, risk_metrics: Dict, trade_analysis: Dict) -> List[str]:
        """Generate actionable insights from performance data"""
        
        insights = []
        
        try:
            # Performance insights
            total_return = basic_metrics.get('total_return', 0)
            if total_return > 15:
                insights.append("🎯 Excellent performance with strong returns above 15%")
            elif total_return > 10:
                insights.append("✅ Good performance with double-digit returns")
            elif total_return < 0:
                insights.append("⚠️ Portfolio is in negative territory - review strategy")
                
            # Risk insights
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 1.5:
                insights.append("🏆 Excellent risk-adjusted returns (Sharpe > 1.5)")
            elif sharpe_ratio < 0.5:
                insights.append("📉 Poor risk-adjusted returns - consider reducing risk")
                
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            if abs(max_drawdown) > 15:
                insights.append("🚨 High maximum drawdown - strengthen risk management")
            elif abs(max_drawdown) < 5:
                insights.append("🛡️ Low drawdown indicates good risk control")
                
            # Trading insights
            win_rate = trade_analysis.get('win_rate', 0)
            if win_rate > 70:
                insights.append("🎯 High win rate indicates good signal quality")
            elif win_rate < 40:
                insights.append("⚠️ Low win rate - review entry criteria")
                
            profit_factor = trade_analysis.get('profit_factor', 0)
            if profit_factor > 2:
                insights.append("💰 Strong profit factor shows good trade management")
            elif profit_factor < 1.2:
                insights.append("📊 Low profit factor - optimize exit strategy")
                
            # Volatility insights
            volatility = basic_metrics.get('volatility', 0)
            if volatility > 25:
                insights.append("⚡ High volatility - consider position sizing adjustments")
            elif volatility < 10:
                insights.append("😴 Low volatility may indicate insufficient risk-taking")
                
            # Recent performance
            daily_return = basic_metrics.get('daily_return', 0)
            if abs(daily_return) > 3:
                insights.append("📈 High daily volatility - monitor closely")
                
            if not insights:
                insights.append("📊 Performance metrics are within normal ranges")
                
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("❌ Error generating insights")
            
        return insights
        
    async def _generate_charts(self) -> Dict[str, str]:
        """Generate performance charts as base64 encoded images"""
        
        try:
            charts = {}
            
            if self.portfolio_data.empty:
                return charts
                
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Portfolio value chart
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = pd.to_datetime(self.portfolio_data['date'])
            values = self.portfolio_data['total_value']
            
            ax.plot(dates, values, linewidth=2, color='#2E86AB')
            ax.fill_between(dates, values, alpha=0.3, color='#2E86AB')
            ax.set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value (₹)')
            ax.grid(True, alpha=0.3)
            
            # Format y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            charts['portfolio_value'] = chart_base64
            plt.close()
            
            # 2. Daily returns distribution
            daily_returns = self.portfolio_data['total_value'].pct_change().dropna() * 100
            
            if len(daily_returns) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(daily_returns, bins=30, alpha=0.7, color='#F18F01', edgecolor='black')
                ax.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_returns.mean():.2f}%')
                ax.set_title('Daily Returns Distribution', fontsize=16, fontweight='bold')
                ax.set_xlabel('Daily Return (%)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                chart_base64 = base64.b64encode(buffer.getvalue()).decode()
                charts['returns_distribution'] = chart_base64
                plt.close()
                
            # 3. Drawdown chart
            cumulative_returns = (1 + daily_returns / 100).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.fill_between(dates[1:], drawdown, alpha=0.7, color='red')
            ax.plot(dates[1:], drawdown, linewidth=1, color='darkred')
            ax.set_title('Portfolio Drawdown', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            charts['drawdown'] = chart_base64
            plt.close()
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return {}
            
    async def _store_report(self, report: Dict):
        """Store performance report in database"""
        
        try:
            from uuid import uuid4
            report_id = str(uuid4())
            
            basic_metrics = report.get('basic_metrics', {})
            risk_metrics = report.get('risk_metrics', {})
            
            insert_query = """
            INSERT INTO performance_reports 
            (report_id, report_type, period_start, period_end, total_return,
             sharpe_ratio, max_drawdown, win_rate, profit_factor, calmar_ratio,
             sortino_ratio, alpha, beta, information_ratio, treynor_ratio,
             total_trades, report_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                report_id,
                report['report_type'],
                report['report_date'],
                report['report_date'],
                basic_metrics.get('total_return'),
                risk_metrics.get('sharpe_ratio'),
                risk_metrics.get('max_drawdown'),
                report.get('trade_analysis', {}).get('win_rate'),
                report.get('trade_analysis', {}).get('profit_factor'),
                risk_metrics.get('calmar_ratio'),
                risk_metrics.get('sortino_ratio'),
                risk_metrics.get('alpha'),
                risk_metrics.get('beta'),
                risk_metrics.get('information_ratio'),
                risk_metrics.get('treynor_ratio'),
                report.get('trade_analysis', {}).get('total_trades'),
                str(report)
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing report: {e}")
            
    async def generate_monthly_report(self) -> Dict[str, Any]:
        """Generate comprehensive monthly report"""
        
        logger.info("Generating monthly performance report...")
        
        try:
            # Load month's data
            start_date = datetime.now().replace(day=1)
            await self._load_performance_data(days_back=60)
            
            # Filter to current month
            month_data = self.portfolio_data[
                pd.to_datetime(self.portfolio_data['date']) >= start_date
            ]
            
            # Calculate monthly metrics
            monthly_metrics = await self._calculate_monthly_metrics(month_data)
            
            # Detailed trade analysis
            trade_analysis = await self._analyze_monthly_trades()
            
            # Strategy analysis
            strategy_analysis = await self._analyze_strategy_performance()
            
            # Risk analysis
            risk_analysis = await self._analyze_monthly_risk()
            
            # Generate recommendations
            recommendations = await self._generate_monthly_recommendations(monthly_metrics, trade_analysis)
            
            report = {
                'report_type': 'MONTHLY',
                'period': start_date.strftime('%B %Y'),
                'monthly_metrics': monthly_metrics,
                'trade_analysis': trade_analysis,
                'strategy_analysis': strategy_analysis,
                'risk_analysis': risk_analysis,
                'recommendations': recommendations,
                'charts': await self._generate_monthly_charts()
            }
            
            await self._store_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating monthly report: {e}")
            return {'error': str(e)}
            
    async def _calculate_monthly_metrics(self, month_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate monthly performance metrics"""
        
        try:
            if month_data.empty or len(month_data) < 2:
                return {}
                
            start_value = month_data['total_value'].iloc[0]
            end_value = month_data['total_value'].iloc[-1]
            
            monthly_return = (end_value / start_value - 1) * 100
            
            # Daily returns for the month
            daily_returns = month_data['total_value'].pct_change().dropna()
            
            metrics = {
                'monthly_return': monthly_return,
                'monthly_volatility': daily_returns.std() * np.sqrt(21) * 100,  # Monthly volatility
                'best_day': daily_returns.max() * 100,
                'worst_day': daily_returns.min() * 100,
                'positive_days': (daily_returns > 0).sum(),
                'total_trading_days': len(daily_returns),
                'avg_daily_return': daily_returns.mean() * 100,
                'start_value': start_value,
                'end_value': end_value,
                'high_water_mark': month_data['total_value'].max(),
                'low_water_mark': month_data['total_value'].min()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating monthly metrics: {e}")
            return {}
            
    async def _analyze_monthly_trades(self) -> Dict[str, Any]:
        """Analyze trades for the month"""
        
        try:
            start_of_month = datetime.now().replace(day=1)
            
            month_trades = self.trade_data[
                pd.to_datetime(self.trade_data['entry_date']) >= start_of_month
            ]
            
            if month_trades.empty:
                return {'message': 'No trades this month'}
                
            # Analyze by week
            month_trades['week'] = pd.to_datetime(month_trades['entry_date']).dt.isocalendar().week
            
            weekly_analysis = {}
            for week in month_trades['week'].unique():
                week_trades = month_trades[month_trades['week'] == week]
                closed_trades = week_trades[week_trades['status'] == 'CLOSED']
                
                weekly_analysis[f'week_{week}'] = {
                    'total_trades': len(week_trades),
                    'completed_trades': len(closed_trades),
                    'total_pnl': closed_trades['pnl'].sum() if len(closed_trades) > 0 else 0,
                    'win_rate': (closed_trades['pnl'] > 0).mean() * 100 if len(closed_trades) > 0 else 0
                }
                
            return {
                'total_month_trades': len(month_trades),
                'weekly_breakdown': weekly_analysis,
                'most_traded_symbols': month_trades['symbol'].value_counts().head(5).to_dict(),
                'avg_trades_per_day': len(month_trades) / 30  # Approximate
            }
            
        except Exception as e:
            logger.error(f"Error analyzing monthly trades: {e}")
            return {}
            
    async def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance by strategy"""
        
        try:
            if 'strategy' not in self.trade_data.columns:
                return {}
                
            completed_trades = self.trade_data[self.trade_data['status'] == 'CLOSED']
            
            strategy_performance = {}
            
            for strategy in completed_trades['strategy'].unique():
                strategy_trades = completed_trades[completed_trades['strategy'] == strategy]
                
                if len(strategy_trades) > 0:
                    strategy_performance[strategy] = {
                        'total_trades': len(strategy_trades),
                        'total_pnl': strategy_trades['pnl'].sum(),
                        'win_rate': (strategy_trades['pnl'] > 0).mean() * 100,
                        'avg_pnl': strategy_trades['pnl'].mean(),
                        'best_trade': strategy_trades['pnl'].max(),
                        'worst_trade': strategy_trades['pnl'].min(),
                        'avg_hold_time': strategy_trades['hold_days'].mean() if 'hold_days' in strategy_trades.columns else 0
                    }
                    
            return strategy_performance
            
        except Exception as e:
            logger.error(f"Error analyzing strategy performance: {e}")
            return {}
            
    async def _analyze_monthly_risk(self) -> Dict[str, Any]:
        """Analyze risk metrics for the month"""
        
        try:
            start_of_month = datetime.now().replace(day=1)
            
            month_portfolio = self.portfolio_data[
                pd.to_datetime(self.portfolio_data['date']) >= start_of_month
            ]
            
            if month_portfolio.empty:
                return {}
                
            daily_returns = month_portfolio['total_value'].pct_change().dropna()
            
            if len(daily_returns) == 0:
                return {}
                
            # Risk metrics
            var_95 = np.percentile(daily_returns, 5) * 100
            var_99 = np.percentile(daily_returns, 1) * 100
            
            # Maximum daily loss
            max_daily_loss = daily_returns.min() * 100
            
            # Days with losses > 1%
            large_loss_days = (daily_returns < -0.01).sum()
            
            risk_analysis = {
                'var_95_daily': var_95,
                'var_99_daily': var_99,
                'max_daily_loss': max_daily_loss,
                'large_loss_days': large_loss_days,
                'volatility': daily_returns.std() * np.sqrt(252) * 100,
                'downside_volatility': daily_returns[daily_returns < 0].std() * np.sqrt(252) * 100,
                'risk_assessment': 'LOW' if abs(var_95) < 1 else 'MODERATE' if abs(var_95) < 2 else 'HIGH'
            }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing monthly risk: {e}")
            return {}
            
    async def _generate_monthly_recommendations(self, monthly_metrics: Dict, trade_analysis: Dict) -> List[str]:
        """Generate monthly recommendations"""
        
        recommendations = []
        
        try:
            monthly_return = monthly_metrics.get('monthly_return', 0)
            monthly_volatility = monthly_metrics.get('monthly_volatility', 0)
            
            # Performance recommendations
            if monthly_return > 5:
                recommendations.append("✅ Strong monthly performance - consider partial profit booking")
            elif monthly_return < -5:
                recommendations.append("⚠️ Weak monthly performance - review strategy and risk management")
                
            # Volatility recommendations
            if monthly_volatility > 15:
                recommendations.append("📊 High monthly volatility - consider reducing position sizes")
            elif monthly_volatility < 5:
                recommendations.append("😴 Low volatility - may indicate under-leveraged positions")
                
            # Trading recommendations
            total_trades = trade_analysis.get('total_month_trades', 0)
            if total_trades > 50:
                recommendations.append("⚡ High trading frequency - ensure transaction costs are justified")
            elif total_trades < 10:
                recommendations.append("🐌 Low trading activity - may be missing opportunities")
                
            # General recommendations
            recommendations.append("📈 Review top performing strategies for scaling opportunities")
            recommendations.append("🔍 Analyze worst performing trades for improvement areas")
            recommendations.append("🎯 Monitor market regime changes for strategy adjustments")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            
        return recommendations
        
    async def _generate_monthly_charts(self) -> Dict[str, str]:
        """Generate monthly performance charts"""
        
        # Similar to daily charts but with monthly focus
        return await self._generate_charts()
        
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics module summary"""
        
        return {
            'portfolio_data_points': len(self.portfolio_data),
            'trade_data_points': len(self.trade_data),
            'benchmark_data_points': len(self.benchmark_data),
            'metrics_cached': len(self.metrics_cache),
            'analysis_cached': len(self.analysis_cache),
            'last_analysis': datetime.now().isoformat()
        }
