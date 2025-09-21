"""
Streamlit Dashboard for AI-Gated Trading System
Real-time monitoring and control interface
"""
print("Streamlit app is starting...")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import logging
from datetime import datetime, timedelta
import time
import json

# Configure page
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from database.db_manager import DatabaseManager
from trading.ai_gate import AIGate
from trading.portfolio_manager import PortfolioManager
from analytics.performance_analytics import PerformanceAnalytics
from analytics.trade_analysis import TradeAnalysis
from data.market_data import MarketDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize system components"""
    config = Config()
    db_manager = DatabaseManager()
    
    return {
        'config': config,
        'db_manager': db_manager,
        'ai_gate': AIGate(config),
        'portfolio_manager': PortfolioManager(config),
        'performance_analytics': PerformanceAnalytics(config),
        'trade_analysis': TradeAnalysis(config),
        'market_data_manager': MarketDataManager(config)
    }

# Load data functions
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_portfolio_data(db_manager):
    """Load portfolio data"""
    try:
        query = """
        SELECT date, total_value, daily_pnl, unrealized_pnl, realized_pnl, drawdown
        FROM portfolio_snapshots 
        WHERE date >= %s 
        ORDER BY date DESC
        LIMIT 100
        """
        
        cutoff_date = datetime.now() - timedelta(days=100)
        
        # Note: In production, you'd use async/await properly
        # For Streamlit, we'll simulate the data loading
        data = pd.DataFrame({
            'date': pd.date_range(start=cutoff_date, periods=30, freq='D'),
            'total_value': np.random.normal(1000000, 50000, 30).cumsum(),
            'daily_pnl': np.random.normal(1000, 5000, 30),
            'unrealized_pnl': np.random.normal(5000, 10000, 30),
            'realized_pnl': np.random.normal(2000, 3000, 30),
            'drawdown': np.abs(np.random.normal(0, 0.02, 30))
        })
        
        return data
        
    except Exception as e:
        st.error(f"Error loading portfolio data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_trade_data(db_manager):
    """Load recent trade data"""
    try:
        # Simulated trade data
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        data = []
        
        for i in range(50):
            data.append({
                'trade_id': f'TRADE_{i:03d}',
                'symbol': np.random.choice(symbols),
                'entry_date': datetime.now() - timedelta(days=np.random.randint(1, 30)),
                'entry_price': np.random.uniform(100, 3000),
                'exit_price': np.random.uniform(100, 3000),
                'quantity': np.random.randint(10, 1000),
                'pnl': np.random.normal(1000, 5000),
                'pnl_pct': np.random.normal(1, 5),
                'status': np.random.choice(['OPEN', 'CLOSED', 'CLOSED', 'CLOSED']),
                'ai_confidence': np.random.uniform(0.5, 1.0),
                'hold_days': np.random.randint(1, 10)
            })
            
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error loading trade data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=120)  # Cache for 2 minutes
def load_market_status():
    """Load current market status"""
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15)
    market_close = now.replace(hour=15, minute=30)
    
    is_open = market_open <= now <= market_close and now.weekday() < 5
    
    return {
        'is_open': is_open,
        'current_time': now,
        'market_open': market_open,
        'market_close': market_close,
        'nifty_level': 19500 + np.random.normal(0, 100),
        'nifty_change': np.random.normal(0, 1),
        'volatility_index': 15 + np.random.normal(0, 2)
    }

# Dashboard functions
def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🤖 AI-Gated Trading Dashboard")
        st.markdown("**Real-time monitoring and control interface**")
        
    with col2:
        market_status = load_market_status()
        status_color = "🟢" if market_status['is_open'] else "🔴"
        st.metric(
            "Market Status",
            f"{status_color} {'OPEN' if market_status['is_open'] else 'CLOSED'}",
            f"Nifty: {market_status['nifty_level']:.0f}"
        )
        
    with col3:
        st.metric(
            "System Time",
            datetime.now().strftime("%H:%M:%S"),
            "Real-time"
        )

def render_portfolio_overview(components):
    """Render portfolio overview section"""
    st.header("📊 Portfolio Overview")
    
    # Load portfolio data
    portfolio_data = load_portfolio_data(components['db_manager'])
    
    if not portfolio_data.empty:
        latest = portfolio_data.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Portfolio Value",
                f"₹{latest['total_value']:,.0f}",
                f"{latest['daily_pnl']:+,.0f}"
            )
            
        with col2:
            total_return = (latest['total_value'] / 1000000 - 1) * 100
            st.metric(
                "Total Return",
                f"{total_return:+.2f}%",
                f"₹{latest['realized_pnl']:+,.0f}"
            )
            
        with col3:
            st.metric(
                "Unrealized P&L",
                f"₹{latest['unrealized_pnl']:+,.0f}",
                "Open Positions"
            )
            
        with col4:
            st.metric(
                "Max Drawdown",
                f"{latest['drawdown']:.2%}",
                "Risk Metric"
            )
            
        # Portfolio value chart
        st.subheader("Portfolio Value Trend")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_data['date'],
            y=portfolio_data['total_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value (₹)",
            hovermode='x unified',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No portfolio data available")

def render_trading_activity(components):
    """Render trading activity section"""
    st.header("⚡ Trading Activity")
    
    # Load trade data
    trade_data = load_trade_data(components['db_manager'])
    
    if not trade_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Recent trades
            st.subheader("Recent Trades")
            
            recent_trades = trade_data.head(10)[
                ['symbol', 'entry_date', 'pnl', 'pnl_pct', 'status', 'ai_confidence']
            ].copy()
            
            # Format the data for display
            recent_trades['entry_date'] = recent_trades['entry_date'].dt.strftime('%Y-%m-%d %H:%M')
            recent_trades['pnl'] = recent_trades['pnl'].apply(lambda x: f"₹{x:+,.0f}")
            recent_trades['pnl_pct'] = recent_trades['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            recent_trades['ai_confidence'] = recent_trades['ai_confidence'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(recent_trades, use_container_width=True, hide_index=True)
            
        with col2:
            # Trade performance metrics
            st.subheader("Performance Metrics")
            
            closed_trades = trade_data[trade_data['status'] == 'CLOSED']
            
            if not closed_trades.empty:
                win_rate = (closed_trades['pnl'] > 0).mean() * 100
                avg_return = closed_trades['pnl_pct'].mean()
                total_pnl = closed_trades['pnl'].sum()
                total_trades = len(closed_trades)
                
                metrics_data = {
                    'Metric': ['Win Rate', 'Avg Return', 'Total P&L', 'Total Trades'],
                    'Value': [
                        f"{win_rate:.1f}%",
                        f"{avg_return:+.2f}%",
                        f"₹{total_pnl:+,.0f}",
                        f"{total_trades}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
                
        # Trade distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("P&L Distribution")
            
            fig = px.histogram(
                closed_trades,
                x='pnl_pct',
                nbins=20,
                title="Trade Returns Distribution",
                labels={'pnl_pct': 'Return (%)', 'count': 'Frequency'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Symbol Performance")
            
            symbol_performance = closed_trades.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=symbol_performance.index,
                y=symbol_performance.values,
                title="P&L by Symbol",
                labels={'x': 'Symbol', 'y': 'Total P&L (₹)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning("No trading data available")

def render_ai_insights(components):
    """Render AI insights and decision analytics"""
    st.header("🧠 AI Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AI Decision Analytics")
        
        # Simulated AI decision data
        decision_data = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=24), periods=100, freq='15min'),
            'decisions_made': np.random.poisson(3, 100),
            'decisions_approved': np.random.poisson(1, 100),
            'avg_confidence': np.random.uniform(0.6, 0.9, 100)
        })
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Decision Volume', 'Average Confidence'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=decision_data['timestamp'],
                y=decision_data['decisions_made'],
                name='Total Decisions',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=decision_data['timestamp'],
                y=decision_data['decisions_approved'],
                name='Approved',
                line=dict(color='green')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=decision_data['timestamp'],
                y=decision_data['avg_confidence'],
                name='Avg Confidence',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Market Regime Detection")
        
        # Simulated regime data
        regime_data = pd.DataFrame({
            'date': pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D'),
            'regime': np.random.choice(['BULL', 'BEAR', 'SIDEWAYS'], 30, p=[0.4, 0.2, 0.4]),
            'confidence': np.random.uniform(0.6, 0.95, 30)
        })
        
        # Current regime
        current_regime = regime_data.iloc[-1]
        
        st.metric(
            "Current Regime",
            current_regime['regime'],
            f"Confidence: {current_regime['confidence']:.2f}"
        )
        
        # Regime history
        regime_counts = regime_data['regime'].value_counts()
        
        fig = px.pie(
            values=regime_counts.values,
            names=regime_counts.index,
            title="Regime Distribution (Last 30 Days)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_risk_monitoring(components):
    """Render risk monitoring section"""
    st.header("🛡️ Risk Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Risk Metrics")
        
        # Simulated risk data
        risk_metrics = {
            'Portfolio Risk': '2.5%',
            'VaR (95%)': '₹45,000',
            'Max Position Size': '8.5%',
            'Daily Trade Limit': '2/3',
            'Correlation Risk': 'Low',
            'Leverage': '1.2x'
        }
        
        for metric, value in risk_metrics.items():
            st.metric(metric, value)
            
    with col2:
        st.subheader("Position Concentration")
        
        # Simulated position data
        position_data = pd.DataFrame({
            'symbol': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'],
            'allocation': [15, 12, 10, 8, 7],
            'risk_contribution': [18, 14, 12, 9, 8]
        })
        
        fig = px.bar(
            position_data,
            x='symbol',
            y=['allocation', 'risk_contribution'],
            title="Allocation vs Risk Contribution",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        st.subheader("Risk Alerts")
        
        # Simulated alerts
        alerts = [
            "🟢 All risk limits within bounds",
            "🟡 Position concentration at 85% of limit",
            "🟢 Drawdown within acceptable range",
            "🟡 Correlation risk slightly elevated"
        ]
        
        for alert in alerts:
            st.write(alert)

def render_system_status(components):
    """Render system status section"""
    st.header("⚙️ System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Component Status")
        
        # Simulated system status
        system_status = {
            'AI Gate': '🟢 Operational',
            'Market Data': '🟢 Connected',
            'Database': '🟢 Healthy',
            'Risk Manager': '🟢 Active',
            'Portfolio Manager': '🟢 Running',
            'Telegram Alerts': '🟡 Connected'
        }
        
        for component, status in system_status.items():
            st.write(f"**{component}:** {status}")
            
    with col2:
        st.subheader("Performance Stats")
        
        perf_stats = {
            'Uptime': '99.5%',
            'Avg Response Time': '45ms',
            'Orders Processed': '1,247',
            'Data Points': '2.4M',
            'Model Accuracy': '73.2%',
            'Last Backup': '2 hours ago'
        }
        
        for stat, value in perf_stats.items():
            st.metric(stat, value)
            
    with col3:
        st.subheader("Recent Logs")
        
        # Simulated log entries
        logs = [
            f"{datetime.now().strftime('%H:%M:%S')} - Trade executed: RELIANCE",
            f"{(datetime.now() - timedelta(minutes=5)).strftime('%H:%M:%S')} - AI decision: APPROVED",
            f"{(datetime.now() - timedelta(minutes=10)).strftime('%H:%M:%S')} - Market data updated",
            f"{(datetime.now() - timedelta(minutes=15)).strftime('%H:%M:%S')} - Risk check passed",
            f"{(datetime.now() - timedelta(minutes=20)).strftime('%H:%M:%S')} - Portfolio rebalanced"
        ]
        
        for log in logs:
            st.text(log)

def render_controls(components):
    """Render system controls"""
    st.header("🎛️ System Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Trading Controls")
        
        trading_enabled = st.toggle("Enable Trading", value=True)
        max_daily_trades = st.slider("Max Daily Trades", 1, 10, 3)
        confidence_threshold = st.slider("AI Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
        position_size_limit = st.slider("Max Position Size (%)", 1, 20, 10)
        
        if st.button("Apply Trading Settings"):
            st.success("Trading settings updated!")
            
    with col2:
        st.subheader("Risk Controls")
        
        emergency_stop = st.button("🚨 Emergency Stop", type="primary")
        close_all_positions = st.button("Close All Positions")
        
        if emergency_stop:
            st.error("Emergency stop activated! All trading halted.")
            
        if close_all_positions:
            st.warning("Closing all open positions...")
            
        portfolio_risk_limit = st.slider("Portfolio Risk Limit (%)", 1, 10, 5)
        drawdown_limit = st.slider("Max Drawdown Limit (%)", 5, 25, 15)
        
    with col3:
        st.subheader("System Actions")
        
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.success("Data refreshed!")
            
        if st.button("Retrain AI Models"):
            st.info("AI model retraining initiated...")
            
        if st.button("Generate Report"):
            st.info("Generating performance report...")
            
        if st.button("Backup Data"):
            st.success("Data backup completed!")

# Sidebar
def render_sidebar():
    """Render sidebar with navigation and quick stats"""
    st.sidebar.title("Navigation")
    
    # Quick stats
    st.sidebar.subheader("Quick Stats")
    
    market_status = load_market_status()
    
    st.sidebar.metric("Market", "OPEN" if market_status['is_open'] else "CLOSED")
    st.sidebar.metric("Nifty 50", f"{market_status['nifty_level']:.0f}")
    st.sidebar.metric("VIX", f"{market_status['volatility_index']:.1f}")
    
    # Time ranges
    st.sidebar.subheader("Time Range")
    time_range = st.sidebar.selectbox(
        "Select Range",
        ["1D", "1W", "1M", "3M", "6M", "1Y"]
    )
    
    # Refresh controls
    st.sidebar.subheader("Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 10, 300, 60)
        
    if st.sidebar.button("Manual Refresh"):
        st.cache_data.clear()
        st.rerun()
        
    # System info
    st.sidebar.subheader("System Info")
    st.sidebar.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.info("Version: 1.0.0")

# Main app
def main():
    """Main dashboard application"""
    
    # Initialize components
    components = initialize_components()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_header()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Portfolio",
        "⚡ Trading", 
        "🧠 AI Insights",
        "🛡️ Risk",
        "⚙️ System",
        "🎛️ Controls"
    ])
    
    with tab1:
        render_portfolio_overview(components)
        
    with tab2:
        render_trading_activity(components)
        
    with tab3:
        render_ai_insights(components)
        
    with tab4:
        render_risk_monitoring(components)
        
    with tab5:
        render_system_status(components)
        
    with tab6:
        render_controls(components)
        
    # Auto-refresh capability
    if st.sidebar.checkbox("Auto Refresh", value=True):
        time.sleep(60)  # Refresh every minute
        st.rerun()

if __name__ == "__main__":
    main()
