"""
Configuration settings for the AI-Gated Trading System
"""

import os
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Config:
    """System configuration parameters"""
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost/trading_db")
    PGDATABASE: str = os.getenv("PGDATABASE", "trading_db")
    PGHOST: str = os.getenv("PGHOST", "localhost")
    PGPASSWORD: str = os.getenv("PGPASSWORD", "")
    PGPORT: str = os.getenv("PGPORT", "5432")
    PGUSER: str = os.getenv("PGUSER", "postgres")
    
    # Trading Parameters
    MAX_DAILY_TRADES: int = 3
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio per position
    MAX_PORTFOLIO_RISK: float = 0.02  # 2% portfolio risk per trade
    MIN_CONFIDENCE_THRESHOLD: float = 0.7  # Minimum AI confidence for trade approval
    
    # Market Data
    SCAN_INTERVAL: int = 30  # seconds between scans
    NSE_STOCKS_COUNT: int = 1700
    
    # Technical Indicators Parameters
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70
    RSI_OVERSOLD: float = 30
    
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER: float = 2.0  # For stop loss calculation
    
    # ML Model Parameters
    ENSEMBLE_MODELS: List[str] = None
    LOOKBACK_PERIOD: int = 252  # Trading days for training
    RETRAIN_FREQUENCY: int = 7  # Days between model retraining
    
    # Market Regime Detection
    REGIME_LOOKBACK: int = 60  # Days for regime detection
    REGIME_STATES: int = 3  # Bull, Bear, Sideways
    
    # Risk Management
    STOP_LOSS_METHOD: str = "ATR"  # ATR, PERCENTAGE, FIXED
    TAKE_PROFIT_RATIO: float = 2.0  # Risk:Reward ratio
    TRAILING_STOP: bool = True
    
    # Alerts
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Performance Tracking
    BENCHMARK_SYMBOL: str = "NIFTY50"
    PERFORMANCE_METRICS: List[str] = None
    
    # API Keys and External Services
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    
    def __post_init__(self):
        """Initialize default values for complex types"""
        if self.ENSEMBLE_MODELS is None:
            self.ENSEMBLE_MODELS = [
                "xgboost",
                "random_forest", 
                "neural_network",
                "gradient_boosting"
            ]
            
        if self.PERFORMANCE_METRICS is None:
            self.PERFORMANCE_METRICS = [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "calmar_ratio",
                "sortino_ratio"
            ]

# Market timings for NSE
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"
PRE_MARKET_OPEN = "09:00"
POST_MARKET_CLOSE = "16:00"

# Stock categories for screening
STOCK_CATEGORIES = {
    "LARGE_CAP": {"min_market_cap": 20000, "max_stocks": 100},
    "MID_CAP": {"min_market_cap": 5000, "max_stocks": 250},
    "SMALL_CAP": {"min_market_cap": 500, "max_stocks": 500}
}

# Feature engineering parameters
FEATURE_CONFIG = {
    "price_features": ["open", "high", "low", "close", "volume"],
    "technical_indicators": ["rsi", "macd", "bb", "atr", "adx", "stoch"],
    "market_microstructure": ["bid_ask_spread", "order_imbalance", "vwap"],
    "relative_strength": ["vs_nifty", "vs_sector"],
    "regime_features": ["volatility_regime", "trend_regime", "volume_regime"]
}

# Model hyperparameters
ML_CONFIG = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5
    },
    "neural_network": {
        "hidden_layers": [64, 32, 16],
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32
    }
}

# Reinforcement learning parameters
RL_CONFIG = {
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10
}
