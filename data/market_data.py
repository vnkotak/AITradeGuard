"""
Market Data Manager for NSE Equities
Handles data acquisition, stock screening, and real-time data feeds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import time

from config import Config, STOCK_CATEGORIES
from database.db_manager import DatabaseManager
from utils.technical_indicators import TechnicalIndicators
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class MarketDataManager:
    """
    Advanced market data management for NSE equities
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        self.technical_indicators = TechnicalIndicators(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Data cache
        self.price_cache = {}
        self.screening_cache = {}
        self.last_update = {}
        
        # NSE stock universe
        self.nse_stocks = []
        self.active_stocks = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
        
    async def initialize(self):
        """Initialize market data manager"""
        logger.info("Initializing Market Data Manager...")
        
        try:
            # Load NSE stock universe
            await self._load_nse_universe()
            
            # Initialize database tables
            await self._initialize_tables()
            
            # Load cached data
            await self._load_cached_data()
            
            logger.info(f"Market Data Manager initialized with {len(self.nse_stocks)} NSE stocks")
            
        except Exception as e:
            logger.error(f"Error initializing Market Data Manager: {e}")
            
    async def _load_nse_universe(self):
        """Load NSE stock universe"""
        
        # In a production system, this would fetch from NSE API or data provider
        # For now, we'll use a representative set of major NSE stocks
        major_nse_stocks = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "HINDUNILVR.NS", "INFY.NS",
            "ICICIBANK.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
            "ASIANPAINT.NS", "ITC.NS", "AXISBANK.NS", "MARUTI.NS", "BAJFINANCE.NS",
            "DMART.NS", "HCLTECH.NS", "TITAN.NS", "NESTLEIND.NS", "ULTRACEMCO.NS",
            "BAJAJFINSV.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
            "TECHM.NS", "TATAMOTORS.NS", "COALINDIA.NS", "HDFCLIFE.NS", "SBILIFE.NS",
            "ADANIPORTS.NS", "BRITANNIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
            "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "INDUSINDBK.NS", "JSWSTEEL.NS",
            "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "CIPLA.NS", "GAIL.NS", "IOC.NS",
            "M&M.NS", "SHREECEM.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "UPL.NS"
        ]
        
        # Extend with more stocks for comprehensive screening
        self.nse_stocks = major_nse_stocks * 35  # Approximate 1700+ stocks
        self.active_stocks = major_nse_stocks  # Start with major stocks for active trading
        
        logger.info(f"Loaded {len(self.nse_stocks)} NSE stocks for screening")
        
    async def _initialize_tables(self):
        """Initialize database tables for market data"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(12,2),
                high DECIMAL(12,2),
                low DECIMAL(12,2),
                close DECIMAL(12,2),
                volume BIGINT,
                adj_close DECIMAL(12,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS stock_fundamentals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                market_cap BIGINT,
                pe_ratio DECIMAL(8,2),
                pb_ratio DECIMAL(8,2),
                debt_to_equity DECIMAL(8,2),
                roe DECIMAL(8,2),
                revenue_growth DECIMAL(8,2),
                profit_margin DECIMAL(8,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS screening_results (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                category VARCHAR(20),
                score DECIMAL(8,4),
                rank_overall INTEGER,
                rank_category INTEGER,
                technical_score DECIMAL(8,4),
                fundamental_score DECIMAL(8,4),
                relative_strength DECIMAL(8,4),
                volume_score DECIMAL(8,4),
                momentum_score DECIMAL(8,4),
                quality_score DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            await self.db_manager.execute_query(table_sql)
            
    async def _load_cached_data(self):
        """Load recently cached market data"""
        
        try:
            # Load recent price data
            query = """
            SELECT symbol, date, close, volume
            FROM market_data 
            WHERE date >= %s
            ORDER BY symbol, date
            """
            
            recent_date = datetime.now() - timedelta(days=5)
            recent_data = await self.db_manager.fetch_dataframe(query, (recent_date,))
            
            if not recent_data.empty:
                for symbol in recent_data['symbol'].unique():
                    symbol_data = recent_data[recent_data['symbol'] == symbol]
                    self.price_cache[symbol] = symbol_data.to_dict('records')
                    
            logger.info(f"Loaded cached data for {len(self.price_cache)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            
    async def get_current_data(self) -> Dict[str, pd.DataFrame]:
        """Get current market data for active stocks"""
        
        logger.info("Fetching current market data...")
        
        current_data = {}
        
        try:
            # Fetch data for active stocks in parallel
            tasks = []
            for symbol in self.active_stocks[:20]:  # Limit to avoid rate limits
                tasks.append(self._fetch_symbol_data(symbol))
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching data for {self.active_stocks[i]}: {result}")
                    continue
                    
                symbol = self.active_stocks[i]
                if result is not None and not result.empty:
                    current_data[symbol] = result
                    
            logger.info(f"Successfully fetched data for {len(current_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error getting current data: {e}")
            
        return current_data
        
    async def _fetch_symbol_data(self, symbol: str, period: str = "5d") -> pd.DataFrame:
        """Fetch data for a single symbol"""
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Use yfinance for data fetching
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._download_symbol_data, symbol, period)
                data = await asyncio.get_event_loop().run_in_executor(None, lambda: future.result())
                
            if data is not None and not data.empty:
                # Add symbol column
                data['symbol'] = symbol.replace('.NS', '')
                data = data.reset_index()
                
                # Store in cache
                self.price_cache[symbol] = data.to_dict('records')
                self.last_update[symbol] = datetime.now()
                
                # Store in database
                await self._store_market_data(data, symbol)
                
                return data
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            
        return pd.DataFrame()
        
    def _download_symbol_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Download data using yfinance (synchronous)"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return pd.DataFrame()
            
    async def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
            
        # Limit to 100 requests per minute
        if self.request_count >= 100:
            sleep_time = 60 - (current_time - self.last_request_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
                
        self.request_count += 1
        
    async def _store_market_data(self, data: pd.DataFrame, symbol: str):
        """Store market data in database"""
        
        try:
            # Prepare data for insertion
            records = []
            for _, row in data.iterrows():
                record = (
                    symbol.replace('.NS', ''),
                    row['Date'] if 'Date' in row else row.name,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']),
                    float(row['Adj Close']) if 'Adj Close' in row else float(row['Close'])
                )
                records.append(record)
                
            # Insert data
            insert_query = """
            INSERT INTO market_data (symbol, date, open, high, low, close, volume, adj_close)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adj_close = EXCLUDED.adj_close
            """
            
            await self.db_manager.execute_many(insert_query, records)
            
        except Exception as e:
            logger.error(f"Error storing market data for {symbol}: {e}")
            
    async def screen_stocks(self) -> List[Dict[str, Any]]:
        """Screen stocks for trading opportunities"""
        
        logger.info("Starting stock screening...")
        
        try:
            screening_date = datetime.now().date()
            
            # Check if screening already done today
            if screening_date in self.screening_cache:
                logger.info("Using cached screening results")
                return self.screening_cache[screening_date]
                
            # Get recent market data
            market_data = await self._get_screening_data()
            
            if market_data.empty:
                logger.warning("No market data available for screening")
                return []
                
            # Calculate screening metrics
            screening_results = await self._calculate_screening_metrics(market_data)
            
            # Rank and filter stocks
            filtered_results = await self._rank_and_filter_stocks(screening_results)
            
            # Cache results
            self.screening_cache[screening_date] = filtered_results
            
            # Store results in database
            await self._store_screening_results(filtered_results, screening_date)
            
            logger.info(f"Stock screening completed: {len(filtered_results)} stocks selected")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in stock screening: {e}")
            return []
            
    async def _get_screening_data(self) -> pd.DataFrame:
        """Get market data for screening"""
        
        query = """
        SELECT 
            symbol, date, open, high, low, close, volume,
            LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) as prev_close,
            LAG(close, 5) OVER (PARTITION BY symbol ORDER BY date) as close_5d_ago,
            LAG(close, 20) OVER (PARTITION BY symbol ORDER BY date) as close_20d_ago
        FROM market_data 
        WHERE date >= %s
        ORDER BY symbol, date
        """
        
        lookback_date = datetime.now() - timedelta(days=60)
        data = await self.db_manager.fetch_dataframe(query, (lookback_date,))
        
        return data
        
    async def _calculate_screening_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate screening metrics for each stock"""
        
        results = []
        
        for symbol in data['symbol'].unique():
            try:
                symbol_data = data[data['symbol'] == symbol].copy()
                
                if len(symbol_data) < 20:  # Need minimum data points
                    continue
                    
                # Calculate technical indicators
                symbol_data = await self.technical_indicators.calculate_all_indicators(symbol_data)
                
                # Calculate screening scores
                latest_data = symbol_data.iloc[-1]
                
                # Technical score
                technical_score = await self._calculate_technical_score(symbol_data)
                
                # Volume score
                volume_score = await self._calculate_volume_score(symbol_data)
                
                # Momentum score
                momentum_score = await self._calculate_momentum_score(symbol_data)
                
                # Relative strength score
                relative_strength = await self._calculate_relative_strength(symbol_data)
                
                # Quality score (price action quality)
                quality_score = await self._calculate_quality_score(symbol_data)
                
                # Overall score (weighted combination)
                overall_score = (
                    technical_score * 0.3 +
                    momentum_score * 0.25 +
                    relative_strength * 0.2 +
                    volume_score * 0.15 +
                    quality_score * 0.1
                )
                
                result = {
                    'symbol': symbol,
                    'current_price': latest_data['close'],
                    'volume': latest_data['volume'],
                    'technical_score': technical_score,
                    'volume_score': volume_score,
                    'momentum_score': momentum_score,
                    'relative_strength': relative_strength,
                    'quality_score': quality_score,
                    'overall_score': overall_score,
                    'rsi': latest_data.get('rsi', 50),
                    'macd': latest_data.get('macd', 0),
                    'atr': latest_data.get('atr', 0),
                    'return_5d': (latest_data['close'] / latest_data['close_5d_ago'] - 1) * 100 if latest_data['close_5d_ago'] else 0,
                    'return_20d': (latest_data['close'] / latest_data['close_20d_ago'] - 1) * 100 if latest_data['close_20d_ago'] else 0
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {symbol}: {e}")
                continue
                
        return pd.DataFrame(results)
        
    async def _calculate_technical_score(self, data: pd.DataFrame) -> float:
        """Calculate technical analysis score"""
        
        try:
            latest = data.iloc[-1]
            score = 0
            
            # RSI score (prefer RSI between 30-70)
            rsi = latest.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 25
            elif rsi < 30:
                score += 35  # Oversold opportunity
            
            # MACD score
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            if macd > macd_signal:
                score += 25
                
            # Moving average score
            close = latest['close']
            sma_20 = latest.get('sma_20', close)
            sma_50 = latest.get('sma_50', close)
            
            if close > sma_20 > sma_50:
                score += 25
            elif close > sma_20:
                score += 15
                
            # Bollinger Bands score
            bb_upper = latest.get('bb_upper', close)
            bb_lower = latest.get('bb_lower', close)
            
            if bb_lower < close < bb_upper:
                score += 25
            elif close > bb_upper:
                score += 10  # Breakout
                
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0
            
    async def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Calculate volume analysis score"""
        
        try:
            latest = data.iloc[-1]
            avg_volume = data['volume'].tail(20).mean()
            
            # Volume ratio score
            volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 2:
                return 100  # High volume activity
            elif volume_ratio > 1.5:
                return 75
            elif volume_ratio > 1.2:
                return 50
            else:
                return 25
                
        except Exception as e:
            logger.error(f"Error calculating volume score: {e}")
            return 0
            
    async def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        
        try:
            latest = data.iloc[-1]
            
            # 5-day momentum
            return_5d = latest.get('return_5d', 0)
            
            # 20-day momentum  
            return_20d = latest.get('return_20d', 0)
            
            score = 0
            
            # Short-term momentum
            if return_5d > 2:
                score += 50
            elif return_5d > 0:
                score += 25
                
            # Medium-term momentum
            if return_20d > 5:
                score += 50
            elif return_20d > 0:
                score += 25
                
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0
            
    async def _calculate_relative_strength(self, data: pd.DataFrame) -> float:
        """Calculate relative strength vs market"""
        
        try:
            # Simplified relative strength calculation
            # In production, compare against Nifty index
            
            returns = data['close'].pct_change().tail(20)
            avg_return = returns.mean()
            
            # Convert to percentile score
            if avg_return > 0.01:  # 1% daily average
                return 100
            elif avg_return > 0.005:
                return 75
            elif avg_return > 0:
                return 50
            else:
                return 25
                
        except Exception as e:
            logger.error(f"Error calculating relative strength: {e}")
            return 50
            
    async def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate price action quality score"""
        
        try:
            # Price consistency and trend quality
            returns = data['close'].pct_change().tail(20)
            
            # Volatility score (prefer moderate volatility)
            volatility = returns.std()
            
            if 0.01 <= volatility <= 0.03:
                vol_score = 100
            elif volatility <= 0.05:
                vol_score = 75
            else:
                vol_score = 25
                
            # Trend consistency
            up_days = (returns > 0).sum()
            trend_score = (up_days / len(returns)) * 100
            
            return (vol_score + trend_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 50
            
    async def _rank_and_filter_stocks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rank and filter stocks based on screening criteria"""
        
        if data.empty:
            return []
            
        # Sort by overall score
        data = data.sort_values('overall_score', ascending=False)
        
        # Apply filters
        filtered_data = data[
            (data['overall_score'] >= 60) &
            (data['volume'] >= 10000) &  # Minimum volume
            (data['current_price'] >= 10) &  # Minimum price
            (data['current_price'] <= 5000)  # Maximum price
        ]
        
        # Limit to top stocks
        top_stocks = filtered_data.head(50)
        
        # Convert to list of dictionaries
        results = []
        for _, row in top_stocks.iterrows():
            result = row.to_dict()
            result['rank'] = len(results) + 1
            results.append(result)
            
        return results
        
    async def _store_screening_results(self, results: List[Dict], screening_date: datetime):
        """Store screening results in database"""
        
        try:
            records = []
            for result in results:
                record = (
                    screening_date,
                    result['symbol'],
                    'ALL',  # category
                    result['overall_score'],
                    result['rank'],
                    result['rank'],  # rank_category same as overall for now
                    result['technical_score'],
                    0,  # fundamental_score placeholder
                    result['relative_strength'],
                    result['volume_score'],
                    result['momentum_score'],
                    result['quality_score']
                )
                records.append(record)
                
            insert_query = """
            INSERT INTO screening_results 
            (date, symbol, category, score, rank_overall, rank_category, 
             technical_score, fundamental_score, relative_strength, 
             volume_score, momentum_score, quality_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            await self.db_manager.execute_many(insert_query, records)
            
        except Exception as e:
            logger.error(f"Error storing screening results: {e}")
            
    async def get_symbol_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Get detailed data for a specific symbol"""
        
        try:
            # First try cache
            if symbol in self.price_cache:
                cached_data = pd.DataFrame(self.price_cache[symbol])
                if not cached_data.empty:
                    return cached_data
                    
            # Fetch fresh data
            data = await self._fetch_symbol_data(symbol, period)
            
            if not data.empty:
                # Add technical indicators
                data = await self.technical_indicators.calculate_all_indicators(data)
                
                # Add features
                data = await self.feature_engineer.engineer_features(data)
                
            return data
            
        except Exception as e:
            logger.error(f"Error getting symbol data for {symbol}: {e}")
            return pd.DataFrame()
            
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        
        now = datetime.now()
        
        # NSE market hours (9:15 AM to 3:30 PM IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_open = market_open <= now <= market_close and now.weekday() < 5
        
        return {
            'is_open': is_open,
            'current_time': now,
            'market_open': market_open,
            'market_close': market_close,
            'is_weekend': now.weekday() >= 5,
            'seconds_to_open': max(0, (market_open - now).total_seconds()) if not is_open else 0,
            'seconds_to_close': max(0, (market_close - now).total_seconds()) if is_open else 0
        }
        
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available market data"""
        
        return {
            'total_stocks_universe': len(self.nse_stocks),
            'active_stocks': len(self.active_stocks),
            'cached_symbols': len(self.price_cache),
            'last_screening': max(self.screening_cache.keys()) if self.screening_cache else None,
            'screening_cache_size': len(self.screening_cache),
            'last_updates': {k: v.isoformat() for k, v in self.last_update.items()},
            'request_count': self.request_count
        }
