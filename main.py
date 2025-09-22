#!/usr/bin/env python3
"""
Advanced AI-Gated Algorithmic Trading System for NSE Equities
Main entry point for the trading system
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, time
from typing import Dict, Any
import os

from config import Config
from database.db_manager import DatabaseManager
from data.market_data import MarketDataManager
from trading.ai_gate import AIGate
from trading.execution_engine import ExecutionEngine
from trading.portfolio_manager import PortfolioManager
from analytics.performance_analytics import PerformanceAnalytics
from alerts.telegram_alerts import TelegramAlerts

from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        self.market_data = MarketDataManager()
        self.ai_gate = AIGate()
        self.execution_engine = ExecutionEngine()
        self.portfolio_manager = PortfolioManager()
        self.performance_analytics = PerformanceAnalytics()
        self.telegram_alerts = TelegramAlerts()
        
        self.running = False
        self.daily_trade_count = 0
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing AI-Gated Trading System...")
        
        # Initialize database
        await self.db_manager.initialize()
        
        # Initialize AI models
        await self.ai_gate.initialize()
        
        # Initialize market data feeds
        await self.market_data.initialize()
        
        # Load portfolio state
        await self.portfolio_manager.initialize()
        
        logger.info("System initialization completed successfully")
        
    async def run_trading_session(self):
        """Main trading session loop"""
        logger.info("Starting trading session...")
        
        try:
            # Pre-market analysis
            await self.pre_market_analysis()
            
            # Main trading loop during market hours
            while self.is_market_open() and self.running:
                await self.trading_cycle()
                await asyncio.sleep(self.config.SCAN_INTERVAL)
                
            # Post-market analysis
            await self.post_market_analysis()
            
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
            await self.telegram_alerts.send_error_alert(str(e))
            
    async def pre_market_analysis(self):
        """Pre-market preparation and analysis"""
        logger.info("Conducting pre-market analysis...")
        
        # Update market regime detection
        await self.ai_gate.update_market_regime()
        
        # Screen stocks for potential trades
        candidate_stocks = await self.market_data.screen_stocks()
        
        # Generate watchlist
        watchlist = await self.ai_gate.generate_watchlist(candidate_stocks)
        
        logger.info(f"Generated watchlist with {len(watchlist)} stocks")
        
        # Send pre-market summary
        await self.telegram_alerts.send_premarket_summary(watchlist)
        
    async def trading_cycle(self):
        """Single trading cycle iteration"""
        try:
            # Skip if daily trade limit reached
            if self.daily_trade_count >= self.config.MAX_DAILY_TRADES:
                return
                
            # Get current market data
            market_data = await self.market_data.get_current_data()
            
            # Generate trade signals
            signals = await self.ai_gate.generate_signals(market_data)
            
            # Process each signal through AI gate
            for signal in signals:
                if self.daily_trade_count >= self.config.MAX_DAILY_TRADES:
                    break
                    
                # AI gate decision
                decision = await self.ai_gate.evaluate_trade(signal)
                
                if decision['approved']:
                    # Execute approved trade
                    trade_result = await self.execution_engine.execute_trade(
                        signal, decision
                    )
                    
                    if trade_result['success']:
                        self.daily_trade_count += 1
                        
                        # Update portfolio
                        await self.portfolio_manager.update_position(trade_result)
                        
                        # Send trade alert
                        await self.telegram_alerts.send_trade_alert(
                            trade_result, decision['rationale']
                        )
                        
                        logger.info(f"Trade executed: {trade_result['symbol']}")
                
                # Log decision for learning
                await self.ai_gate.log_decision(signal, decision)
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            
    async def post_market_analysis(self):
        """Post-market analysis and learning"""
        logger.info("Conducting post-market analysis...")
        
        # Update trade outcomes
        await self.portfolio_manager.update_trade_outcomes()
        
        # Retrain AI models with new data
        await self.ai_gate.retrain_models()
        
        # Generate performance report
        performance_report = await self.performance_analytics.generate_daily_report()
        
        # Send daily summary
        await self.telegram_alerts.send_daily_summary(performance_report)
        
        # Reset daily counters
        self.daily_trade_count = 0
        
        logger.info("Post-market analysis completed")
        
    def is_market_open(self) -> bool:
        """Check if NSE market is open"""
        now = datetime.now().time()
        market_open = time(9, 15)  # 9:15 AM
        market_close = time(15, 30)  # 3:30 PM
        
        # Check if it's a weekday and within market hours
        weekday = datetime.now().weekday() < 5  # Monday = 0, Sunday = 6
        return weekday and market_open <= now <= market_close
        
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading system...")
        self.running = False
        
        # Close all positions if required
        await self.portfolio_manager.emergency_close_positions()
        
        # Save models and data
        await self.ai_gate.save_models()
        
        # Close database connections
        await self.db_manager.close()
        
        logger.info("System shutdown completed")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
        

async def main():
    """Main application entry point"""
    # Set up signal handlers
    trading_system = TradingSystem()
    
    signal.signal(signal.SIGINT, trading_system.signal_handler)
    signal.signal(signal.SIGTERM, trading_system.signal_handler)
    
    try:
        # Initialize system
        await trading_system.initialize()
        
        # Start trading
        trading_system.running = True
        
        if trading_system.is_market_open():
            await trading_system.run_trading_session()
        else:
            logger.info("Market is closed. Running in monitoring mode...")
            # Run background tasks for model training, analysis etc.
            while trading_system.running:
                await asyncio.sleep(60)  # Check every minute
                
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await trading_system.shutdown()

#if __name__ == "__main__":
#    asyncio.run(main())


app = FastAPI()

# Create a global trading system instance
trading_system = TradingSystem()

@app.on_event("startup")
async def startup_event():
    await trading_system.initialize()
    trading_system.running = True
    logger.info("Trading system initialized on startup")

@app.get("/")
async def root():
    return {"status": "Trading system is running"}

@app.get("/run")
async def run_trading():
    if trading_system.is_market_open():
        await trading_system.run_trading_session()
        return {"message": "Trading session completed"}
    else:
        return {"message": "Market is closed"}

@app.on_event("shutdown")
async def shutdown_event():
    await trading_system.shutdown()
    logger.info("Trading system shutdown on app termination")
