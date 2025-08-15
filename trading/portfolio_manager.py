"""
Portfolio Manager for Position Tracking and P&L Management
Handles portfolio state, position management, and performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
import uuid

from config import Config
from database.db_manager import DatabaseManager
from utils.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Advanced portfolio management system
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        self.statistical_analysis = StatisticalAnalysis()
        
        # Portfolio state
        self.current_positions = {}
        self.cash_balance = 1000000  # 10 lakh starting capital
        self.initial_capital = 1000000
        self.total_portfolio_value = self.initial_capital
        
        # Performance tracking
        self.daily_pnl = 0
        self.total_pnl = 0
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        
        # Risk metrics
        self.max_drawdown = 0
        self.peak_portfolio_value = self.initial_capital
        self.current_drawdown = 0
        
        # Trade tracking
        self.active_trades = {}
        self.completed_trades = []
        self.trade_performance = {}
        
    async def initialize(self):
        """Initialize portfolio manager"""
        logger.info("Initializing Portfolio Manager...")
        
        try:
            # Initialize portfolio tracking tables
            await self._initialize_portfolio_tables()
            
            # Load existing portfolio state
            await self._load_portfolio_state()
            
            # Load trade history
            await self._load_trade_history()
            
            # Calculate current metrics
            await self._calculate_portfolio_metrics()
            
            logger.info(f"Portfolio Manager initialized - Total Value: ₹{self.total_portfolio_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error initializing Portfolio Manager: {e}")
            
    async def _initialize_portfolio_tables(self):
        """Initialize database tables for portfolio tracking"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS portfolio_positions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                quantity DECIMAL(15,2) NOT NULL,
                avg_price DECIMAL(12,4) NOT NULL,
                market_value DECIMAL(15,2),
                unrealized_pnl DECIMAL(15,2),
                realized_pnl DECIMAL(15,2) DEFAULT 0,
                first_purchase_date TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'ACTIVE',
                UNIQUE(symbol)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                trade_id VARCHAR(50) UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                entry_price DECIMAL(12,4) NOT NULL,
                exit_price DECIMAL(12,4),
                quantity DECIMAL(15,2) NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                exit_date TIMESTAMP,
                pnl DECIMAL(15,2),
                pnl_pct DECIMAL(8,4),
                commission DECIMAL(10,2) DEFAULT 0,
                slippage DECIMAL(8,4) DEFAULT 0,
                hold_days INTEGER,
                strategy VARCHAR(50),
                ai_confidence DECIMAL(8,4),
                status VARCHAR(20) DEFAULT 'OPEN',
                stop_loss DECIMAL(12,4),
                take_profit DECIMAL(12,4),
                exit_reason VARCHAR(50),
                max_favorable_excursion DECIMAL(8,4),
                max_adverse_excursion DECIMAL(8,4)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                total_value DECIMAL(15,2) NOT NULL,
                cash_balance DECIMAL(15,2) NOT NULL,
                positions_value DECIMAL(15,2) NOT NULL,
                daily_pnl DECIMAL(15,2) NOT NULL,
                total_pnl DECIMAL(15,2) NOT NULL,
                unrealized_pnl DECIMAL(15,2) NOT NULL,
                realized_pnl DECIMAL(15,2) NOT NULL,
                drawdown DECIMAL(8,4) NOT NULL,
                num_positions INTEGER NOT NULL,
                benchmark_value DECIMAL(15,2),
                alpha DECIMAL(8,4),
                beta DECIMAL(8,4),
                sharpe_ratio DECIMAL(8,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trade_performance (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                symbol VARCHAR(20),
                strategy VARCHAR(50),
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate DECIMAL(8,4) DEFAULT 0,
                avg_win DECIMAL(8,4) DEFAULT 0,
                avg_loss DECIMAL(8,4) DEFAULT 0,
                profit_factor DECIMAL(8,4) DEFAULT 0,
                max_consecutive_wins INTEGER DEFAULT 0,
                max_consecutive_losses INTEGER DEFAULT 0,
                largest_win DECIMAL(15,2) DEFAULT 0,
                largest_loss DECIMAL(15,2) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            await self.db_manager.execute_query(table_sql)
            
    async def _load_portfolio_state(self):
        """Load current portfolio state from database"""
        
        try:
            # Load positions
            positions_query = """
            SELECT * FROM portfolio_positions 
            WHERE status = 'ACTIVE'
            """
            
            positions = await self.db_manager.fetch_all(positions_query)
            
            for position in positions:
                self.current_positions[position['symbol']] = {
                    'quantity': float(position['quantity']),
                    'avg_price': float(position['avg_price']),
                    'market_value': float(position['market_value'] or 0),
                    'unrealized_pnl': float(position['unrealized_pnl'] or 0),
                    'realized_pnl': float(position['realized_pnl'] or 0),
                    'first_purchase_date': position['first_purchase_date']
                }
                
            # Load latest portfolio snapshot
            snapshot_query = """
            SELECT * FROM portfolio_snapshots 
            ORDER BY date DESC 
            LIMIT 1
            """
            
            snapshot = await self.db_manager.fetch_one(snapshot_query)
            
            if snapshot:
                self.cash_balance = float(snapshot['cash_balance'])
                self.total_portfolio_value = float(snapshot['total_value'])
                self.total_pnl = float(snapshot['total_pnl'])
                self.unrealized_pnl = float(snapshot['unrealized_pnl'])
                self.realized_pnl = float(snapshot['realized_pnl'])
                self.current_drawdown = float(snapshot['drawdown'])
                
            logger.info(f"Loaded portfolio state: {len(self.current_positions)} positions, "
                       f"₹{self.total_portfolio_value:,.2f} total value")
                       
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            
    async def _load_trade_history(self):
        """Load trade history"""
        
        try:
            # Load active trades
            active_trades_query = """
            SELECT * FROM trades 
            WHERE status = 'OPEN'
            ORDER BY entry_date DESC
            """
            
            active_trades = await self.db_manager.fetch_all(active_trades_query)
            
            for trade in active_trades:
                self.active_trades[trade['trade_id']] = dict(trade)
                
            # Load recent completed trades
            completed_trades_query = """
            SELECT * FROM trades 
            WHERE status = 'CLOSED' 
            ORDER BY exit_date DESC 
            LIMIT 100
            """
            
            completed_trades = await self.db_manager.fetch_all(completed_trades_query)
            self.completed_trades = [dict(trade) for trade in completed_trades]
            
            logger.info(f"Loaded {len(self.active_trades)} active trades, "
                       f"{len(self.completed_trades)} recent completed trades")
                       
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            
    async def open_position(self, trade_data: Dict) -> str:
        """Open a new position"""
        
        try:
            symbol = trade_data['symbol']
            side = trade_data['side']
            quantity = trade_data['quantity']
            price = trade_data['price']
            commission = trade_data.get('commission', 0)
            ai_confidence = trade_data.get('ai_confidence', 0)
            strategy = trade_data.get('strategy', 'AI_GATE')
            
            # Generate trade ID
            trade_id = str(uuid.uuid4())
            
            # Calculate trade value
            trade_value = quantity * price + commission
            
            # Check if we have enough cash
            if side == 'BUY' and trade_value > self.cash_balance:
                raise ValueError(f"Insufficient cash balance: ₹{self.cash_balance:,.2f} < ₹{trade_value:,.2f}")
                
            # Update cash balance
            if side == 'BUY':
                self.cash_balance -= trade_value
            else:
                self.cash_balance += (quantity * price - commission)
                
            # Update or create position
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                
                if side == 'BUY':
                    # Add to existing position
                    total_quantity = position['quantity'] + quantity
                    total_cost = (position['quantity'] * position['avg_price']) + (quantity * price)
                    new_avg_price = total_cost / total_quantity
                    
                    position['quantity'] = total_quantity
                    position['avg_price'] = new_avg_price
                else:
                    # Reduce existing position
                    position['quantity'] -= quantity
                    
                    # Calculate realized P&L for the sold portion
                    realized_pnl = (price - position['avg_price']) * quantity - commission
                    position['realized_pnl'] += realized_pnl
                    self.realized_pnl += realized_pnl
                    
                    if position['quantity'] <= 0:
                        # Position closed
                        del self.current_positions[symbol]
                        await self._update_position_in_db(symbol, None, 'CLOSED')
                    
            else:
                # New position
                if side == 'BUY':
                    self.current_positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'market_value': quantity * price,
                        'unrealized_pnl': 0,
                        'realized_pnl': 0,
                        'first_purchase_date': datetime.now()
                    }
                    
            # Update position in database
            if symbol in self.current_positions:
                await self._update_position_in_db(symbol, self.current_positions[symbol], 'ACTIVE')
                
            # Record trade
            trade_record = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'quantity': quantity,
                'entry_date': datetime.now(),
                'commission': commission,
                'strategy': strategy,
                'ai_confidence': ai_confidence,
                'status': 'OPEN',
                'stop_loss': trade_data.get('stop_loss'),
                'take_profit': trade_data.get('take_profit')
            }
            
            await self._record_trade(trade_record)
            self.active_trades[trade_id] = trade_record
            
            logger.info(f"Opened position: {symbol} {side} {quantity} @ ₹{price:.2f}")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            raise
            
    async def close_position(self, symbol: str, exit_price: float, exit_reason: str = 'MANUAL') -> Dict:
        """Close a position"""
        
        try:
            if symbol not in self.current_positions:
                raise ValueError(f"No active position found for {symbol}")
                
            position = self.current_positions[symbol]
            quantity = position['quantity']
            avg_price = position['avg_price']
            
            # Calculate P&L
            pnl = (exit_price - avg_price) * quantity
            pnl_pct = (exit_price / avg_price - 1) * 100
            
            # Update cash balance
            exit_value = quantity * exit_price
            commission = exit_value * 0.0003  # Estimated commission
            self.cash_balance += exit_value - commission
            
            # Update realized P&L
            self.realized_pnl += pnl - commission
            
            # Find corresponding active trade
            trade_to_close = None
            for trade_id, trade in self.active_trades.items():
                if trade['symbol'] == symbol and trade['status'] == 'OPEN':
                    trade_to_close = trade_id
                    break
                    
            if trade_to_close:
                # Update trade record
                trade = self.active_trades[trade_to_close]
                trade['exit_price'] = exit_price
                trade['exit_date'] = datetime.now()
                trade['pnl'] = pnl - commission
                trade['pnl_pct'] = pnl_pct
                trade['commission'] += commission
                trade['status'] = 'CLOSED'
                trade['exit_reason'] = exit_reason
                trade['hold_days'] = (datetime.now() - trade['entry_date']).days
                
                # Update trade in database
                await self._update_trade(trade)
                
                # Move to completed trades
                self.completed_trades.append(trade)
                del self.active_trades[trade_to_close]
                
            # Remove position
            del self.current_positions[symbol]
            await self._update_position_in_db(symbol, None, 'CLOSED')
            
            logger.info(f"Closed position: {symbol} @ ₹{exit_price:.2f}, P&L: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
            
            return {
                'symbol': symbol,
                'exit_price': exit_price,
                'quantity': quantity,
                'pnl': pnl - commission,
                'pnl_pct': pnl_pct,
                'commission': commission
            }
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            raise
            
    async def update_position(self, trade_result: Dict):
        """Update position based on trade execution result"""
        
        try:
            trade_data = {
                'symbol': trade_result['symbol'],
                'side': 'BUY',  # Assuming buy for now
                'quantity': trade_result['quantity'],
                'price': trade_result['execution_price'],
                'commission': trade_result.get('commission', 0),
                'ai_confidence': trade_result.get('ai_confidence', 0),
                'strategy': trade_result.get('strategy', 'AI_GATE'),
                'stop_loss': trade_result.get('stop_loss'),
                'take_profit': trade_result.get('take_profit')
            }
            
            trade_id = await self.open_position(trade_data)
            
            # Update portfolio metrics
            await self._calculate_portfolio_metrics()
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            raise
            
    async def update_market_values(self, price_data: Dict[str, float]):
        """Update market values of positions"""
        
        try:
            total_unrealized_pnl = 0
            total_positions_value = 0
            
            for symbol, position in self.current_positions.items():
                if symbol in price_data:
                    current_price = price_data[symbol]
                    quantity = position['quantity']
                    avg_price = position['avg_price']
                    
                    # Calculate market value and unrealized P&L
                    market_value = quantity * current_price
                    unrealized_pnl = (current_price - avg_price) * quantity
                    
                    # Update position
                    position['market_value'] = market_value
                    position['unrealized_pnl'] = unrealized_pnl
                    
                    total_positions_value += market_value
                    total_unrealized_pnl += unrealized_pnl
                    
                    # Update in database
                    await self._update_position_in_db(symbol, position, 'ACTIVE')
                    
            # Update portfolio totals
            self.unrealized_pnl = total_unrealized_pnl
            self.total_portfolio_value = self.cash_balance + total_positions_value
            self.total_pnl = self.realized_pnl + self.unrealized_pnl
            
            # Update drawdown
            if self.total_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.total_portfolio_value
                
            self.current_drawdown = (self.peak_portfolio_value - self.total_portfolio_value) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            logger.debug(f"Updated market values: Total P&L: ₹{self.total_pnl:.2f}, "
                        f"Unrealized: ₹{self.unrealized_pnl:.2f}")
                        
        except Exception as e:
            logger.error(f"Error updating market values: {e}")
            
    async def update_trade_outcomes(self):
        """Update trade outcomes for AI learning"""
        
        try:
            for trade_id, trade in self.active_trades.items():
                symbol = trade['symbol']
                
                if symbol in self.current_positions:
                    position = self.current_positions[symbol]
                    current_unrealized_pnl = position['unrealized_pnl']
                    
                    # Calculate maximum favorable and adverse excursions
                    entry_price = trade['entry_price']
                    current_price = position['market_value'] / position['quantity']
                    
                    # Update max excursions (simplified)
                    if current_price > entry_price:
                        mfe = (current_price / entry_price - 1) * 100
                        trade['max_favorable_excursion'] = max(
                            trade.get('max_favorable_excursion', 0), mfe
                        )
                    else:
                        mae = (1 - current_price / entry_price) * 100
                        trade['max_adverse_excursion'] = max(
                            trade.get('max_adverse_excursion', 0), mae
                        )
                        
                    # Update trade in database
                    await self._update_trade(trade)
                    
        except Exception as e:
            logger.error(f"Error updating trade outcomes: {e}")
            
    async def emergency_close_positions(self):
        """Emergency close all positions"""
        
        logger.warning("Emergency closing all positions...")
        
        try:
            symbols_to_close = list(self.current_positions.keys())
            
            for symbol in symbols_to_close:
                # Get current market price (simplified - would need real-time data)
                position = self.current_positions[symbol]
                current_price = position['avg_price']  # Use avg price as fallback
                
                await self.close_position(symbol, current_price, 'EMERGENCY')
                
            logger.warning(f"Emergency closed {len(symbols_to_close)} positions")
            
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
            
    async def _update_position_in_db(self, symbol: str, position: Dict = None, status: str = 'ACTIVE'):
        """Update position in database"""
        
        try:
            if position is None:
                # Delete position
                delete_query = "DELETE FROM portfolio_positions WHERE symbol = %s"
                await self.db_manager.execute_query(delete_query, (symbol,))
            else:
                # Upsert position
                upsert_query = """
                INSERT INTO portfolio_positions 
                (symbol, quantity, avg_price, market_value, unrealized_pnl, 
                 realized_pnl, first_purchase_date, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    avg_price = EXCLUDED.avg_price,
                    market_value = EXCLUDED.market_value,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    realized_pnl = EXCLUDED.realized_pnl,
                    last_updated = CURRENT_TIMESTAMP,
                    status = EXCLUDED.status
                """
                
                values = (
                    symbol,
                    position['quantity'],
                    position['avg_price'],
                    position.get('market_value', 0),
                    position.get('unrealized_pnl', 0),
                    position.get('realized_pnl', 0),
                    position.get('first_purchase_date', datetime.now()),
                    status
                )
                
                await self.db_manager.execute_query(upsert_query, values)
                
        except Exception as e:
            logger.error(f"Error updating position in DB: {e}")
            
    async def _record_trade(self, trade: Dict):
        """Record trade in database"""
        
        try:
            insert_query = """
            INSERT INTO trades 
            (trade_id, symbol, side, entry_price, quantity, entry_date,
             commission, strategy, ai_confidence, status, stop_loss, take_profit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                trade['trade_id'],
                trade['symbol'],
                trade['side'],
                trade['entry_price'],
                trade['quantity'],
                trade['entry_date'],
                trade.get('commission', 0),
                trade.get('strategy', 'AI_GATE'),
                trade.get('ai_confidence', 0),
                trade['status'],
                trade.get('stop_loss'),
                trade.get('take_profit')
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            
    async def _update_trade(self, trade: Dict):
        """Update trade in database"""
        
        try:
            update_query = """
            UPDATE trades SET
                exit_price = %s,
                exit_date = %s,
                pnl = %s,
                pnl_pct = %s,
                commission = %s,
                hold_days = %s,
                status = %s,
                exit_reason = %s,
                max_favorable_excursion = %s,
                max_adverse_excursion = %s
            WHERE trade_id = %s
            """
            
            values = (
                trade.get('exit_price'),
                trade.get('exit_date'),
                trade.get('pnl'),
                trade.get('pnl_pct'),
                trade.get('commission', 0),
                trade.get('hold_days'),
                trade['status'],
                trade.get('exit_reason'),
                trade.get('max_favorable_excursion'),
                trade.get('max_adverse_excursion'),
                trade['trade_id']
            )
            
            await self.db_manager.execute_query(update_query, values)
            
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            
    async def _calculate_portfolio_metrics(self):
        """Calculate portfolio performance metrics"""
        
        try:
            # Basic metrics
            positions_value = sum(pos['market_value'] for pos in self.current_positions.values())
            self.total_portfolio_value = self.cash_balance + positions_value
            
            # Performance metrics
            total_return = (self.total_portfolio_value / self.initial_capital - 1) * 100
            
            # Daily P&L (simplified)
            yesterday_value = self.total_portfolio_value - self.total_pnl  # Approximate
            self.daily_pnl = self.total_portfolio_value - yesterday_value
            
            logger.debug(f"Portfolio metrics calculated: Total Return: {total_return:.2f}%, "
                        f"Drawdown: {self.current_drawdown:.2f}%")
                        
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            
    async def create_daily_snapshot(self):
        """Create daily portfolio snapshot"""
        
        try:
            today = datetime.now().date()
            
            # Calculate positions value
            positions_value = sum(pos['market_value'] for pos in self.current_positions.values())
            
            # Calculate metrics
            await self._calculate_portfolio_metrics()
            
            # Calculate benchmark performance (simplified)
            benchmark_value = self.initial_capital  # Placeholder
            alpha = 0  # Placeholder
            beta = 1  # Placeholder
            
            # Calculate Sharpe ratio (simplified)
            if len(self.completed_trades) > 0:
                returns = [trade['pnl_pct'] for trade in self.completed_trades if trade.get('pnl_pct')]
                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
                
            # Insert snapshot
            insert_query = """
            INSERT INTO portfolio_snapshots 
            (date, total_value, cash_balance, positions_value, daily_pnl,
             total_pnl, unrealized_pnl, realized_pnl, drawdown, num_positions,
             benchmark_value, alpha, beta, sharpe_ratio)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                total_value = EXCLUDED.total_value,
                cash_balance = EXCLUDED.cash_balance,
                positions_value = EXCLUDED.positions_value,
                daily_pnl = EXCLUDED.daily_pnl,
                total_pnl = EXCLUDED.total_pnl,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                realized_pnl = EXCLUDED.realized_pnl,
                drawdown = EXCLUDED.drawdown,
                num_positions = EXCLUDED.num_positions,
                sharpe_ratio = EXCLUDED.sharpe_ratio
            """
            
            values = (
                today,
                self.total_portfolio_value,
                self.cash_balance,
                positions_value,
                self.daily_pnl,
                self.total_pnl,
                self.unrealized_pnl,
                self.realized_pnl,
                self.current_drawdown,
                len(self.current_positions),
                benchmark_value,
                alpha,
                beta,
                sharpe_ratio
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
            logger.info(f"Created daily snapshot for {today}")
            
        except Exception as e:
            logger.error(f"Error creating daily snapshot: {e}")
            
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        
        positions_value = sum(pos['market_value'] for pos in self.current_positions.values())
        total_return = (self.total_portfolio_value / self.initial_capital - 1) * 100
        
        # Calculate win rate
        if self.completed_trades:
            winning_trades = sum(1 for trade in self.completed_trades if trade.get('pnl', 0) > 0)
            win_rate = (winning_trades / len(self.completed_trades)) * 100
        else:
            win_rate = 0
            
        return {
            'total_portfolio_value': self.total_portfolio_value,
            'cash_balance': self.cash_balance,
            'positions_value': positions_value,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_return_pct': total_return,
            'current_drawdown_pct': self.current_drawdown * 100,
            'max_drawdown_pct': self.max_drawdown * 100,
            'num_positions': len(self.current_positions),
            'num_active_trades': len(self.active_trades),
            'num_completed_trades': len(self.completed_trades),
            'win_rate': win_rate,
            'positions': {symbol: {
                'quantity': pos['quantity'],
                'avg_price': pos['avg_price'],
                'market_value': pos['market_value'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'unrealized_pnl_pct': (pos['unrealized_pnl'] / (pos['quantity'] * pos['avg_price'])) * 100
            } for symbol, pos in self.current_positions.items()}
        }
