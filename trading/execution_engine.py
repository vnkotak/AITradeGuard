"""
Trade Execution Engine
Handles order execution, position management, and trade lifecycle
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
from trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Advanced trade execution engine with order management
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_manager = DatabaseManager()
        self.risk_manager = RiskManager(self.config)
        
        # Order management
        self.pending_orders = {}
        self.executed_orders = {}
        self.order_history = []
        
        # Execution parameters
        self.slippage_model = SlippageModel()
        self.execution_algo = ExecutionAlgorithm()
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_slippage': 0,
            'average_execution_time': 0
        }
        
    async def initialize(self):
        """Initialize execution engine"""
        logger.info("Initializing Execution Engine...")
        
        try:
            # Initialize risk manager
            await self.risk_manager.initialize()
            
            # Initialize order tracking tables
            await self._initialize_order_tables()
            
            # Load pending orders
            await self._load_pending_orders()
            
            logger.info("Execution Engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Execution Engine: {e}")
            
    async def _initialize_order_tables(self):
        """Initialize database tables for order tracking"""
        
        tables = [
            """
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                order_id VARCHAR(50) UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                order_type VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(15,2) NOT NULL,
                price DECIMAL(12,4),
                stop_price DECIMAL(12,4),
                time_in_force VARCHAR(10) DEFAULT 'GTC',
                status VARCHAR(20) DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                submitted_at TIMESTAMP,
                filled_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                filled_quantity DECIMAL(15,2) DEFAULT 0,
                avg_fill_price DECIMAL(12,4),
                commission DECIMAL(10,2) DEFAULT 0,
                slippage DECIMAL(8,4) DEFAULT 0,
                rejection_reason TEXT,
                parent_order_id VARCHAR(50),
                ai_decision_id INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS executions (
                id SERIAL PRIMARY KEY,
                execution_id VARCHAR(50) UNIQUE NOT NULL,
                order_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(15,2) NOT NULL,
                price DECIMAL(12,4) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_venue VARCHAR(50),
                commission DECIMAL(10,2) DEFAULT 0,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS execution_metrics (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                total_orders INTEGER DEFAULT 0,
                successful_orders INTEGER DEFAULT 0,
                failed_orders INTEGER DEFAULT 0,
                average_slippage DECIMAL(8,4) DEFAULT 0,
                average_execution_time DECIMAL(8,2) DEFAULT 0,
                total_commission DECIMAL(12,2) DEFAULT 0,
                UNIQUE(date)
            )
            """
        ]
        
        for table_sql in tables:
            await self.db_manager.execute_query(table_sql)
            
    async def _load_pending_orders(self):
        """Load pending orders from database"""
        
        try:
            query = """
            SELECT * FROM orders 
            WHERE status IN ('PENDING', 'PARTIALLY_FILLED', 'SUBMITTED')
            ORDER BY created_at
            """
            
            orders = await self.db_manager.fetch_all(query)
            
            for order in orders:
                order_dict = dict(order)
                self.pending_orders[order_dict['order_id']] = order_dict
                
            logger.info(f"Loaded {len(self.pending_orders)} pending orders")
            
        except Exception as e:
            logger.error(f"Error loading pending orders: {e}")
            
    async def execute_trade(self, signal: Dict, ai_decision: Dict) -> Dict[str, Any]:
        """Execute trade based on AI-approved signal"""
        
        logger.info(f"Executing trade for {signal['symbol']} - {signal['signal_type']}")
        
        try:
            # Validate trade parameters
            validation_result = await self._validate_trade_parameters(signal, ai_decision)
            
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'trade_id': None
                }
                
            # Create orders
            orders = await self._create_trade_orders(signal, ai_decision)
            
            if not orders:
                return {
                    'success': False,
                    'error': 'Failed to create orders',
                    'trade_id': None
                }
                
            # Execute primary order
            primary_order = orders['primary']
            execution_result = await self._execute_order(primary_order)
            
            if execution_result['success']:
                # Create stop loss and take profit orders
                await self._create_exit_orders(orders, execution_result)
                
                # Update position
                await self._update_position_after_execution(execution_result)
                
                # Record trade
                trade_record = await self._record_trade(signal, ai_decision, execution_result, orders)
                
                return {
                    'success': True,
                    'trade_id': trade_record['trade_id'],
                    'execution_price': execution_result['execution_price'],
                    'quantity': execution_result['executed_quantity'],
                    'slippage': execution_result['slippage'],
                    'commission': execution_result['commission'],
                    'orders': orders
                }
            else:
                return {
                    'success': False,
                    'error': execution_result['error'],
                    'trade_id': None
                }
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'error': str(e),
                'trade_id': None
            }
            
    async def _validate_trade_parameters(self, signal: Dict, ai_decision: Dict) -> Dict[str, Any]:
        """Validate trade parameters before execution"""
        
        try:
            # Check AI approval
            if not ai_decision.get('approved', False):
                return {'valid': False, 'error': 'Trade not approved by AI'}
                
            # Check position size
            position_size = ai_decision.get('position_size', 0)
            if position_size <= 0:
                return {'valid': False, 'error': 'Invalid position size'}
                
            # Check price validity
            current_price = signal.get('current_price', 0)
            if current_price <= 0:
                return {'valid': False, 'error': 'Invalid current price'}
                
            # Check risk limits
            risk_check = await self.risk_manager.assess_trade_risk(signal)
            if not risk_check.get('risk_approved', False):
                return {'valid': False, 'error': 'Trade rejected by risk management'}
                
            # Check market hours
            market_status = await self._check_market_status()
            if not market_status['is_open']:
                return {'valid': False, 'error': 'Market is closed'}
                
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Error validating trade parameters: {e}")
            return {'valid': False, 'error': str(e)}
            
    async def _check_market_status(self) -> Dict[str, bool]:
        """Check if market is open for trading"""
        
        now = datetime.now()
        
        # NSE market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekday = now.weekday() < 5
        is_market_hours = market_open <= now <= market_close
        
        return {
            'is_open': is_weekday and is_market_hours,
            'is_weekday': is_weekday,
            'is_market_hours': is_market_hours
        }
        
    async def _create_trade_orders(self, signal: Dict, ai_decision: Dict) -> Dict[str, Dict]:
        """Create orders for the trade"""
        
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            current_price = signal['current_price']
            position_size = ai_decision['position_size']
            
            # Calculate portfolio value for position sizing
            portfolio_value = await self._get_portfolio_value()
            
            # Calculate order quantity
            order_value = portfolio_value * position_size
            quantity = order_value / current_price
            
            # Create primary order
            primary_order = {
                'order_id': str(uuid.uuid4()),
                'symbol': symbol,
                'order_type': 'MARKET',  # Start with market orders for simplicity
                'side': 'BUY' if signal_type == 'BUY' else 'SELL',
                'quantity': round(quantity, 2),
                'price': None,  # Market order
                'time_in_force': 'IOC',  # Immediate or Cancel
                'status': 'CREATED',
                'ai_decision_confidence': ai_decision['confidence']
            }
            
            # Create stop loss order (contingent)
            stop_loss_price = ai_decision.get('stop_loss', 0)
            stop_loss_order = None
            
            if stop_loss_price > 0:
                stop_loss_order = {
                    'order_id': str(uuid.uuid4()),
                    'symbol': symbol,
                    'order_type': 'STOP_LOSS',
                    'side': 'SELL' if signal_type == 'BUY' else 'BUY',
                    'quantity': quantity,
                    'stop_price': stop_loss_price,
                    'time_in_force': 'GTC',
                    'status': 'CREATED',
                    'parent_order_id': primary_order['order_id']
                }
                
            # Create take profit order (contingent)
            take_profit_price = ai_decision.get('take_profit', 0)
            take_profit_order = None
            
            if take_profit_price > 0:
                take_profit_order = {
                    'order_id': str(uuid.uuid4()),
                    'symbol': symbol,
                    'order_type': 'LIMIT',
                    'side': 'SELL' if signal_type == 'BUY' else 'BUY',
                    'quantity': quantity,
                    'price': take_profit_price,
                    'time_in_force': 'GTC',
                    'status': 'CREATED',
                    'parent_order_id': primary_order['order_id']
                }
                
            orders = {
                'primary': primary_order,
                'stop_loss': stop_loss_order,
                'take_profit': take_profit_order
            }
            
            return orders
            
        except Exception as e:
            logger.error(f"Error creating trade orders: {e}")
            return {}
            
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        
        try:
            query = """
            SELECT SUM(market_value) as total_value
            FROM portfolio_positions 
            WHERE status = 'ACTIVE'
            """
            
            result = await self.db_manager.fetch_one(query)
            
            if result and result['total_value']:
                return float(result['total_value'])
            else:
                # Default portfolio value if no positions
                return 1000000  # 10 lakh default
                
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 1000000  # Default value
            
    async def _execute_order(self, order: Dict) -> Dict[str, Any]:
        """Execute individual order"""
        
        logger.info(f"Executing order {order['order_id']} for {order['symbol']}")
        
        try:
            # Store order in database
            await self._store_order(order)
            
            # Update order status
            order['status'] = 'SUBMITTED'
            order['submitted_at'] = datetime.now()
            
            # Simulate order execution (in production, this would connect to broker API)
            execution_result = await self._simulate_order_execution(order)
            
            if execution_result['success']:
                # Update order status
                order['status'] = 'FILLED'
                order['filled_at'] = datetime.now()
                order['filled_quantity'] = execution_result['executed_quantity']
                order['avg_fill_price'] = execution_result['execution_price']
                order['slippage'] = execution_result['slippage']
                order['commission'] = execution_result['commission']
                
                # Store execution
                await self._store_execution(order, execution_result)
                
                # Update statistics
                self.execution_stats['total_orders'] += 1
                self.execution_stats['successful_executions'] += 1
                
            else:
                # Update order status
                order['status'] = 'REJECTED'
                order['rejection_reason'] = execution_result['error']
                
                self.execution_stats['total_orders'] += 1
                self.execution_stats['failed_executions'] += 1
                
            # Update order in database
            await self._update_order(order)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing order {order['order_id']}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _simulate_order_execution(self, order: Dict) -> Dict[str, Any]:
        """Simulate order execution (replace with actual broker integration)"""
        
        try:
            symbol = order['symbol']
            order_type = order['order_type']
            side = order['side']
            quantity = order['quantity']
            
            # Get current market price
            current_price = await self._get_current_price(symbol)
            
            if current_price <= 0:
                return {
                    'success': False,
                    'error': 'Unable to get current price'
                }
                
            # Calculate slippage
            slippage = self.slippage_model.calculate_slippage(order, current_price)
            
            # Calculate execution price
            if order_type == 'MARKET':
                if side == 'BUY':
                    execution_price = current_price * (1 + slippage)
                else:
                    execution_price = current_price * (1 - slippage)
            else:
                execution_price = order.get('price', current_price)
                
            # Calculate commission
            commission = self._calculate_commission(quantity, execution_price)
            
            # Simulate execution delay
            await asyncio.sleep(0.1)  # 100ms execution delay
            
            return {
                'success': True,
                'execution_id': str(uuid.uuid4()),
                'executed_quantity': quantity,
                'execution_price': round(execution_price, 4),
                'slippage': slippage,
                'commission': commission,
                'execution_time': datetime.now(),
                'venue': 'NSE'
            }
            
        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        
        try:
            # In production, this would get real-time price from data feed
            # For simulation, get latest price from database
            
            query = """
            SELECT close FROM market_data 
            WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
            """
            
            result = await self.db_manager.fetch_one(query, (symbol,))
            
            if result:
                return float(result['close'])
            else:
                logger.warning(f"No price data found for {symbol}")
                return 0
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0
            
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission"""
        
        # NSE commission structure (simplified)
        trade_value = quantity * price
        
        # Brokerage: 0.01% to 0.05% of trade value
        brokerage = trade_value * 0.0003  # 0.03%
        
        # STT (Securities Transaction Tax): 0.1% for delivery
        stt = trade_value * 0.001
        
        # Exchange charges: ~0.00325%
        exchange_charges = trade_value * 0.0000325
        
        # GST: 18% on brokerage + exchange charges
        gst = (brokerage + exchange_charges) * 0.18
        
        # SEBI charges: ₹10 per crore
        sebi_charges = trade_value * 0.000001
        
        total_commission = brokerage + stt + exchange_charges + gst + sebi_charges
        
        return round(total_commission, 2)
        
    async def _store_order(self, order: Dict):
        """Store order in database"""
        
        try:
            insert_query = """
            INSERT INTO orders 
            (order_id, symbol, order_type, side, quantity, price, stop_price, 
             time_in_force, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                order['order_id'],
                order['symbol'],
                order['order_type'],
                order['side'],
                order['quantity'],
                order.get('price'),
                order.get('stop_price'),
                order['time_in_force'],
                order['status'],
                datetime.now()
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing order: {e}")
            
    async def _update_order(self, order: Dict):
        """Update order in database"""
        
        try:
            update_query = """
            UPDATE orders SET
                status = %s,
                submitted_at = %s,
                filled_at = %s,
                cancelled_at = %s,
                filled_quantity = %s,
                avg_fill_price = %s,
                commission = %s,
                slippage = %s,
                rejection_reason = %s
            WHERE order_id = %s
            """
            
            values = (
                order['status'],
                order.get('submitted_at'),
                order.get('filled_at'),
                order.get('cancelled_at'),
                order.get('filled_quantity', 0),
                order.get('avg_fill_price'),
                order.get('commission', 0),
                order.get('slippage', 0),
                order.get('rejection_reason'),
                order['order_id']
            )
            
            await self.db_manager.execute_query(update_query, values)
            
        except Exception as e:
            logger.error(f"Error updating order: {e}")
            
    async def _store_execution(self, order: Dict, execution_result: Dict):
        """Store execution details"""
        
        try:
            insert_query = """
            INSERT INTO executions 
            (execution_id, order_id, symbol, side, quantity, price, 
             timestamp, execution_venue, commission)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                execution_result['execution_id'],
                order['order_id'],
                order['symbol'],
                order['side'],
                execution_result['executed_quantity'],
                execution_result['execution_price'],
                execution_result['execution_time'],
                execution_result['venue'],
                execution_result['commission']
            )
            
            await self.db_manager.execute_query(insert_query, values)
            
        except Exception as e:
            logger.error(f"Error storing execution: {e}")
            
    async def _create_exit_orders(self, orders: Dict, execution_result: Dict):
        """Create stop loss and take profit orders after main execution"""
        
        try:
            if orders.get('stop_loss'):
                stop_loss_order = orders['stop_loss']
                stop_loss_order['status'] = 'PENDING'
                
                # Store stop loss order
                await self._store_order(stop_loss_order)
                self.pending_orders[stop_loss_order['order_id']] = stop_loss_order
                
            if orders.get('take_profit'):
                take_profit_order = orders['take_profit']
                take_profit_order['status'] = 'PENDING'
                
                # Store take profit order
                await self._store_order(take_profit_order)
                self.pending_orders[take_profit_order['order_id']] = take_profit_order
                
        except Exception as e:
            logger.error(f"Error creating exit orders: {e}")
            
    async def _update_position_after_execution(self, execution_result: Dict):
        """Update position tracking after execution"""
        
        try:
            # Update risk manager
            position_data = {
                'quantity': execution_result['executed_quantity'],
                'avg_price': execution_result['execution_price'],
                'market_value': execution_result['executed_quantity'] * execution_result['execution_price'],
                'unrealized_pnl': 0  # Initial PnL is zero
            }
            
            await self.risk_manager.update_position(
                execution_result['symbol'], 
                position_data
            )
            
        except Exception as e:
            logger.error(f"Error updating position after execution: {e}")
            
    async def _record_trade(self, signal: Dict, ai_decision: Dict, execution_result: Dict, orders: Dict) -> Dict:
        """Record completed trade"""
        
        try:
            trade_id = str(uuid.uuid4())
            
            trade_record = {
                'trade_id': trade_id,
                'symbol': signal['symbol'],
                'signal_type': signal['signal_type'],
                'entry_price': execution_result['execution_price'],
                'quantity': execution_result['executed_quantity'],
                'ai_confidence': ai_decision['confidence'],
                'execution_time': execution_result['execution_time'],
                'commission': execution_result['commission'],
                'slippage': execution_result['slippage'],
                'orders': orders
            }
            
            # Store in trade history
            self.order_history.append(trade_record)
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return {'trade_id': str(uuid.uuid4())}
            
    async def monitor_pending_orders(self):
        """Monitor and manage pending orders"""
        
        logger.info("Monitoring pending orders...")
        
        try:
            for order_id, order in list(self.pending_orders.items()):
                # Check if order should be executed or cancelled
                await self._check_pending_order(order)
                
        except Exception as e:
            logger.error(f"Error monitoring pending orders: {e}")
            
    async def _check_pending_order(self, order: Dict):
        """Check individual pending order"""
        
        try:
            symbol = order['symbol']
            order_type = order['order_type']
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            
            if current_price <= 0:
                return
                
            should_execute = False
            
            # Check execution conditions
            if order_type == 'STOP_LOSS':
                stop_price = order['stop_price']
                side = order['side']
                
                if side == 'SELL' and current_price <= stop_price:
                    should_execute = True
                elif side == 'BUY' and current_price >= stop_price:
                    should_execute = True
                    
            elif order_type == 'LIMIT':
                limit_price = order['price']
                side = order['side']
                
                if side == 'BUY' and current_price <= limit_price:
                    should_execute = True
                elif side == 'SELL' and current_price >= limit_price:
                    should_execute = True
                    
            if should_execute:
                # Execute the order
                execution_result = await self._execute_order(order)
                
                if execution_result['success']:
                    # Remove from pending orders
                    if order['order_id'] in self.pending_orders:
                        del self.pending_orders[order['order_id']]
                        
        except Exception as e:
            logger.error(f"Error checking pending order {order.get('order_id', 'unknown')}: {e}")
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        
        try:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                order['status'] = 'CANCELLED'
                order['cancelled_at'] = datetime.now()
                
                # Update in database
                await self._update_order(order)
                
                # Remove from pending
                del self.pending_orders[order_id]
                
                logger.info(f"Order {order_id} cancelled")
                return True
            else:
                logger.warning(f"Order {order_id} not found in pending orders")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
            
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution engine summary"""
        
        return {
            'pending_orders': len(self.pending_orders),
            'execution_stats': self.execution_stats,
            'recent_executions': len(self.order_history),
            'success_rate': (
                self.execution_stats['successful_executions'] / 
                max(self.execution_stats['total_orders'], 1)
            ) * 100
        }


class SlippageModel:
    """Model for calculating expected slippage"""
    
    def calculate_slippage(self, order: Dict, current_price: float) -> float:
        """Calculate expected slippage for an order"""
        
        try:
            # Base slippage
            base_slippage = 0.0005  # 0.05%
            
            # Order size impact
            quantity = order['quantity']
            order_value = quantity * current_price
            
            # Size impact (simplified)
            if order_value > 1000000:  # > 10 lakh
                size_impact = 0.002
            elif order_value > 500000:  # > 5 lakh
                size_impact = 0.001
            else:
                size_impact = 0.0005
                
            # Market impact (time of day)
            hour = datetime.now().hour
            if 9 <= hour <= 10 or 14 <= hour <= 15:  # High volatility hours
                time_impact = 0.001
            else:
                time_impact = 0.0003
                
            total_slippage = base_slippage + size_impact + time_impact
            
            return min(total_slippage, 0.01)  # Cap at 1%
            
        except Exception:
            return 0.001  # Default 0.1%


class ExecutionAlgorithm:
    """Execution algorithm for optimizing order execution"""
    
    def optimize_execution(self, order: Dict) -> Dict:
        """Optimize order execution strategy"""
        
        # For now, use simple execution
        # In production, implement TWAP, VWAP, or other algorithms
        
        return {
            'algorithm': 'IMMEDIATE',
            'slice_count': 1,
            'time_horizon': 0
        }
