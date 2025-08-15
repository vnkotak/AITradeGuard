"""
Telegram Alerts Module
Sends trading alerts and reports via Telegram
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from io import BytesIO

from config import Config

logger = logging.getLogger(__name__)

class TelegramAlerts:
    """
    Telegram bot for sending trading alerts and reports
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # Telegram API base URL
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Message formatting
        self.max_message_length = 4000
        self.emoji_map = {
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️',
            'money': '💰',
            'chart': '📈',
            'fire': '🔥',
            'rocket': '🚀',
            'target': '🎯',
            'shield': '🛡️',
            'clock': '⏰',
            'bell': '🔔'
        }
        
    async def initialize(self):
        """Initialize Telegram bot"""
        logger.info("Initializing Telegram alerts...")
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot token or chat ID not configured")
            return False
            
        try:
            # Test bot connection
            bot_info = await self._get_bot_info()
            if bot_info:
                logger.info(f"Telegram bot initialized: {bot_info.get('username', 'Unknown')}")
                return True
            else:
                logger.error("Failed to connect to Telegram bot")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Telegram bot: {e}")
            return False
            
    async def _get_bot_info(self) -> Optional[Dict]:
        """Get bot information"""
        
        try:
            url = f"{self.base_url}/getMe"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('result')
                    else:
                        logger.error(f"Telegram API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting bot info: {e}")
            return None
            
    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send text message to Telegram"""
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured, skipping message")
            return False
            
        try:
            # Split long messages
            messages = self._split_message(message)
            
            for msg in messages:
                success = await self._send_single_message(msg, parse_mode)
                if not success:
                    return False
                    
                # Rate limiting
                if len(messages) > 1:
                    await asyncio.sleep(1)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
            
    async def _send_single_message(self, message: str, parse_mode: str) -> bool:
        """Send single message to Telegram"""
        
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending single message: {e}")
            return False
            
    def _split_message(self, message: str) -> List[str]:
        """Split long messages into chunks"""
        
        if len(message) <= self.max_message_length:
            return [message]
            
        messages = []
        lines = message.split('\n')
        current_message = ""
        
        for line in lines:
            if len(current_message + line + '\n') <= self.max_message_length:
                current_message += line + '\n'
            else:
                if current_message:
                    messages.append(current_message.strip())
                current_message = line + '\n'
                
        if current_message:
            messages.append(current_message.strip())
            
        return messages
        
    async def send_trade_alert(self, trade_result: Dict, rationale: str) -> bool:
        """Send trade execution alert"""
        
        try:
            symbol = trade_result.get('symbol', 'N/A')
            execution_price = trade_result.get('execution_price', 0)
            quantity = trade_result.get('quantity', 0)
            trade_value = execution_price * quantity
            
            message = f"""
{self.emoji_map['success']} <b>TRADE EXECUTED</b>

{self.emoji_map['target']} <b>Symbol:</b> {symbol}
{self.emoji_map['money']} <b>Price:</b> ₹{execution_price:.2f}
{self.emoji_map['chart']} <b>Quantity:</b> {quantity:,.0f}
{self.emoji_map['money']} <b>Value:</b> ₹{trade_value:,.2f}

{self.emoji_map['info']} <b>AI Rationale:</b>
{rationale}

{self.emoji_map['clock']} <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending trade alert: {e}")
            return False
            
    async def send_premarket_summary(self, watchlist: List[Dict]) -> bool:
        """Send pre-market analysis summary"""
        
        try:
            if not watchlist:
                message = f"""
{self.emoji_map['info']} <b>PRE-MARKET ANALYSIS</b>

{self.emoji_map['clock']} <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

No stocks in today's watchlist. Market conditions may not be favorable for trading.
                """.strip()
            else:
                top_picks = sorted(watchlist, key=lambda x: x.get('ai_score', 0), reverse=True)[:5]
                
                watchlist_text = ""
                for i, stock in enumerate(top_picks, 1):
                    watchlist_text += f"""
{i}. <b>{stock.get('symbol', 'N/A')}</b>
   Price: ₹{stock.get('current_price', 0):.2f}
   AI Score: {stock.get('ai_score', 0):.2f}
   Risk: {stock.get('risk_level', 'N/A')}
                    """.strip() + "\n"
                
                message = f"""
{self.emoji_map['bell']} <b>PRE-MARKET ANALYSIS</b>

{self.emoji_map['clock']} <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{self.emoji_map['target']} <b>Watchlist:</b> {len(watchlist)} stocks

{self.emoji_map['fire']} <b>Top Picks:</b>
{watchlist_text}

{self.emoji_map['rocket']} Ready for market open!
                """.strip()
                
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending pre-market summary: {e}")
            return False
            
    async def send_daily_summary(self, performance_report: Dict) -> bool:
        """Send daily performance summary"""
        
        try:
            basic_metrics = performance_report.get('basic_metrics', {})
            trade_analysis = performance_report.get('trade_analysis', {})
            insights = performance_report.get('insights', [])
            
            # Performance data
            total_value = basic_metrics.get('total_value', 0)
            daily_return = basic_metrics.get('daily_return', 0)
            total_return = basic_metrics.get('total_return', 0)
            
            # Trade data
            total_trades = trade_analysis.get('total_trades', 0)
            win_rate = trade_analysis.get('win_rate', 0)
            total_pnl = trade_analysis.get('total_pnl', 0)
            
            # Format performance emoji
            perf_emoji = self.emoji_map['success'] if daily_return >= 0 else self.emoji_map['warning']
            
            message = f"""
{self.emoji_map['chart']} <b>DAILY PERFORMANCE SUMMARY</b>

{self.emoji_map['money']} <b>Portfolio Value:</b> ₹{total_value:,.2f}
{perf_emoji} <b>Daily Return:</b> {daily_return:.2f}%
{self.emoji_map['chart']} <b>Total Return:</b> {total_return:.2f}%

{self.emoji_map['target']} <b>Trading Activity:</b>
• Total Trades: {total_trades}
• Win Rate: {win_rate:.1f}%
• P&L: ₹{total_pnl:,.2f}

{self.emoji_map['info']} <b>Key Insights:</b>
            """.strip()
            
            # Add insights
            for insight in insights[:3]:  # Limit to top 3
                message += f"\n• {insight}"
                
            message += f"\n\n{self.emoji_map['clock']} <b>Report Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False
            
    async def send_risk_alert(self, risk_type: str, details: Dict) -> bool:
        """Send risk management alert"""
        
        try:
            emoji = self.emoji_map['warning']
            
            if risk_type == 'HIGH_DRAWDOWN':
                message = f"""
{emoji} <b>RISK ALERT: HIGH DRAWDOWN</b>

{self.emoji_map['shield']} <b>Current Drawdown:</b> {details.get('drawdown', 0):.2f}%
{self.emoji_map['money']} <b>Portfolio Value:</b> ₹{details.get('portfolio_value', 0):,.2f}
{self.emoji_map['chart']} <b>Peak Value:</b> ₹{details.get('peak_value', 0):,.2f}

{self.emoji_map['warning']} Consider reducing position sizes or taking defensive action.
                """.strip()
                
            elif risk_type == 'POSITION_LIMIT':
                message = f"""
{emoji} <b>RISK ALERT: POSITION LIMIT</b>

{self.emoji_map['target']} <b>Symbol:</b> {details.get('symbol', 'N/A')}
{self.emoji_map['shield']} <b>Position Size:</b> {details.get('position_size', 0):.2f}%
{self.emoji_map['warning']} <b>Limit:</b> {details.get('limit', 0):.2f}%

Trade was rejected due to position size limits.
                """.strip()
                
            elif risk_type == 'DAILY_LOSS':
                message = f"""
{emoji} <b>RISK ALERT: DAILY LOSS LIMIT</b>

{self.emoji_map['money']} <b>Daily P&L:</b> ₹{details.get('daily_pnl', 0):,.2f}
{self.emoji_map['shield']} <b>Loss Limit:</b> ₹{details.get('loss_limit', 0):,.2f}

Consider stopping trading for today.
                """.strip()
                
            else:
                message = f"""
{emoji} <b>RISK ALERT</b>

{self.emoji_map['warning']} {details.get('message', 'Risk alert triggered')}
                """.strip()
                
            message += f"\n\n{self.emoji_map['clock']} <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")
            return False
            
    async def send_system_status(self, status: Dict) -> bool:
        """Send system status update"""
        
        try:
            ai_summary = status.get('ai_summary', {})
            market_status = status.get('market_status', {})
            portfolio_summary = status.get('portfolio_summary', {})
            
            # System health emoji
            health_emoji = self.emoji_map['success'] if status.get('status') == 'healthy' else self.emoji_map['error']
            
            message = f"""
{health_emoji} <b>SYSTEM STATUS UPDATE</b>

{self.emoji_map['info']} <b>Market:</b> {'OPEN' if market_status.get('is_open', False) else 'CLOSED'}
{self.emoji_map['target']} <b>Active Positions:</b> {portfolio_summary.get('num_positions', 0)}
{self.emoji_map['chart']} <b>AI Approval Rate:</b> {ai_summary.get('recent_approval_rate', 0)*100:.1f}%

{self.emoji_map['shield']} <b>Risk Status:</b> {status.get('risk_status', 'OK')}
{self.emoji_map['money']} <b>Portfolio Value:</b> ₹{portfolio_summary.get('total_portfolio_value', 0):,.2f}

{self.emoji_map['clock']} <b>Status Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending system status: {e}")
            return False
            
    async def send_backtest_results(self, backtest_results: Dict) -> bool:
        """Send backtest results summary"""
        
        try:
            performance_metrics = backtest_results.get('performance_metrics', {})
            
            total_return = performance_metrics.get('total_return', 0)
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            win_rate = performance_metrics.get('win_rate', 0)
            total_trades = performance_metrics.get('total_trades', 0)
            
            # Grade emoji
            grade_emoji = self.emoji_map['success'] if total_return > 10 else self.emoji_map['warning']
            
            message = f"""
{self.emoji_map['chart']} <b>BACKTEST RESULTS</b>

{grade_emoji} <b>Performance Grade:</b> {backtest_results.get('analysis_results', {}).get('summary', {}).get('performance_grade', 'N/A')}

{self.emoji_map['money']} <b>Total Return:</b> {total_return:.2f}%
{self.emoji_map['target']} <b>Sharpe Ratio:</b> {sharpe_ratio:.2f}
{self.emoji_map['shield']} <b>Max Drawdown:</b> {max_drawdown:.2f}%
{self.emoji_map['chart']} <b>Win Rate:</b> {win_rate:.1f}%
{self.emoji_map['info']} <b>Total Trades:</b> {total_trades}

{self.emoji_map['fire']} <b>Final Value:</b> ₹{backtest_results.get('final_value', 0):,.2f}

{self.emoji_map['clock']} <b>Backtest ID:</b> {backtest_results.get('run_id', 'N/A')[:8]}
            """.strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending backtest results: {e}")
            return False
            
    async def send_error_alert(self, error_message: str, component: str = None) -> bool:
        """Send error alert"""
        
        try:
            component_text = f" in {component}" if component else ""
            
            message = f"""
{self.emoji_map['error']} <b>SYSTEM ERROR</b>

{self.emoji_map['warning']} <b>Error{component_text}:</b>
{error_message}

{self.emoji_map['clock']} <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check system logs for details.
            """.strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
            return False
            
    async def send_weekly_report(self, weekly_data: Dict) -> bool:
        """Send weekly performance report"""
        
        try:
            message = f"""
{self.emoji_map['chart']} <b>WEEKLY PERFORMANCE REPORT</b>

{self.emoji_map['money']} <b>Weekly Return:</b> {weekly_data.get('weekly_return', 0):.2f}%
{self.emoji_map['target']} <b>Total Trades:</b> {weekly_data.get('total_trades', 0)}
{self.emoji_map['fire']} <b>Best Day:</b> {weekly_data.get('best_day_return', 0):.2f}%
{self.emoji_map['shield']} <b>Worst Day:</b> {weekly_data.get('worst_day_return', 0):.2f}%

{self.emoji_map['chart']} <b>Win Rate:</b> {weekly_data.get('win_rate', 0):.1f}%
{self.emoji_map['money']} <b>Profit Factor:</b> {weekly_data.get('profit_factor', 0):.2f}
{self.emoji_map['target']} <b>Sharpe Ratio:</b> {weekly_data.get('sharpe_ratio', 0):.2f}

{self.emoji_map['info']} <b>Top Performing Stocks:</b>
            """.strip()
            
            # Add top performers
            top_performers = weekly_data.get('top_performers', [])
            for i, stock in enumerate(top_performers[:3], 1):
                symbol = stock.get('symbol', 'N/A')
                return_pct = stock.get('return_pct', 0)
                message += f"\n{i}. {symbol}: {return_pct:+.2f}%"
                
            message += f"\n\n{self.emoji_map['clock']} <b>Week Ending:</b> {datetime.now().strftime('%Y-%m-%d')}"
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending weekly report: {e}")
            return False
            
    async def send_photo(self, photo_data: bytes, caption: str = None) -> bool:
        """Send photo to Telegram"""
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured, skipping photo")
            return False
            
        try:
            url = f"{self.base_url}/sendPhoto"
            
            data = aiohttp.FormData()
            data.add_field('chat_id', self.chat_id)
            data.add_field('photo', photo_data, filename='chart.png', content_type='image/png')
            
            if caption:
                data.add_field('caption', caption)
                data.add_field('parse_mode', 'HTML')
                
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram photo API error {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending photo: {e}")
            return False
            
    async def send_chart_with_analysis(self, chart_data: bytes, analysis: Dict) -> bool:
        """Send chart with accompanying analysis"""
        
        try:
            # Prepare caption
            caption = f"""
{self.emoji_map['chart']} <b>MARKET ANALYSIS</b>

{self.emoji_map['target']} <b>Signal:</b> {analysis.get('signal', 'NEUTRAL')}
{self.emoji_map['info']} <b>Confidence:</b> {analysis.get('confidence', 0):.1f}%
{self.emoji_map['shield']} <b>Risk Level:</b> {analysis.get('risk_level', 'MEDIUM')}

{analysis.get('summary', 'Technical analysis chart')}
            """.strip()
            
            return await self.send_photo(chart_data, caption)
            
        except Exception as e:
            logger.error(f"Error sending chart with analysis: {e}")
            return False
            
    def format_currency(self, amount: float) -> str:
        """Format currency with Indian numbering system"""
        
        if amount >= 10000000:  # 1 crore
            return f"₹{amount/10000000:.1f}Cr"
        elif amount >= 100000:  # 1 lakh
            return f"₹{amount/100000:.1f}L"
        elif amount >= 1000:  # 1 thousand
            return f"₹{amount/1000:.1f}K"
        else:
            return f"₹{amount:.2f}"
            
    def format_percentage(self, value: float, precision: int = 2) -> str:
        """Format percentage with appropriate sign and color"""
        
        if value > 0:
            return f"+{value:.{precision}f}%"
        else:
            return f"{value:.{precision}f}%"
            
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        
        try:
            test_message = f"""
{self.emoji_map['info']} <b>TELEGRAM BOT TEST</b>

Connection test successful!

{self.emoji_map['clock']} <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            return await self.send_message(test_message)
            
        except Exception as e:
            logger.error(f"Error testing Telegram connection: {e}")
            return False
            
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert service status"""
        
        return {
            'configured': bool(self.bot_token and self.chat_id),
            'bot_token_set': bool(self.bot_token),
            'chat_id_set': bool(self.chat_id),
            'base_url': self.base_url if self.bot_token else None,
            'max_message_length': self.max_message_length
        }
