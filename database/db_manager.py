"""
Database Manager for AI-Gated Trading System
Handles all database operations with PostgreSQL
"""

import asyncio
import asyncpg
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import json
import os

from config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Advanced database manager with connection pooling and transaction support
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.pool = None
        self.connection_params = self._get_connection_params()
        
    def _get_connection_params(self) -> Dict[str, str]:
        """Get database connection parameters"""
        
        # Use DATABASE_URL if available, otherwise construct from individual params
        database_url = self.config.DATABASE_URL
        
        if database_url and database_url != "postgresql://localhost/trading_db":
            return {'dsn': database_url}
        else:
            return {
                'host': self.config.PGHOST,
                'port': int(self.config.PGPORT),
                'database': self.config.PGDATABASE,
                'user': self.config.PGUSER,
                'password': self.config.PGPASSWORD
            }
    
    async def initialize(self):
        """Initialize database connection pool"""
        logger.info("Initializing database connection pool...")
        
        try:
            if 'dsn' in self.connection_params:
                self.pool = await asyncpg.create_pool(
                    dsn=self.connection_params['dsn'],
                    min_size=5,
                    max_size=20,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300.0,
                    command_timeout=60
                )
            else:
                self.pool = await asyncpg.create_pool(
                    host=self.connection_params['host'],
                    port=self.connection_params['port'],
                    database=self.connection_params['database'],
                    user=self.connection_params['user'],
                    password=self.connection_params['password'],
                    min_size=5,
                    max_size=20,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300.0,
                    command_timeout=60
                )
                
            logger.info("Database connection pool initialized successfully")
            
            # Test connection
            async with self.pool.acquire() as connection:
                result = await connection.fetchval("SELECT version()")
                logger.info(f"Connected to PostgreSQL: {result}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
            
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
            
    async def execute_query(self, query: str, params: Tuple = None) -> str:
        """Execute a query (INSERT, UPDATE, DELETE)"""
        
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as connection:
                if params:
                    result = await connection.execute(query, *params)
                else:
                    result = await connection.execute(query)
                return result
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
            
    async def execute_many(self, query: str, params_list: List[Tuple]) -> str:
        """Execute query with multiple parameter sets"""
        
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as connection:
                result = await connection.executemany(query, params_list)
                return result
                
        except Exception as e:
            logger.error(f"Error executing many: {e}")
            logger.error(f"Query: {query}")
            raise
            
    async def fetch_one(self, query: str, params: Tuple = None) -> Optional[Dict]:
        """Fetch single row"""
        
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as connection:
                if params:
                    row = await connection.fetchrow(query, *params)
                else:
                    row = await connection.fetchrow(query)
                    
                return dict(row) if row else None
                
        except Exception as e:
            logger.error(f"Error fetching one: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
            
    async def fetch_all(self, query: str, params: Tuple = None) -> List[Dict]:
        """Fetch multiple rows"""
        
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as connection:
                if params:
                    rows = await connection.fetch(query, *params)
                else:
                    rows = await connection.fetch(query)
                    
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error fetching all: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
            
    async def fetch_dataframe(self, query: str, params: Tuple = None) -> pd.DataFrame:
        """Fetch query results as pandas DataFrame"""
        
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as connection:
                if params:
                    rows = await connection.fetch(query, *params)
                else:
                    rows = await connection.fetch(query)
                    
                if not rows:
                    return pd.DataFrame()
                    
                # Convert to DataFrame
                data = [dict(row) for row in rows]
                df = pd.DataFrame(data)
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching dataframe: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
            
    async def fetch_value(self, query: str, params: Tuple = None) -> Any:
        """Fetch single value"""
        
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as connection:
                if params:
                    value = await connection.fetchval(query, *params)
                else:
                    value = await connection.fetchval(query)
                    
                return value
                
        except Exception as e:
            logger.error(f"Error fetching value: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
            
    async def execute_transaction(self, operations: List[Dict]) -> bool:
        """Execute multiple operations in a transaction"""
        
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as connection:
                async with connection.transaction():
                    for operation in operations:
                        query = operation['query']
                        params = operation.get('params')
                        
                        if params:
                            await connection.execute(query, *params)
                        else:
                            await connection.execute(query)
                            
                return True
                
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            return False
            
    async def copy_from_dataframe(self, table_name: str, df: pd.DataFrame, columns: List[str] = None):
        """Bulk insert from pandas DataFrame using COPY"""
        
        if not self.pool:
            await self.initialize()
            
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame provided for table {table_name}")
                return
                
            if columns is None:
                columns = df.columns.tolist()
                
            # Prepare data
            records = []
            for _, row in df.iterrows():
                record = []
                for col in columns:
                    value = row[col]
                    if pd.isna(value):
                        record.append(None)
                    else:
                        record.append(value)
                records.append(tuple(record))
                
            async with self.pool.acquire() as connection:
                await connection.copy_records_to_table(
                    table_name, 
                    records=records,
                    columns=columns
                )
                
            logger.info(f"Bulk inserted {len(records)} records to {table_name}")
            
        except Exception as e:
            logger.error(f"Error in bulk insert to {table_name}: {e}")
            raise
            
    async def create_indexes(self):
        """Create database indexes for performance"""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data(symbol, date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON trades(entry_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_timestamp ON ai_decisions(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_symbol ON ai_decisions(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_screening_results_date ON screening_results(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_screening_results_score ON screening_results(score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trade_features_symbol_date ON trade_features(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_backtest_runs_created_at ON backtest_runs(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_backtest_trades_run_id ON backtest_trades(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_performance_reports_created_at ON performance_reports(created_at DESC)"
        ]
        
        try:
            for index_sql in indexes:
                await self.execute_query(index_sql)
                
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            
    async def create_partitions(self):
        """Create table partitions for large tables"""
        
        try:
            # Partition market_data by date (monthly partitions)
            partition_queries = [
                """
                -- Create partitioned market_data table if not exists
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'market_data_partitioned') THEN
                        CREATE TABLE market_data_partitioned (LIKE market_data INCLUDING ALL)
                        PARTITION BY RANGE (date);
                    END IF;
                END $$;
                """,
                """
                -- Create current month partition
                DO $$
                DECLARE
                    start_date date := date_trunc('month', CURRENT_DATE);
                    end_date date := start_date + interval '1 month';
                    partition_name text := 'market_data_' || to_char(start_date, 'YYYY_MM');
                BEGIN
                    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF market_data_partitioned 
                                   FOR VALUES FROM (%L) TO (%L)', 
                                   partition_name, start_date, end_date);
                END $$;
                """
            ]
            
            for query in partition_queries:
                await self.execute_query(query)
                
            logger.info("Table partitions created successfully")
            
        except Exception as e:
            logger.error(f"Error creating partitions: {e}")
            
    async def analyze_tables(self):
        """Update table statistics for query optimization"""
        
        try:
            analyze_queries = [
                "ANALYZE market_data",
                "ANALYZE trades",
                "ANALYZE portfolio_snapshots",
                "ANALYZE ai_decisions",
                "ANALYZE orders",
                "ANALYZE screening_results"
            ]
            
            for query in analyze_queries:
                await self.execute_query(query)
                
            logger.info("Table statistics updated")
            
        except Exception as e:
            logger.error(f"Error analyzing tables: {e}")
            
    async def vacuum_tables(self):
        """Vacuum tables to reclaim space and update statistics"""
        
        try:
            # Use a separate connection for VACUUM as it cannot be in a transaction
            if 'dsn' in self.connection_params:
                connection = await asyncpg.connect(dsn=self.connection_params['dsn'])
            else:
                connection = await asyncpg.connect(**self.connection_params)
                
            try:
                vacuum_queries = [
                    "VACUUM ANALYZE market_data",
                    "VACUUM ANALYZE trades", 
                    "VACUUM ANALYZE portfolio_snapshots",
                    "VACUUM ANALYZE ai_decisions"
                ]
                
                for query in vacuum_queries:
                    await connection.execute(query)
                    
                logger.info("Tables vacuumed successfully")
                
            finally:
                await connection.close()
                
        except Exception as e:
            logger.error(f"Error vacuuming tables: {e}")
            
    async def get_table_sizes(self) -> Dict[str, Dict]:
        """Get size information for all tables"""
        
        try:
            query = """
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """
            
            rows = await self.fetch_all(query)
            
            table_sizes = {}
            for row in rows:
                table_sizes[row['tablename']] = {
                    'size': row['size'],
                    'size_bytes': row['size_bytes']
                }
                
            return table_sizes
            
        except Exception as e:
            logger.error(f"Error getting table sizes: {e}")
            return {}
            
    async def backup_table(self, table_name: str, backup_name: str = None):
        """Create a backup of a table"""
        
        try:
            if backup_name is None:
                backup_name = f"{table_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            query = f"CREATE TABLE {backup_name} AS SELECT * FROM {table_name}"
            await self.execute_query(query)
            
            logger.info(f"Table {table_name} backed up as {backup_name}")
            
        except Exception as e:
            logger.error(f"Error backing up table {table_name}: {e}")
            raise
            
    async def restore_table(self, table_name: str, backup_name: str):
        """Restore a table from backup"""
        
        try:
            # Truncate current table and restore from backup
            operations = [
                {'query': f"TRUNCATE TABLE {table_name}"},
                {'query': f"INSERT INTO {table_name} SELECT * FROM {backup_name}"}
            ]
            
            success = await self.execute_transaction(operations)
            
            if success:
                logger.info(f"Table {table_name} restored from {backup_name}")
            else:
                raise Exception("Restore transaction failed")
                
        except Exception as e:
            logger.error(f"Error restoring table {table_name}: {e}")
            raise
            
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        
        try:
            queries = {
                'version': "SELECT version()",
                'current_database': "SELECT current_database()",
                'current_user': "SELECT current_user",
                'active_connections': "SELECT count(*) FROM pg_stat_activity",
                'database_size': "SELECT pg_size_pretty(pg_database_size(current_database()))"
            }
            
            info = {}
            for key, query in queries.items():
                info[key] = await self.fetch_value(query)
                
            return info
            
        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            return {}
            
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        
        try:
            start_time = datetime.now()
            
            # Test basic connectivity
            version = await self.fetch_value("SELECT version()")
            
            # Test write operation
            test_query = "CREATE TEMP TABLE health_check_test (id int)"
            await self.execute_query(test_query)
            
            # Test read operation
            tables_count = await self.fetch_value(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy',
                'version': version[:50],  # Truncate long version string
                'tables_count': tables_count,
                'response_time_ms': response_time,
                'pool_size': self.pool.get_size() if self.pool else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
