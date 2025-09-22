"""
Reinforcement Learning Framework for Trade Decision Optimization
Uses PPO algorithm to continuously improve trading decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import pickle
import os

import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from config import Config, RL_CONFIG
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning
    """
    
    def __init__(self, config: Config, historical_data: pd.DataFrame, initial_balance: float = 100000):
        super(TradingEnvironment, self).__init__()
        
        self.config = config
        self.historical_data = historical_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.portfolio_value = initial_balance
        self.max_portfolio_value = initial_balance
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: market features + portfolio state
        n_features = len(self.historical_data.columns) - 3  # Exclude date, symbol, target
        n_portfolio_features = 5  # balance, positions, portfolio_value, drawdown, returns
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + n_portfolio_features,),
            dtype=np.float32
        )
        
        # Episode parameters
        self.max_steps = len(self.historical_data) - 1
        self.lookback_window = 20
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        
        return self._get_observation()
        
    def step(self, action):
        """Execute one trading step"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
            
        # Get current market data
        current_data = self.historical_data.iloc[self.current_step]
        symbol = current_data['symbol']
        price = current_data['close']
        
        # Execute action
        reward = self._execute_action(action, symbol, price)
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Check if episode is done
        done = (
            self.current_step >= self.max_steps - 1 or
            self.portfolio_value <= self.initial_balance * 0.5  # Stop loss
        )
        
        # Move to next step
        self.current_step += 1
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'num_positions': len(self.positions),
            'total_trades': len(self.trade_history)
        }
        
        return self._get_observation(), reward, done, info
        
    def _execute_action(self, action, symbol, price) -> float:
        """Execute trading action and return reward"""
        reward = 0
        
        if action == 1:  # Buy
            reward = self._execute_buy(symbol, price)
        elif action == 2:  # Sell
            reward = self._execute_sell(symbol, price)
        # action == 0 is hold, no execution needed
        
        return reward
        
    def _execute_buy(self, symbol, price) -> float:
        """Execute buy order"""
        # Calculate position size based on risk management
        position_size = self._calculate_position_size(price)
        
        if position_size == 0 or self.balance < position_size * price:
            return -0.1  # Penalty for invalid trade
            
        # Execute trade
        self.balance -= position_size * price
        
        if symbol in self.positions:
            self.positions[symbol]['quantity'] += position_size
            # Update average price
            total_value = self.positions[symbol]['avg_price'] * (self.positions[symbol]['quantity'] - position_size) + price * position_size
            self.positions[symbol]['avg_price'] = total_value / self.positions[symbol]['quantity']
        else:
            self.positions[symbol] = {
                'quantity': position_size,
                'avg_price': price,
                'entry_time': self.current_step
            }
            
        # Record trade
        self.trade_history.append({
            'step': self.current_step,
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'quantity': position_size,
            'balance': self.balance
        })
        
        # Small negative reward for transaction costs
        return -0.01
        
    def _execute_sell(self, symbol, price) -> float:
        """Execute sell order"""
        if symbol not in self.positions or self.positions[symbol]['quantity'] == 0:
            return -0.1  # Penalty for invalid trade
            
        # Sell entire position
        quantity = self.positions[symbol]['quantity']
        avg_price = self.positions[symbol]['avg_price']
        
        self.balance += quantity * price
        
        # Calculate profit/loss
        pnl = (price - avg_price) * quantity
        pnl_pct = pnl / (avg_price * quantity)
        
        # Remove position
        del self.positions[symbol]
        
        # Record trade
        self.trade_history.append({
            'step': self.current_step,
            'action': 'SELL',
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'balance': self.balance
        })
        
        # Reward based on profit/loss
        return pnl_pct * 10  # Scale reward
        
    def _calculate_position_size(self, price) -> float:
        """Calculate position size based on risk management"""
        max_position_value = self.portfolio_value * self.config.MAX_POSITION_SIZE
        max_risk_value = self.portfolio_value * self.config.MAX_PORTFOLIO_RISK
        
        # Use smaller of the two limits
        position_value = min(max_position_value, max_risk_value)
        position_size = position_value / price
        
        # Ensure we have enough balance
        if position_size * price > self.balance:
            position_size = self.balance / price
            
        return max(0, position_size)
        
    def _update_portfolio_value(self):
        """Update current portfolio value"""
        positions_value = 0
        
        if self.current_step < len(self.historical_data):
            current_data = self.historical_data.iloc[self.current_step]
            
            for symbol, position in self.positions.items():
                # Get current price (simplified - in real trading, we'd get actual current price)
                current_price = current_data['close'] if current_data['symbol'] == symbol else position['avg_price']
                positions_value += position['quantity'] * current_price
                
        self.portfolio_value = self.balance + positions_value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation"""
        if self.current_step >= len(self.historical_data):
            # Return zeros if we're at the end
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
        # Market features (exclude date, symbol, target columns)
        current_data = self.historical_data.iloc[self.current_step]
        market_features = []
        
        for col in self.historical_data.columns:
            if col not in ['date', 'symbol', 'target', 'close']:  # close is used for price
                market_features.append(current_data[col])
                
        # Portfolio features
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        portfolio_returns = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        portfolio_features = [
            self.balance / self.initial_balance,  # Normalized balance
            len(self.positions),  # Number of positions
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            current_drawdown,  # Current drawdown
            portfolio_returns  # Portfolio returns
        ]
        
        # Combine features
        observation = np.array(market_features + portfolio_features, dtype=np.float32)
        
        # Handle any NaN values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
        
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio performance metrics"""
        returns = (self.portfolio_value - self.initial_balance) / self.initial_balance
        max_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        # Calculate Sharpe ratio (simplified)
        if len(self.trade_history) > 0:
            trade_returns = [t.get('pnl_pct', 0) for t in self.trade_history if 'pnl_pct' in t]
            if trade_returns:
                sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) if np.std(trade_returns) > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
            
        return {
            'total_return': returns,
            'portfolio_value': self.portfolio_value,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trade_history),
            'current_positions': len(self.positions)
        }

class RLTradingAgent:
    """
    Reinforcement Learning Trading Agent using PPO
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager()
        
        self.model = None
        self.training_env = None
        self.eval_env = None
        
        # Training parameters
        self.model_path = "models/rl_models/"
        self.best_model_path = f"{self.model_path}best_model"
        
        # Performance tracking
        self.training_history = []
        self.evaluation_results = []
        
    async def initialize(self):
        """Initialize RL agent"""
        logger.info("Initializing RL trading agent...")
        
        # Create model directory
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing model
        if os.path.exists(f"{self.best_model_path}.zip"):
            await self.load_model()
            logger.info("Loaded existing RL model")
        else:
            logger.info("No existing RL model found, will train new model")
            
    async def prepare_training_data(self, lookback_days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and validation data"""
        
        # Get historical data with features and targets
        query = """
        SELECT 
            date, symbol, close,
            feature_rsi, feature_macd, feature_bb_upper, feature_bb_lower,
            feature_atr, feature_volume_ratio, feature_momentum,
            feature_relative_strength, feature_volatility,
            CASE 
                WHEN future_return_5d > 0.02 THEN 1 
                ELSE 0 
            END as target
        FROM trade_features 
        WHERE date >= %s
        ORDER BY date, symbol
        """
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        df = await self.db_manager.fetch_dataframe(query, (cutoff_date,))
        
        if df.empty:
            raise ValueError("No training data available for RL")
            
        # Split into training and validation
        split_date = df['date'].quantile(0.8)
        train_df = df[df['date'] <= split_date].reset_index(drop=True)
        val_df = df[df['date'] > split_date].reset_index(drop=True)
        
        logger.info(f"RL training data: {len(train_df)} samples")
        logger.info(f"RL validation data: {len(val_df)} samples")
        
        return train_df, val_df
        
    async def train_agent(self, total_timesteps: int = 100000) -> Dict:
        """Train the RL agent"""
        logger.info("Starting RL agent training...")
        
        try:
            # Prepare data
            train_df, val_df = await self.prepare_training_data()
            
            # Create environments
            self.training_env = DummyVecEnv([
                lambda: TradingEnvironment(self.config, train_df)
            ])
            
            self.eval_env = TradingEnvironment(self.config, val_df)
            
            # Create or update model
            if self.model is None:
                self.model = PPO(
                    "MlpPolicy",
                    self.training_env,
                    learning_rate=RL_CONFIG['learning_rate'],
                    gamma=RL_CONFIG['gamma'],
                    gae_lambda=RL_CONFIG['gae_lambda'],
                    ent_coef=RL_CONFIG['ent_coef'],
                    vf_coef=RL_CONFIG['vf_coef'],
                    max_grad_norm=RL_CONFIG['max_grad_norm'],
                    n_steps=RL_CONFIG['n_steps'],
                    batch_size=RL_CONFIG['batch_size'],
                    n_epochs=RL_CONFIG['n_epochs'],
                    verbose=1,
                    tensorboard_log="./rl_tensorboard/"
                )
                
            # Evaluation callback
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=self.model_path,
                log_path="./rl_logs/",
                eval_freq=5000,
                deterministic=True,
                render=False
            )
            
            # Train model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback
            )
            
            # Evaluate final performance
            evaluation_results = await self.evaluate_agent()
            
            # Save model
            self.model.save(f"{self.model_path}final_model")
            
            logger.info("RL agent training completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error training RL agent: {e}")
            return {'error': str(e)}
            
    async def evaluate_agent(self, n_episodes: int = 10) -> Dict:
        """Evaluate agent performance"""
        
        if self.model is None or self.eval_env is None:
            return {'error': 'Model or evaluation environment not available'}
            
        logger.info(f"Evaluating RL agent over {n_episodes} episodes...")
        
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                
            # Get episode metrics
            metrics = self.eval_env.get_portfolio_metrics()
            episode_rewards.append(episode_reward)
            episode_metrics.append(metrics)
            
        # Aggregate results
        avg_reward = np.mean(episode_rewards)
        avg_return = np.mean([m['total_return'] for m in episode_metrics])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in episode_metrics])
        avg_max_drawdown = np.mean([m['max_drawdown'] for m in episode_metrics])
        
        results = {
            'avg_episode_reward': avg_reward,
            'avg_total_return': avg_return,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_max_drawdown': avg_max_drawdown,
            'episode_rewards': episode_rewards,
            'episode_metrics': episode_metrics
        }
        
        self.evaluation_results.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        logger.info(f"RL evaluation - Avg Reward: {avg_reward:.4f}, Avg Return: {avg_return:.4f}")
        
        return results
        
    async def get_action_recommendation(self, market_features: Dict[str, float]) -> Dict:
        """Get trading action recommendation from RL agent"""
        
        if self.model is None:
            return {
                'action': 0,  # Hold
                'confidence': 0,
                'action_probs': [1.0, 0.0, 0.0],
                'error': 'Model not trained'
            }
            
        try:
            # Create observation from market features
            observation = self._create_observation_from_features(market_features)
            
            # Get action probabilities
            action_probs = self.model.predict_proba(observation.reshape(1, -1))[0]
            
            # Get recommended action
            action, _states = self.model.predict(observation, deterministic=True)
            
            # Calculate confidence
            confidence = np.max(action_probs)
            
            action_names = ['HOLD', 'BUY', 'SELL']
            
            return {
                'action': int(action),
                'action_name': action_names[action],
                'confidence': float(confidence),
                'action_probs': action_probs.tolist(),
                'recommendation_strength': self._calculate_recommendation_strength(action_probs)
            }
            
        except Exception as e:
            logger.error(f"Error getting RL action recommendation: {e}")
            return {
                'action': 0,
                'confidence': 0,
                'action_probs': [1.0, 0.0, 0.0],
                'error': str(e)
            }
            
    def _create_observation_from_features(self, features: Dict[str, float]) -> np.ndarray:
        """Create observation array from feature dictionary"""
        
        # Expected feature order (should match training data)
        expected_features = [
            'feature_rsi', 'feature_macd', 'feature_bb_upper', 'feature_bb_lower',
            'feature_atr', 'feature_volume_ratio', 'feature_momentum',
            'feature_relative_strength', 'feature_volatility'
        ]
        
        # Extract market features
        market_features = []
        for feature_name in expected_features:
            market_features.append(features.get(feature_name, 0.0))
            
        # Add portfolio features (simplified for real-time prediction)
        portfolio_features = [
            1.0,  # Normalized balance (assume full balance available)
            0.0,  # Number of positions (assume no current positions)
            1.0,  # Normalized portfolio value
            0.0,  # Current drawdown
            0.0   # Portfolio returns
        ]
        
        observation = np.array(market_features + portfolio_features, dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
        
    def _calculate_recommendation_strength(self, action_probs: np.ndarray) -> str:
        """Calculate recommendation strength based on action probabilities"""
        
        max_prob = np.max(action_probs)
        
        if max_prob > 0.8:
            return "STRONG"
        elif max_prob > 0.6:
            return "MODERATE"
        else:
            return "WEAK"
            
    async def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.model_path}model_{timestamp}"
            self.model.save(save_path)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'training_history': self.training_history,
                'evaluation_results': self.evaluation_results,
                'config': RL_CONFIG
            }
            
            with open(f"{save_path}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"RL model saved: {save_path}")
            
    async def load_model(self, model_path: str = None):
        """Load a previously trained model"""
        try:
            if model_path is None:
                model_path = self.best_model_path
                
            self.model = PPO.load(f"{model_path}.zip")
            
            # Load metadata if available
            metadata_path = f"{model_path}_metadata.pkl"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.training_history = metadata.get('training_history', [])
                    self.evaluation_results = metadata.get('evaluation_results', [])
                    
            logger.info(f"RL model loaded: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            return False
            
    def get_model_summary(self) -> Dict:
        """Get summary of the RL model"""
        
        return {
            'model_trained': self.model is not None,
            'training_episodes': len(self.training_history),
            'evaluation_episodes': len(self.evaluation_results),
            'last_evaluation': self.evaluation_results[-1] if self.evaluation_results else None,
            'model_config': RL_CONFIG,
            'model_path': self.model_path
        }
