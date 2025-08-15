"""
Advanced ML Models for Trade Decision Making
Implements ensemble methods with sophisticated feature engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
import joblib
import optuna
from datetime import datetime, timedelta
import sqlite3

from config import Config, ML_CONFIG
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class EnsembleMLModel:
    """
    Advanced ensemble ML model for trade decision making
    Combines multiple algorithms with sophisticated feature engineering
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_weights = {}
        self.last_training_date = None
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all ML models in the ensemble"""
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            **ML_CONFIG['xgboost'],
            random_state=42,
            eval_metric='logloss'
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            **ML_CONFIG['random_forest'],
            random_state=42
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        # Neural Network
        self.models['neural_network'] = None  # Will be built dynamically
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
            
    def _build_neural_network(self, input_dim: int) -> keras.Model:
        """Build neural network architecture"""
        
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
        
    async def prepare_training_data(self, lookback_days: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with advanced feature engineering
        """
        if lookback_days is None:
            lookback_days = self.config.LOOKBACK_PERIOD
            
        # Get historical trade data with outcomes
        query = """
        SELECT 
            t.*,
            f.*,
            CASE 
                WHEN t.exit_price > t.entry_price THEN 1 
                ELSE 0 
            END as successful_trade
        FROM trades t
        JOIN trade_features f ON t.trade_id = f.trade_id
        WHERE t.exit_date >= %s
        AND t.exit_price IS NOT NULL
        ORDER BY t.entry_date
        """
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        df = await self.db_manager.fetch_dataframe(query, (cutoff_date,))
        
        if df.empty:
            raise ValueError("No training data available")
            
        # Prepare features and target
        feature_columns = [col for col in df.columns if col.startswith('feature_')]
        X = df[feature_columns]
        y = df['successful_trade']
        
        logger.info(f"Prepared training data: {len(X)} samples, {len(feature_columns)} features")
        
        return X, y
        
    async def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train all models in the ensemble with cross-validation
        """
        logger.info("Starting ensemble model training...")
        
        results = {}
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, model in self.models.items():
            if model_name == 'neural_network':
                continue  # Handle separately
                
            logger.info(f"Training {model_name}...")
            
            try:
                # Scale features
                X_scaled = self.scalers[model_name].fit_transform(X)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_scaled, y, 
                    cv=tscv, 
                    scoring='f1',
                    n_jobs=-1
                )
                
                # Final training
                model.fit(X_scaled, y)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(
                        zip(X.columns, model.feature_importances_)
                    )
                
                results[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'final_score': cv_scores[-1]  # Most recent fold
                }
                
                logger.info(f"{model_name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
                
        # Train neural network
        await self._train_neural_network(X, y, results)
        
        # Calculate model weights based on performance
        self._calculate_model_weights(results)
        
        self.last_training_date = datetime.now()
        
        # Save models
        await self.save_models()
        
        logger.info("Ensemble training completed")
        return results
        
    async def _train_neural_network(self, X: pd.DataFrame, y: pd.Series, results: Dict):
        """Train neural network model"""
        try:
            logger.info("Training neural network...")
            
            # Scale features
            X_scaled = self.scalers['neural_network'].fit_transform(X)
            
            # Build model
            self.models['neural_network'] = self._build_neural_network(X.shape[1])
            
            # Prepare data for TensorFlow
            X_train = X_scaled.astype(np.float32)
            y_train = y.values.astype(np.float32)
            
            # Split for validation
            split_idx = int(0.8 * len(X_train))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
            
            # Training with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.models['neural_network'].fit(
                X_train, y_train,
                epochs=ML_CONFIG['neural_network']['epochs'],
                batch_size=ML_CONFIG['neural_network']['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            val_loss, val_accuracy, val_precision, val_recall = self.models['neural_network'].evaluate(
                X_val, y_val, verbose=0
            )
            
            results['neural_network'] = {
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_loss': val_loss
            }
            
            logger.info(f"Neural network validation accuracy: {val_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            results['neural_network'] = {'error': str(e)}
            
    def _calculate_model_weights(self, results: Dict[str, Dict]):
        """Calculate ensemble weights based on model performance"""
        
        weights = {}
        total_score = 0
        
        for model_name, result in results.items():
            if 'error' in result:
                weights[model_name] = 0
                continue
                
            if model_name == 'neural_network':
                score = result.get('val_accuracy', 0)
            else:
                score = result.get('cv_mean', 0)
                
            weights[model_name] = max(score, 0)
            total_score += weights[model_name]
            
        # Normalize weights
        if total_score > 0:
            for model_name in weights:
                weights[model_name] /= total_score
        else:
            # Equal weights if all models failed
            n_models = len(weights)
            for model_name in weights:
                weights[model_name] = 1.0 / n_models
                
        self.model_weights = weights
        logger.info(f"Model weights: {weights}")
        
    async def predict_trade_outcome(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict trade outcome using ensemble of models
        """
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            if model is None or self.model_weights.get(model_name, 0) == 0:
                continue
                
            try:
                # Scale features
                X_scaled = self.scalers[model_name].transform(feature_df)
                
                if model_name == 'neural_network':
                    # Neural network prediction
                    prob = model.predict(X_scaled.astype(np.float32))[0][0]
                    pred = 1 if prob > 0.5 else 0
                else:
                    # Scikit-learn models
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0][1]
                    
                predictions[model_name] = pred
                probabilities[model_name] = prob
                
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                continue
                
        # Ensemble prediction
        ensemble_prob = 0
        total_weight = 0
        
        for model_name, prob in probabilities.items():
            weight = self.model_weights.get(model_name, 0)
            ensemble_prob += prob * weight
            total_weight += weight
            
        if total_weight > 0:
            ensemble_prob /= total_weight
        else:
            ensemble_prob = 0.5  # Neutral if no models available
            
        ensemble_prediction = 1 if ensemble_prob > 0.5 else 0
        
        # Calculate confidence
        individual_probs = list(probabilities.values())
        confidence = 1 - np.std(individual_probs) if individual_probs else 0
        
        return {
            'prediction': ensemble_prediction,
            'probability': ensemble_prob,
            'confidence': confidence,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'model_weights': self.model_weights
        }
        
    async def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 100):
        """
        Optimize hyperparameters using Optuna
        """
        logger.info("Starting hyperparameter optimization...")
        
        def objective(trial):
            # XGBoost hyperparameters
            xgb_params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0)
            }
            
            # Random Forest hyperparameters
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10)
            }
            
            # Create models with suggested parameters
            xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
            rf_model = RandomForestClassifier(**rf_params, random_state=42)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            xgb_scores = cross_val_score(xgb_model, X_scaled, y, cv=tscv, scoring='f1')
            rf_scores = cross_val_score(rf_model, X_scaled, y, cv=tscv, scoring='f1')
            
            # Return average of both models
            return (xgb_scores.mean() + rf_scores.mean()) / 2
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best cross-validation score: {study.best_value}")
        
        return study.best_params
        
    async def save_models(self):
        """Save trained models and scalers"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for model_name, model in self.models.items():
                if model is None:
                    continue
                    
                if model_name == 'neural_network':
                    model.save(f"models/saved/{model_name}_{timestamp}")
                else:
                    joblib.dump(model, f"models/saved/{model_name}_{timestamp}.pkl")
                    
                # Save scaler
                joblib.dump(
                    self.scalers[model_name], 
                    f"models/saved/{model_name}_scaler_{timestamp}.pkl"
                )
                
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'model_weights': self.model_weights,
                'feature_importance': self.feature_importance,
                'last_training_date': self.last_training_date.isoformat()
            }
            
            joblib.dump(metadata, f"models/saved/metadata_{timestamp}.pkl")
            
            logger.info(f"Models saved with timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            
    async def load_models(self, timestamp: str = None):
        """Load previously trained models"""
        try:
            if timestamp is None:
                # Load latest models
                import glob
                model_files = glob.glob("models/saved/metadata_*.pkl")
                if not model_files:
                    logger.warning("No saved models found")
                    return False
                    
                latest_file = max(model_files)
                timestamp = latest_file.split('_')[-1].split('.')[0]
                
            # Load metadata
            metadata = joblib.load(f"models/saved/metadata_{timestamp}.pkl")
            self.model_weights = metadata['model_weights']
            self.feature_importance = metadata['feature_importance']
            self.last_training_date = datetime.fromisoformat(metadata['last_training_date'])
            
            # Load models and scalers
            for model_name in self.models.keys():
                try:
                    if model_name == 'neural_network':
                        self.models[model_name] = keras.models.load_model(
                            f"models/saved/{model_name}_{timestamp}"
                        )
                    else:
                        self.models[model_name] = joblib.load(
                            f"models/saved/{model_name}_{timestamp}.pkl"
                        )
                        
                    self.scalers[model_name] = joblib.load(
                        f"models/saved/{model_name}_scaler_{timestamp}.pkl"
                    )
                    
                except FileNotFoundError:
                    logger.warning(f"Model {model_name} not found for timestamp {timestamp}")
                    continue
                    
            logger.info(f"Models loaded from timestamp: {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
            
    def get_feature_importance_summary(self) -> Dict[str, float]:
        """Get aggregated feature importance across all models"""
        
        if not self.feature_importance:
            return {}
            
        # Aggregate feature importance
        aggregated_importance = {}
        
        for model_name, importance_dict in self.feature_importance.items():
            weight = self.model_weights.get(model_name, 0)
            
            for feature, importance in importance_dict.items():
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = 0
                aggregated_importance[feature] += importance * weight
                
        # Normalize
        total_importance = sum(aggregated_importance.values())
        if total_importance > 0:
            for feature in aggregated_importance:
                aggregated_importance[feature] /= total_importance
                
        # Sort by importance
        return dict(sorted(
            aggregated_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
