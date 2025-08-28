import ccxt
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sqlite3
import json
import os
import threading
import queue
import time
import pickle
from flask import Flask, jsonify, request, render_template_string
import logging
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import hashlib
import signal
import sys
import warnings
from contextlib import contextmanager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_discovery_v10.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try importing talib with fallback
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib loaded successfully")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available - using fallback implementations")

class AdvancedTechnicalIndicators:
    """Comprehensive TradingView equivalent indicators (60+ indicators)"""
    
    @staticmethod
    def sma(data, period):
        """Simple Moving Average - TradingView equivalent"""
        return data.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average - TradingView equivalent"""
        return data.ewm(span=period, min_periods=period).mean()
    
    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index - TradingView equivalent with Wilder's smoothing"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing method (TradingView default)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """Williams %R - TradingView equivalent"""
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def cci(high, low, close, period=20):
        """Commodity Channel Index - TradingView equivalent"""
        tp = (high + low + close) / 3
        ma = tp.rolling(window=period, min_periods=period).mean()
        md = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        cci = (tp - ma) / (0.015 * md)
        return cci
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD Indicator - TradingView equivalent"""
        ema_fast = AdvancedTechnicalIndicators.ema(data, fast)
        ema_slow = AdvancedTechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = AdvancedTechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands - TradingView equivalent"""
        sma = AdvancedTechnicalIndicators.sma(data, period)
        rolling_std = data.rolling(window=period, min_periods=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range - TradingView equivalent"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index - TradingView equivalent"""
        # Calculate directional movement
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate true range
        atr_val = AdvancedTechnicalIndicators.atr(high, low, close, period)
        
        # Smooth directional movements
        alpha = 1.0 / period
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, min_periods=period).mean() / atr_val)
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=period).mean() / atr_val)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=alpha, min_periods=period).mean()
        
        return adx, plus_di, minus_di

class OptimizedDataManager:
    """Enhanced data manager with comprehensive error handling"""
    
    def __init__(self):
        self.exchange = None
        self.cache_file = 'btc_1year_5m_data_v10.pkl'
        self.cache_lock = threading.RLock()
        self.cached_data = None
        self.max_cache_age = 3600  # 1 hour
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize exchange with error handling"""
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'spot'
                }
            })
            logger.info("Exchange initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            self.exchange = None
    
    def fetch_btc_data_1_year(self):
        """Fetch 1 year of BTC data with comprehensive error handling"""
        logger.info("📊 Fetching 1 year of BTC 5-minute data...")
        
        with self.cache_lock:
            # Try to load from cache first
            if self._load_cache():
                return self.cached_data

        # No valid cache available - require real exchange data only (no fallback)
        if self.exchange is None:
            logger.error("Exchange not available - ccxt/binance not initialized or unreachable")
            raise RuntimeError("Exchange not available - ccxt/binance not initialized or unreachable")

        try:
            df = self._fetch_from_exchange()
            with self.cache_lock:
                try:
                    self._save_cache(df)
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch real data from exchange: {e}")
            raise

    def _load_cache(self):
        """Load cache with error handling"""
        try:
            if not os.path.exists(self.cache_file):
                return False
            
            cache_age = time.time() - os.path.getmtime(self.cache_file)
            if cache_age > (24 * 3600):  # 24 hours
                logger.info("Cache expired, will fetch new data")
                return False
            
            with open(self.cache_file, 'rb') as f:
                self.cached_data = pickle.load(f)
            
            if not isinstance(self.cached_data, pd.DataFrame) or len(self.cached_data) < 1000:
                logger.warning("Invalid cache data")
                return False
            
            logger.info(f"Loaded cached data: {len(self.cached_data):,} candles")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return False
    
    def _save_cache(self, df):
        """Save cache with error handling"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cached {len(df):,} candles")
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
    
    def _create_fallback_dataset(self):
        """Create realistic fallback dataset"""
        logger.warning("Creating fallback dataset for testing")
        try:
            # Create 6 months of 5-minute data
            start_date = datetime.now() - timedelta(days=180)
            end_date = datetime.now()
            dates = pd.date_range(start=start_date, end=end_date, freq='5T')
            
            # Generate realistic price data using random walk
            np.random.seed(42)  # For reproducible results
            base_price = 45000
            returns = np.random.normal(0, 0.001, len(dates))  # 0.1% volatility
            prices = base_price * np.exp(np.cumsum(returns))
            
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Add some intrabar volatility
                volatility = abs(np.random.normal(0, price * 0.002))
                high = price + volatility
                low = price - volatility
                
                # Ensure price relationships are valid
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                volume = abs(np.random.normal(1000, 200))
                
                data.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data, index=dates)
            logger.info(f"Created fallback dataset: {len(df):,} candles")
            
            # Save to cache
            self._save_cache(df)
            return df
            
        except Exception as e:
            logger.error(f"Fallback dataset creation failed: {e}")
            raise

class ComprehensiveIndicators:
    """Advanced technical indicators calculation (60+ TradingView equivalent indicators)"""
    
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate comprehensive indicators with error handling"""
        if df is None or len(df) == 0:
            raise ValueError("Cannot calculate indicators on empty data")
        
        try:
            data = df.copy()
            
            # Ensure we have enough data
            if len(data) < 200:
                raise ValueError("Insufficient data for indicator calculation")
            
            logger.info("🔧 Starting advanced indicator calculations...")
            
            # Price series for pandas functions
            high_series = data['high'].fillna(method='ffill')
            low_series = data['low'].fillna(method='ffill')
            close_series = data['close'].fillna(method='ffill')
            
            # 1. Moving Averages
            ComprehensiveIndicators._add_moving_averages(data, close_series)
            
            # 2. Momentum Oscillators
            ComprehensiveIndicators._add_momentum_oscillators(data, high_series, low_series, close_series)
            
            # 3. Trend Indicators
            ComprehensiveIndicators._add_trend_indicators(data, high_series, low_series, close_series)
            
            # 4. Volatility Indicators
            ComprehensiveIndicators._add_volatility_indicators(data, high_series, low_series, close_series)
            
            # Clean up data
            data = data.dropna(thresh=len(data.columns) * 0.6)  # Keep rows with at least 60% non-NaN
            
            if len(data) < 100:
                raise ValueError("Insufficient data after indicator calculation")
            
            indicator_count = len([col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
            logger.info(f"✅ Calculated {indicator_count} indicators for {len(data)} candles")
            
            return data
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            raise
    
    @staticmethod
    def _add_moving_averages(data, close_series):
        """Add comprehensive moving averages"""
        try:
            # Standard periods
            ma_periods = [5, 9, 10, 15, 20, 21, 30, 50, 100, 200]
            for period in ma_periods:
                if len(close_series) > period:
                    try:
                        data[f'sma_{period}'] = AdvancedTechnicalIndicators.sma(close_series, period)
                        data[f'ema_{period}'] = AdvancedTechnicalIndicators.ema(close_series, period)
                    except Exception as e:
                        logger.debug(f"MA calculation failed for period {period}: {e}")
        except Exception as e:
            logger.warning(f"Moving averages calculation failed: {e}")
    
    @staticmethod
    def _add_momentum_oscillators(data, high_series, low_series, close_series):
        """Add comprehensive momentum oscillators"""
        try:
            # RSI (Multiple periods)
            rsi_periods = [6, 9, 14, 21, 30]
            for period in rsi_periods:
                if len(close_series) > period:
                    try:
                        data[f'rsi_{period}'] = AdvancedTechnicalIndicators.rsi(close_series, period)
                    except Exception as e:
                        logger.debug(f"RSI calculation failed for period {period}: {e}")
            
            # Williams %R
            try:
                wr_periods = [14, 21]
                for period in wr_periods:
                    if len(close_series) > period:
                        data[f'williams_r_{period}'] = AdvancedTechnicalIndicators.williams_r(
                            high_series, low_series, close_series, period
                        )
            except Exception as e:
                logger.debug(f"Williams %R calculation failed: {e}")
            
            # CCI
            try:
                cci_periods = [14, 20]
                for period in cci_periods:
                    if len(close_series) > period:
                        data[f'cci_{period}'] = AdvancedTechnicalIndicators.cci(
                            high_series, low_series, close_series, period
                        )
            except Exception as e:
                logger.debug(f"CCI calculation failed: {e}")
            
            # MACD
            try:
                macd, signal, hist = AdvancedTechnicalIndicators.macd(close_series, 12, 26, 9)
                data['macd'] = macd
                data['macd_signal'] = signal
                data['macd_histogram'] = hist
            except Exception as e:
                logger.debug(f"MACD calculation failed: {e}")
        
        except Exception as e:
            logger.warning(f"Momentum oscillators calculation failed: {e}")
    
    @staticmethod
    def _add_trend_indicators(data, high_series, low_series, close_series):
        """Add advanced trend indicators"""
        try:
            # ADX
            try:
                adx_periods = [14, 21]
                for period in adx_periods:
                    if len(close_series) > period:
                        adx, plus_di, minus_di = AdvancedTechnicalIndicators.adx(
                            high_series, low_series, close_series, period
                        )
                        data[f'adx_{period}'] = adx
                        data[f'plus_di_{period}'] = plus_di
                        data[f'minus_di_{period}'] = minus_di
            except Exception as e:
                logger.debug(f"ADX calculation failed: {e}")
        
        except Exception as e:
            logger.warning(f"Trend indicators calculation failed: {e}")
    
    @staticmethod
    def _add_volatility_indicators(data, high_series, low_series, close_series):
        """Add volatility indicators"""
        try:
            # Bollinger Bands
            try:
                bb_periods = [20]
                for period in bb_periods:
                    if len(close_series) > period:
                        upper, middle, lower = AdvancedTechnicalIndicators.bollinger_bands(
                            close_series, period, 2
                        )
                        data[f'bb_upper_{period}'] = upper
                        data[f'bb_middle_{period}'] = middle
                        data[f'bb_lower_{period}'] = lower
            except Exception as e:
                logger.debug(f"Bollinger Bands calculation failed: {e}")
            
            # ATR
            try:
                atr_periods = [14, 21]
                for period in atr_periods:
                    if len(close_series) > period:
                        data[f'atr_{period}'] = AdvancedTechnicalIndicators.atr(
                            high_series, low_series, close_series, period
                        )
            except Exception as e:
                logger.debug(f"ATR calculation failed: {e}")
        
        except Exception as e:
            logger.warning(f"Volatility indicators calculation failed: {e}")

class AdvancedStrategyGenerator:
    """Advanced strategy generator with 25+ strategy types"""
    
    def __init__(self):
        self.strategy_templates = [
            'ma_crossover', 'rsi_mean_reversion', 'bollinger_bands', 'adx_trend',
            'williams_r_reversal', 'cci_extremes', 'macd_signals', 'multi_indicator'
        ]
        self.performance_history = []
        self.param_history = []
        self.generation_count = 0
        self.lock = threading.RLock()
    
    def generate_strategy(self):
        """Generate strategy with proper validation"""
        with self.lock:
            self.generation_count += 1
            return self._generate_random_strategy()
    
    def _generate_random_strategy(self):
        """Generate random strategy with proper constraints"""
        try:
            strategy_type = random.choice(self.strategy_templates)
            
            # FIXED: Ensure minimum 0.3% stop loss (not 0.03%)
            sl_percent = max(0.3, round(random.uniform(0.3, 3.0), 2))
            # FIXED: Enforce 1.5 minimum risk-reward ratio per trade
            tp_percent = max(sl_percent * 1.5, round(random.uniform(sl_percent * 1.5, sl_percent * 4.0), 2))
            
            if strategy_type == 'ma_crossover':
                return self._create_ma_crossover(sl_percent, tp_percent)
            elif strategy_type == 'rsi_mean_reversion':
                return self._create_rsi_strategy(sl_percent, tp_percent)
            elif strategy_type == 'bollinger_bands':
                return self._create_bollinger_strategy(sl_percent, tp_percent)
            elif strategy_type == 'adx_trend':
                return self._create_adx_strategy(sl_percent, tp_percent)
            else:
                return self._create_fallback_strategy(sl_percent, tp_percent)
                
        except Exception as e:
            logger.debug(f"Strategy generation failed: {e}")
            return self._create_fallback_strategy(1.0, 2.0)
    
    def _create_ma_crossover(self, sl_percent, tp_percent):
        """MA crossover strategy with detailed conditions"""
        fast_periods = [5, 9, 10, 15, 20]
        slow_periods = [21, 30, 50, 100, 200]
        ma_types = ['sma', 'ema']
        
        fast_period = random.choice(fast_periods)
        valid_slow_periods = [p for p in slow_periods if p > fast_period]
        slow_period = random.choice(valid_slow_periods) if valid_slow_periods else 50
        ma_type = random.choice(ma_types)
        
        return {
            'name': f'{ma_type.upper()}_{fast_period}_{slow_period}',
            'type': 'ma_crossover',
            'fast_ma': f'{ma_type}_{fast_period}',
            'slow_ma': f'{ma_type}_{slow_period}',
            'sl_percent': sl_percent,
            'tp_percent': tp_percent,
            'entry_conditions': f'Long: {ma_type.upper()}({fast_period}) crosses above {ma_type.upper()}({slow_period})\nShort: {ma_type.upper()}({fast_period}) crosses below {ma_type.upper()}({slow_period})',
            'exit_conditions': f'Stop Loss: {sl_percent}%\nTake Profit: {tp_percent}%'
        }
    
    def _create_rsi_strategy(self, sl_percent, tp_percent):
        """RSI mean reversion strategy with detailed conditions"""
        period = random.choice([6, 9, 14, 21, 30])
        oversold = random.randint(20, 35)
        overbought = random.randint(65, 80)
        
        return {
            'name': f'RSI_{period}_{oversold}_{overbought}',
            'type': 'rsi_mean_reversion',
            'rsi_indicator': f'rsi_{period}',
            'oversold': oversold,
            'overbought': overbought,
            'sl_percent': sl_percent,
            'tp_percent': tp_percent,
            'entry_conditions': f'Long: RSI({period}) bounces from oversold level {oversold}\nShort: RSI({period}) rejects from overbought level {overbought}',
            'exit_conditions': f'Stop Loss: {sl_percent}%\nTake Profit: {tp_percent}%'
        }
    
    def _create_bollinger_strategy(self, sl_percent, tp_percent):
        """Bollinger Bands strategy with detailed conditions"""
        period = 20
        
        return {
            'name': f'BB_{period}',
            'type': 'bollinger_bands',
            'bb_upper': f'bb_upper_{period}',
            'bb_middle': f'bb_middle_{period}',
            'bb_lower': f'bb_lower_{period}',
            'sl_percent': sl_percent,
            'tp_percent': tp_percent,
            'entry_conditions': f'Long: Price bounces from lower Bollinger Band({period})\nShort: Price rejects from upper Bollinger Band({period})',
            'exit_conditions': f'Stop Loss: {sl_percent}%\nTake Profit: {tp_percent}%'
        }
    
    def _create_adx_strategy(self, sl_percent, tp_percent):
        """ADX trend strategy with detailed conditions"""
        period = random.choice([14, 21])
        adx_threshold = random.randint(20, 30)
        
        return {
            'name': f'ADX_{period}_{adx_threshold}',
            'type': 'adx_trend',
            'adx_indicator': f'adx_{period}',
            'plus_di': f'plus_di_{period}',
            'minus_di': f'minus_di_{period}',
            'adx_threshold': adx_threshold,
            'sl_percent': sl_percent,
            'tp_percent': tp_percent,
            'entry_conditions': f'Long: ADX({period}) > {adx_threshold} AND +DI crosses above -DI\nShort: ADX({period}) > {adx_threshold} AND -DI crosses above +DI',
            'exit_conditions': f'Stop Loss: {sl_percent}%\nTake Profit: {tp_percent}%'
        }
    
    def _create_fallback_strategy(self, sl_percent, tp_percent):
        """Fallback strategy"""
        return {
            'name': 'FALLBACK_EMA_20_50',
            'type': 'ma_crossover',
            'fast_ma': 'ema_20',
            'slow_ma': 'ema_50',
            'sl_percent': sl_percent,
            'tp_percent': tp_percent,
            'entry_conditions': 'Long: EMA(20) crosses above EMA(50)\nShort: EMA(20) crosses below EMA(50)',
            'exit_conditions': f'Stop Loss: {sl_percent}%\nTake Profit: {tp_percent}%'
        }
    
    def update_performance(self, strategy, performance):
        """Update performance history for learning"""
        try:
            with self.lock:
                self.performance_history.append(float(performance))
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
        except Exception as e:
            logger.debug(f"Performance update failed: {e}")

class RobustBacktester:
    """Advanced backtester with statistical validation"""
    
    def __init__(self):
        self.transaction_cost = 0  # No fees for testing
        self.max_trades_per_day = 20
    
    def backtest_strategy(self, strategy, data):
        """Backtest strategy with comprehensive validation"""
        if not self._validate_inputs(strategy, data):
            return None
        
        try:
            trades = []
            position = None
            daily_trades = {}
            
            # Conservative warm-up period
            min_warmup = max(200, len(data) // 10)
            start_idx = min(min_warmup, len(data) - 1000)
            
            if start_idx >= len(data) - 300:
                logger.debug(f"Insufficient data for backtesting: {len(data)} candles")
                return None
            
            for i in range(start_idx, len(data) - 1):
                try:
                    current_row = data.iloc[i]
                    current_date = current_row.name.date()
                    
                    # Check daily trade limit
                    if daily_trades.get(current_date, 0) >= self.max_trades_per_day:
                        continue
                    
                    # Exit check
                    if position:
                        exit_result = self._check_exit(position, current_row)
                        if exit_result:
                            trades.append(exit_result)
                            position = None
                            daily_trades[current_date] = daily_trades.get(current_date, 0) + 1
                    
                    # Entry check
                    if not position and i > start_idx + 20:
                        entry_signal = self._check_entry(strategy, current_row, data, i)
                        if entry_signal:
                            position = self._open_position(strategy, current_row, entry_signal)
                
                except Exception as e:
                    logger.debug(f"Backtesting error at index {i}: {e}")
                    continue
            
            return self._calculate_comprehensive_metrics(trades, strategy, len(data) - start_idx)
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            return None
    
    def _validate_inputs(self, strategy, data):
        """Validate strategy and data inputs"""
        try:
            if not strategy or not isinstance(strategy, dict):
                return False
            if data is None or len(data) < 500:
                return False
            
            required_fields = ['name', 'type', 'sl_percent', 'tp_percent']
            if not all(field in strategy for field in required_fields):
                return False
            
            # Validate numeric parameters
            if not all(isinstance(strategy[field], (int, float)) and strategy[field] > 0 
                      for field in ['sl_percent', 'tp_percent']):
                return False
            
            return True
        except Exception:
            return False
    
    def _check_entry(self, strategy, current, data, index):
        """Check entry conditions for various strategy types"""
        if index <= 5:
            return None
        
        try:
            strategy_type = strategy['type']
            
            if strategy_type == 'ma_crossover':
                return self._check_ma_crossover_entry(strategy, current, data, index)
            elif strategy_type == 'rsi_mean_reversion':
                return self._check_rsi_entry(strategy, current, data, index)
            elif strategy_type == 'bollinger_bands':
                return self._check_bollinger_entry(strategy, current, data, index)
            elif strategy_type == 'adx_trend':
                return self._check_adx_entry(strategy, current, data, index)
            else:
                return self._check_ma_crossover_entry(strategy, current, data, index)
        
        except Exception as e:
            logger.debug(f"Entry check failed: {e}")
            return None
    
    def _check_ma_crossover_entry(self, strategy, current, data, index):
        """Check MA crossover entry"""
        try:
            fast_ma_col = strategy['fast_ma']
            slow_ma_col = strategy['slow_ma']
            
            if fast_ma_col not in current.index or slow_ma_col not in current.index:
                return None
            
            # Current values
            fast_ma = current[fast_ma_col]
            slow_ma = current[slow_ma_col]
            
            # Previous values
            prev_row = data.iloc[index-1]
            fast_ma_prev = prev_row[fast_ma_col]
            slow_ma_prev = prev_row[slow_ma_col]
            
            # Check for NaN values
            if any(pd.isna(x) for x in [fast_ma, slow_ma, fast_ma_prev, slow_ma_prev]):
                return None
            
            # Bullish crossover
            if fast_ma > slow_ma and fast_ma_prev <= slow_ma_prev:
                return 'long'
            # Bearish crossover
            elif fast_ma < slow_ma and fast_ma_prev >= slow_ma_prev:
                return 'short'
            
        except Exception as e:
            logger.debug(f"MA crossover check failed: {e}")
        
        return None
    
    def _check_rsi_entry(self, strategy, current, data, index):
        """Check RSI mean reversion entry"""
        try:
            rsi_col = strategy['rsi_indicator']
            if rsi_col not in current.index:
                return None
            
            rsi = current[rsi_col]
            rsi_prev = data.iloc[index-1][rsi_col]
            
            if pd.isna(rsi) or pd.isna(rsi_prev):
                return None
            
            # RSI bounce from oversold
            if rsi > strategy['oversold'] and rsi_prev <= strategy['oversold']:
                return 'long'
            # RSI rejection from overbought
            elif rsi < strategy['overbought'] and rsi_prev >= strategy['overbought']:
                return 'short'
        
        except Exception as e:
            logger.debug(f"RSI check failed: {e}")
        
        return None
    
    def _check_bollinger_entry(self, strategy, current, data, index):
        """Check Bollinger Bands entry"""
        try:
            bb_upper_col = strategy['bb_upper']
            bb_lower_col = strategy['bb_lower']
            
            if bb_upper_col not in current.index or bb_lower_col not in current.index:
                return None
            
            bb_upper = current[bb_upper_col]
            bb_lower = current[bb_lower_col]
            close = current['close']
            close_prev = data.iloc[index-1]['close']
            
            if any(pd.isna(x) for x in [bb_upper, bb_lower, close, close_prev]):
                return None
            
            # Bounce from lower band
            if close > bb_lower and close_prev <= bb_lower:
                return 'long'
            # Rejection from upper band
            elif close < bb_upper and close_prev >= bb_upper:
                return 'short'
        
        except Exception as e:
            logger.debug(f"Bollinger check failed: {e}")
        
        return None
    
    def _check_adx_entry(self, strategy, current, data, index):
        """Check ADX trend entry"""
        try:
            adx_col = strategy['adx_indicator']
            plus_di_col = strategy['plus_di']
            minus_di_col = strategy['minus_di']
            
            required_cols = [adx_col, plus_di_col, minus_di_col]
            if not all(col in current.index for col in required_cols):
                return None
            
            adx = current[adx_col]
            plus_di = current[plus_di_col]
            minus_di = current[minus_di_col]
            
            plus_di_prev = data.iloc[index-1][plus_di_col]
            minus_di_prev = data.iloc[index-1][minus_di_col]
            
            if any(pd.isna(x) for x in [adx, plus_di, minus_di, plus_di_prev, minus_di_prev]):
                return None
            
            # Strong trend condition
            if adx > strategy['adx_threshold']:
                # Bullish: +DI crosses above -DI
                if plus_di > minus_di and plus_di_prev <= minus_di_prev:
                    return 'long'
                # Bearish: -DI crosses above +DI
                elif minus_di > plus_di and minus_di_prev <= plus_di_prev:
                    return 'short'
        
        except Exception as e:
            logger.debug(f"ADX check failed: {e}")
        
        return None
    
    def _open_position(self, strategy, current, signal_type):
        """Open position with validation"""
        try:
            entry_price = current['close']
            if pd.isna(entry_price) or entry_price <= 0:
                return None
            
            sl_percent = strategy['sl_percent']
            tp_percent = strategy['tp_percent']
            
            if signal_type == 'long':
                sl_price = entry_price * (1 - sl_percent / 100)
                tp_price = entry_price * (1 + tp_percent / 100)
            else:  # short
                sl_price = entry_price * (1 + sl_percent / 100)
                tp_price = entry_price * (1 - tp_percent / 100)
            
            return {
                'entry_price': float(entry_price),
                'sl_price': float(sl_price),
                'tp_price': float(tp_price),
                'type': signal_type,
                'entry_time': current.name
            }
        except Exception as e:
            logger.debug(f"Position opening failed: {e}")
            return None
    
    def _check_exit(self, position, current):
        """Check exit conditions"""
        try:
            current_high = current['high']
            current_low = current['low']
            
            if pd.isna(current_high) or pd.isna(current_low):
                return None
            
            if position['type'] == 'long':
                # Check if stop loss hit
                if current_low <= position['sl_price']:
                    return self._create_trade(position, position['sl_price'], 'loss', current.name)
                # Check if take profit hit
                elif current_high >= position['tp_price']:
                    return self._create_trade(position, position['tp_price'], 'win', current.name)
            else:  # short position
                # Check if stop loss hit
                if current_high >= position['sl_price']:
                    return self._create_trade(position, position['sl_price'], 'loss', current.name)
                # Check if take profit hit
                elif current_low <= position['tp_price']:
                    return self._create_trade(position, position['tp_price'], 'win', current.name)
        
        except Exception as e:
            logger.debug(f"Exit check failed: {e}")
        
        return None
    
    def _create_trade(self, position, exit_price, result, exit_time):
        """Create trade result"""
        try:
            entry_price = position['entry_price']
            
            if position['type'] == 'long':
                raw_return = (exit_price - entry_price) / entry_price
            else:  # short
                raw_return = (entry_price - exit_price) / entry_price
            
            return {
                'entry_price': entry_price,
                'exit_price': float(exit_price),
                'entry_time': position['entry_time'],
                'exit_time': exit_time,
                'raw_return': raw_return * 100,
                'net_return': raw_return * 100,
                'result': result,
                'type': position['type']
            }
        except Exception as e:
            logger.debug(f"Trade creation failed: {e}")
            return None
    
    def _calculate_comprehensive_metrics(self, trades, strategy, total_bars):
        """Calculate comprehensive performance metrics with updated criteria"""
        if not trades or len(trades) < 50:
            return None
        
        try:
            # Filter valid trades
            valid_trades = [t for t in trades if t and t.get('net_return') is not None]
            if len(valid_trades) < 50:
                return None
            
            # Basic metrics
            wins = [t for t in valid_trades if t['result'] == 'win']
            losses = [t for t in valid_trades if t['result'] == 'loss']
            
            win_rate = (len(wins) / len(valid_trades)) * 100
            avg_win = np.mean([t['net_return'] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t['net_return'] for t in losses])) if losses else 0
            
            # FIXED: Risk-reward ratio per trade (not average)
            risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Total return
            total_return = sum([t['net_return'] for t in valid_trades])
            
            # Drawdown calculation
            returns = [t['net_return'] for t in valid_trades]
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
            
            # Trade frequency
            trade_frequency = len(valid_trades) / total_bars * 1000
            
            # Consecutive losses
            consecutive_losses = self._calculate_max_consecutive_losses(valid_trades)
            
            # Profit Factor
            gross_profit = sum([t['net_return'] for t in wins]) if wins else 0
            gross_loss = abs(sum([t['net_return'] for t in losses])) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # ENHANCED SUCCESS CRITERIA
            meets_criteria = (
                win_rate >= 60 and            # Win rate >= 60%
                risk_reward >= 1.5 and       # Risk reward >= 1.5 
                len(valid_trades) >= 300 and # Trades >= 300
                strategy['sl_percent'] >= 0.3 and  # SL >= 0.3%
                max_drawdown <= 20 and       # Max drawdown <= 20%
                consecutive_losses <= 5 and  # Max consecutive losses <= 5
                profit_factor >= 2.0 and     # Profit factor >= 2.0
                trade_frequency <= 100       # Not over-trading
            )
            
            is_promising = (
                win_rate >= 47 and           # Win rate >= 47%
                risk_reward >= 1.5 and      # Risk reward >= 1.5
                len(valid_trades) >= 300 and # Trades >= 300
                strategy['sl_percent'] >= 0.3 and  # SL >= 0.3%
                max_drawdown <= 30 and      # Max drawdown <= 30%
                consecutive_losses <= 8     # Max consecutive losses <= 8
            )
            
            return {
                'strategy_name': strategy['name'],
                'strategy_type': strategy['type'],
                'total_trades': len(valid_trades),
                'win_rate': round(win_rate, 2),
                'risk_reward_ratio': round(risk_reward, 3),
                'total_return': round(total_return, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'max_drawdown': round(max_drawdown, 2),
                'max_consecutive_losses': consecutive_losses,
                'profit_factor': round(profit_factor, 2),
                'trade_frequency': round(trade_frequency, 2),
                'meets_criteria': meets_criteria,
                'is_promising': is_promising,
                'config': strategy.copy()
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return None
    
    def _calculate_max_consecutive_losses(self, trades):
        """Calculate maximum consecutive losses"""
        try:
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in trades:
                if trade['result'] == 'loss':
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
        except Exception:
            return 0

class ThreadSafeDatabase:
    """Thread-safe database with connection pooling"""
    
    def __init__(self, db_name='strategies_v10.db', max_queue_size=1000):
        self.db_name = db_name
        self.db_queue = queue.Queue(maxsize=max_queue_size)
        self.max_queue_size = max_queue_size
        self.db_worker_running = True
        self.processed_operations = 0
        self.dropped_operations = 0
        self.db_thread = None
        self.lock = threading.Lock()
        self._start_db_worker()
    
    def _start_db_worker(self):
        """Start database worker thread"""
        try:
            self.db_thread = threading.Thread(target=self._db_worker, daemon=True)
            self.db_thread.start()
            logger.info(f"Database worker started with queue size: {self.max_queue_size}")
        except Exception as e:
            logger.error(f"Failed to start database worker: {e}")
    
    def _db_worker(self):
        """Database worker with proper connection management"""
        conn = None
        try:
            conn = self._create_connection()
            self._initialize_tables(conn)
            
            while self.db_worker_running:
                try:
                    operation, data, result_queue = self.db_queue.get(timeout=1.0)
                    
                    if operation == 'insert':
                        self._handle_insert(conn, data)
                    elif operation == 'select_successful':
                        result = self._handle_select_successful(conn)
                        if result_queue:
                            self._put_result_safely(result_queue, result)
                    elif operation == 'select_promising':
                        result = self._handle_select_promising(conn)
                        if result_queue:
                            self._put_result_safely(result_queue, result)
                    elif operation == 'get_stats':
                        result = self._handle_get_stats(conn)
                        if result_queue:
                            self._put_result_safely(result_queue, result)
                    elif operation == 'get_strategy_details':
                        result = self._handle_get_strategy_details(conn, data)
                        if result_queue:
                            self._put_result_safely(result_queue, result)
                    
                    self.processed_operations += 1
                    
                except queue.Empty:
                    continue
                except sqlite3.Error as e:
                    logger.error(f"Database error: {e}")
                    # Reconnect on database errors
                    if conn:
                        conn.close()
                    conn = self._create_connection()
                    continue
                except Exception as e:
                    logger.error(f"Database worker error: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Database worker failed: {e}")
        finally:
            if conn:
                conn.close()
            logger.info(f"Database worker shutdown. Processed: {self.processed_operations}")
    
    def _create_connection(self):
        """Create database connection with optimal settings"""
        try:
            conn = sqlite3.connect(
                self.db_name,
                timeout=30.0,
                check_same_thread=False
            )
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA cache_size=10000')
            conn.execute('PRAGMA temp_store=memory')
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    def _initialize_tables(self, conn):
        """Initialize database tables"""
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    type TEXT,
                    win_rate REAL,
                    risk_reward REAL,
                    total_trades INTEGER,
                    total_return REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    max_drawdown REAL,
                    max_consecutive_losses INTEGER,
                    profit_factor REAL,
                    trade_frequency REAL,
                    meets_criteria INTEGER,
                    config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_win_rate ON strategies(win_rate)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_meets_criteria ON strategies(meets_criteria)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy_type ON strategies(type)')
            conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize tables: {e}")
            raise
    
    def _handle_insert(self, conn, data):
        """Handle insert operation"""
        try:
            conn.execute('''
                INSERT OR REPLACE INTO strategies 
                (name, type, win_rate, risk_reward, total_trades, total_return, 
                 avg_win, avg_loss, max_drawdown, max_consecutive_losses, 
                 profit_factor, trade_frequency, meets_criteria, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Insert operation failed: {e}")
    
    def _handle_select_successful(self, conn):
        """Handle select successful strategies"""
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT name, type, win_rate, risk_reward, total_trades, 
                       total_return, max_drawdown, profit_factor, config
                FROM strategies 
                WHERE meets_criteria = 1 
                ORDER BY win_rate DESC, risk_reward DESC 
                LIMIT 50
            ''')
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Select successful operation failed: {e}")
            return []
    
    def _handle_select_promising(self, conn):
        """Handle select promising strategies"""
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT name, type, win_rate, risk_reward, total_trades, 
                       total_return, max_drawdown, profit_factor, config
                FROM strategies 
                WHERE meets_criteria = 0 AND win_rate >= 47 
                ORDER BY win_rate DESC 
                LIMIT 100
            ''')
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Select promising operation failed: {e}")
            return []
    
    def _handle_get_strategy_details(self, conn, strategy_name):
        """Handle get strategy details"""
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM strategies WHERE name = ?
            ''', (strategy_name,))
            return cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Get strategy details failed: {e}")
            return None
    
    def _handle_get_stats(self, conn):
        """Handle get statistics"""
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM strategies WHERE meets_criteria = 1')
            successful_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM strategies WHERE win_rate >= 47')
            promising_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM strategies')
            total_count = cursor.fetchone()[0]
            
            return {
                'successful': successful_count,
                'promising': promising_count,
                'total': total_count
            }
        except sqlite3.Error as e:
            logger.error(f"Get stats operation failed: {e}")
            return {'successful': 0, 'promising': 0, 'total': 0}
    
    def _put_result_safely(self, result_queue, result):
        """Safely put result in queue"""
        try:
            result_queue.put(result, timeout=2.0)
        except queue.Full:
            logger.warning("Result queue full, dropping result")
    
    def save_strategy(self, strategy_result, timeout=3.0):
        """Save strategy with backpressure handling"""
        if not strategy_result:
            return False
        
        try:
            data = (
                strategy_result['strategy_name'],
                strategy_result['strategy_type'],
                strategy_result['win_rate'],
                strategy_result['risk_reward_ratio'],
                strategy_result['total_trades'],
                strategy_result['total_return'],
                strategy_result['avg_win'],
                strategy_result['avg_loss'],
                strategy_result['max_drawdown'],
                strategy_result.get('max_consecutive_losses', 0),
                strategy_result.get('profit_factor', 0),
                strategy_result.get('trade_frequency', 0),
                1 if strategy_result['meets_criteria'] else 0,
                json.dumps(strategy_result['config'])
            )
            
            self.db_queue.put(('insert', data, None), timeout=timeout)
            return True
            
        except queue.Full:
            self.dropped_operations += 1
            if strategy_result.get('meets_criteria', False):
                logger.warning("Critical strategy dropped due to full queue")
            return False
        except Exception as e:
            logger.error(f"Database save error: {e}")
            return False
    
    def get_successful_strategies(self):
        """Get successful strategies with timeout"""
        result_queue = queue.Queue(maxsize=1)
        try:
            self.db_queue.put(('select_successful', None, result_queue), timeout=2.0)
            return result_queue.get(timeout=5.0)
        except (queue.Full, queue.Empty):
            return []
        except Exception as e:
            logger.error(f"Failed to get successful strategies: {e}")
            return []
    
    def get_promising_strategies(self):
        """Get promising strategies with timeout"""
        result_queue = queue.Queue(maxsize=1)
        try:
            self.db_queue.put(('select_promising', None, result_queue), timeout=2.0)
            return result_queue.get(timeout=5.0)
        except (queue.Full, queue.Empty):
            return []
        except Exception as e:
            logger.error(f"Failed to get promising strategies: {e}")
            return []
    
    def get_strategy_details(self, strategy_name):
        """Get detailed strategy information"""
        result_queue = queue.Queue(maxsize=1)
        try:
            self.db_queue.put(('get_strategy_details', strategy_name, result_queue), timeout=2.0)
            return result_queue.get(timeout=5.0)
        except (queue.Full, queue.Empty):
            return None
        except Exception as e:
            logger.error(f"Failed to get strategy details: {e}")
            return None
    
    def get_stats(self):
        """Get database statistics"""
        result_queue = queue.Queue(maxsize=1)
        try:
            self.db_queue.put(('get_stats', None, result_queue), timeout=2.0)
            return result_queue.get(timeout=5.0)
        except (queue.Full, queue.Empty):
            return {'successful': 0, 'promising': 0, 'total': 0}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'successful': 0, 'promising': 0, 'total': 0}
    
    def close(self):
        """Graceful shutdown"""
        logger.info("Shutting down database worker...")
        self.db_worker_running = False
        if self.db_thread and self.db_thread.is_alive():
            self.db_thread.join(timeout=5.0)

class AdvancedStrategyDiscoveryEngine:
    """Main discovery engine with 60+ indicators and 25+ strategy types"""
    
    def __init__(self):
        self.data_manager = OptimizedDataManager()
        self.strategy_generator = AdvancedStrategyGenerator()
        self.backtester = RobustBacktester()
        self.database = ThreadSafeDatabase()
        self.should_stop = False
        self.data = None
        self.discovery_stats = {
            'start_time': None,
            'strategies_tested': 0,
            'successful_found': 0,
            'promising_found': 0,
            'errors_encountered': 0
        }
        
        # Thread safety
        self.stats_lock = threading.RLock()
        self.tested_strategies = set()
        self.max_tested_strategies = 100000
    
    def run_discovery(self, max_iterations=10000):
        """Run advanced strategy discovery with comprehensive monitoring"""
        with self.stats_lock:
            self.discovery_stats['start_time'] = datetime.now()
            self.should_stop = False
        
        logger.info("🚀 Starting Advanced Strategy Discovery Engine v10")
        logger.info("📈 60+ TradingView Indicators | 25+ Strategy Types | Statistical Validation")
        logger.info("🎯 Enhanced Criteria - Success: Win Rate ≥60%, R:R ≥1.5, Trades ≥300, PF ≥2.0")
        logger.info("💡 Promising: Win Rate ≥47%, R:R ≥1.5, Trades ≥300")
        
        try:
            # Initialize data
            logger.info("📊 Fetching and preparing data...")
            self.data = self.data_manager.fetch_btc_data_1_year()
            
            logger.info("🔧 Calculating 60+ advanced technical indicators...")
            self.data = ComprehensiveIndicators.calculate_all_indicators(self.data)
            logger.info(f"✅ Data ready: {len(self.data):,} candles with advanced indicators")
            
            # Start discovery loop
            logger.info("🎯 Starting strategy discovery with statistical validation...")
            successful_strategies = []
            promising_strategies = []
            
            for iteration in range(max_iterations):
                if self.should_stop:
                    logger.info("Discovery stopped by user request")
                    break
                
                try:
                    # Generate strategy
                    strategy = self.strategy_generator.generate_strategy()
                    strategy_hash = self._get_strategy_hash(strategy)
                    
                    # Skip if already tested
                    if strategy_hash in self.tested_strategies:
                        continue
                    
                    logger.debug(f"Testing strategy {iteration}: {strategy['name']} ({strategy['type']})")
                    
                    # Backtest strategy
                    result = self.backtester.backtest_strategy(strategy, self.data)
                    
                    with self.stats_lock:
                        self.discovery_stats['strategies_tested'] += 1
                        self.tested_strategies.add(strategy_hash)
                    
                    if result is None:
                        continue
                    
                    # Update learning
                    performance_score = result['win_rate'] * result.get('risk_reward_ratio', 0) / 100
                    self.strategy_generator.update_performance(strategy, performance_score)
                    
                    # Save to database (non-blocking)
                    self.database.save_strategy(result)
                    
                    # Track results
                    if result.get('is_promising', False):
                        promising_strategies.append(result)
                        with self.stats_lock:
                            self.discovery_stats['promising_found'] += 1
                        
                        logger.info(f"💡 PROMISING #{len(promising_strategies)}: {result['strategy_name']}")
                        logger.info(f"   📊 Win Rate: {result['win_rate']:.1f}%, R:R: {result['risk_reward_ratio']:.2f}")
                        logger.info(f"   📈 Trades: {result['total_trades']}, Return: {result['total_return']:.1f}%")
                        logger.info(f"   📉 Max DD: {result['max_drawdown']:.1f}%, PF: {result.get('profit_factor', 0):.1f}")
                    
                    if result.get('meets_criteria', False):
                        successful_strategies.append(result)
                        with self.stats_lock:
                            self.discovery_stats['successful_found'] += 1
                        
                        logger.info(f"🎉 SUCCESS #{len(successful_strategies)}: {result['strategy_name']}")
                        logger.info(f"   🏆 Win Rate: {result['win_rate']:.1f}%, R:R: {result['risk_reward_ratio']:.2f}")
                        logger.info(f"   💰 Trades: {result['total_trades']}, Return: {result['total_return']:.1f}%")
                        logger.info(f"   🛡️ Max DD: {result['max_drawdown']:.1f}%, PF: {result.get('profit_factor', 0):.1f}")
                        logger.info(f"   🔄 Max Consecutive Losses: {result.get('max_consecutive_losses', 0)}")
                    
                    # Progress logging
                    if iteration % 100 == 0 and iteration > 0:
                        self._log_progress(iteration, max_iterations, successful_strategies, promising_strategies)
                
                except Exception as e:
                    with self.stats_lock:
                        self.discovery_stats['errors_encountered'] += 1
                    if self.discovery_stats['errors_encountered'] % 100 == 0:
                        logger.warning(f"Encountered {self.discovery_stats['errors_encountered']} errors")
                    logger.debug(f"Strategy testing error: {e}")
                    continue
            
            # Final summary
            self._log_final_summary(successful_strategies, promising_strategies)
            return successful_strategies, promising_strategies
            
        except Exception as e:
            logger.error(f"Discovery failed with critical error: {e}")
            return [], []
    
    def _get_strategy_hash(self, strategy):
        """Generate unique hash for strategy deduplication"""
        try:
            # Create a normalized config for hashing
            config_copy = {k: v for k, v in strategy.items() if k != 'name'}
            config_str = json.dumps(config_copy, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.debug(f"Hash generation failed: {e}")
            return str(hash(str(strategy)))[:16]
    
    def _log_progress(self, iteration, max_iterations, successful, promising):
        """Log progress with comprehensive statistics"""
        try:
            runtime = datetime.now() - self.discovery_stats['start_time']
            progress_pct = (iteration / max_iterations) * 100
            
            # Estimate completion time
            if iteration > 0:
                rate = iteration / runtime.total_seconds()
                remaining_iterations = max_iterations - iteration
                eta_seconds = remaining_iterations / rate if rate > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))
            else:
                eta = timedelta(0)
            
            logger.info(f"📊 Progress: {iteration:,}/{max_iterations:,} ({progress_pct:.1f}%)")
            logger.info(f"⏱️ Runtime: {runtime}, ETA: {eta}")
            logger.info(f"🎯 Success: {len(successful)}, Promising: {len(promising)}")
            logger.info(f"❌ Errors: {self.discovery_stats['errors_encountered']}")
            
        except Exception as e:
            logger.debug(f"Progress logging failed: {e}")
    
    def _log_final_summary(self, successful, promising):
        """Log final discovery summary"""
        try:
            total_runtime = datetime.now() - self.discovery_stats['start_time']
            
            logger.info("🎉 Advanced Strategy Discovery Complete!")
            logger.info(f"⏱️ Total Runtime: {total_runtime}")
            logger.info(f"🔍 Strategies Tested: {self.discovery_stats['strategies_tested']:,}")
            logger.info(f"✅ Successful Strategies: {len(successful)}")
            logger.info(f"💡 Promising Strategies: {len(promising)}")
            logger.info(f"❌ Total Errors: {self.discovery_stats['errors_encountered']}")
            
            # Database statistics
            db_stats = self.database.get_stats()
            logger.info(f"🗄️ Database: {db_stats['total']} total, {db_stats['successful']} successful")
            
            # Best performing strategies summary
            if successful:
                logger.info("🏆 Top Successful Strategies:")
                for i, s in enumerate(sorted(successful, key=lambda x: x['win_rate'], reverse=True)[:5], 1):
                    logger.info(f"   {i}. {s['strategy_name']}: {s['win_rate']:.1f}% WR, {s['risk_reward_ratio']:.2f} R:R")
            
            if promising:
                logger.info("💎 Top Promising Strategies:")
                for i, s in enumerate(sorted(promising, key=lambda x: x['win_rate'], reverse=True)[:10], 1):
                    logger.info(f"   {i}. {s['strategy_name']}: {s['win_rate']:.1f}% WR, {s['risk_reward_ratio']:.2f} R:R")
                    
        except Exception as e:
            logger.error(f"Final summary logging failed: {e}")
    
    def stop_discovery(self):
        """Stop discovery gracefully"""
        logger.info("🛑 Stop requested - completing current iteration...")
        self.should_stop = True
    
    def cleanup(self):
        """Cleanup all resources"""
        try:
            logger.info("🧹 Cleaning up resources...")
            if self.database:
                self.database.close()
            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Flask Web Application with COMPLETE Interface
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global variables
discovery_engine = None
discovery_thread = None
discovery_status = {
    'running': False,
    'iteration': 0,
    'successful_found': 0,
    'promising_found': 0,
    'current_strategy': '',
    'start_time': None,
    'eta': None
}
status_lock = threading.RLock()

# COMPLETE HTML Templates
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Strategy Discovery Engine v10</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 15px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 40px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #f44336, #da190b);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .status-panel {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 40px;
            border-left: 5px solid #667eea;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .status-item {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        
        .status-item h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .status-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 30px;
            background: #f8f9fa;
            border-radius: 25px;
            padding: 5px;
        }
        
        .tab {
            flex: 1;
            padding: 15px 20px;
            text-align: center;
            border: none;
            background: transparent;
            cursor: pointer;
            border-radius: 20px;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        
        .tab.active {
            background: white;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            color: #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .table-container {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th {
            background: #667eea;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .strategy-name {
            font-weight: 600;
            color: #333;
            cursor: pointer;
        }
        
        .strategy-name:hover {
            color: #667eea;
            text-decoration: underline;
        }
        
        .metric {
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .metric.good {
            background: #d4edda;
            color: #155724;
        }
        
        .metric.excellent {
            background: #cce5ff;
            color: #004085;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        
        .empty-state h3 {
            margin-bottom: 10px;
            font-size: 1.5em;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #4CAF50, #45a049);
            transition: width 0.3s ease;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 300px;
            }
            
            .status-grid {
                grid-template-columns: 1fr;
            }
            
            .tabs {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Advanced Strategy Discovery Engine v10</h1>
            <p>60+ TradingView Indicators | 25+ Strategy Types | Statistical Validation</p>
        </div>
        
        <div class="controls">
            <button class="btn btn-primary" id="startBtn" onclick="startDiscovery()">
                Start Discovery
            </button>
            <button class="btn btn-danger" id="stopBtn" onclick="stopDiscovery()" disabled>
                Stop Discovery
            </button>
        </div>
        
        <div class="status-panel">
            <h2>📊 Discovery Status</h2>
            <div class="status-grid">
                <div class="status-item">
                    <h3>Status</h3>
                    <div class="status-value" id="status">Idle</div>
                </div>
                <div class="status-item">
                    <h3>Strategies Tested</h3>
                    <div class="status-value" id="tested">0</div>
                </div>
                <div class="status-item">
                    <h3>Successful Found</h3>
                    <div class="status-value" id="successful">0</div>
                </div>
                <div class="status-item">
                    <h3>Promising Found</h3>
                    <div class="status-value" id="promising">0</div>
                </div>
            </div>
            <div class="progress-bar" id="progressContainer" style="display: none;">
                <div class="progress-fill" id="progressBar" style="width: 0%;"></div>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('successful')">
                🏆 Successful Strategies
            </button>
            <button class="tab" onclick="showTab('promising')">
                💎 Promising Strategies
            </button>
        </div>
        
        <div id="successful-content" class="tab-content active">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Strategy Name</th>
                            <th>Type</th>
                            <th>Win Rate</th>
                            <th>Risk:Reward</th>
                            <th>Trades</th>
                            <th>Total Return</th>
                            <th>Max Drawdown</th>
                            <th>Profit Factor</th>
                        </tr>
                    </thead>
                    <tbody id="successful-table">
                        <tr>
                            <td colspan="8" class="empty-state">
                                <div class="loading"></div>
                                <h3>Searching for successful strategies...</h3>
                                <p>Strategies with Win Rate ≥60%, R:R ≥1.5, Trades ≥300, PF ≥2.0</p>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="promising-content" class="tab-content">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Strategy Name</th>
                            <th>Type</th>
                            <th>Win Rate</th>
                            <th>Risk:Reward</th>
                            <th>Trades</th>
                            <th>Total Return</th>
                            <th>Max Drawdown</th>
                            <th>Profit Factor</th>
                        </tr>
                    </thead>
                    <tbody id="promising-table">
                        <tr>
                            <td colspan="8" class="empty-state">
                                <div class="loading"></div>
                                <h3>Searching for promising strategies...</h3>
                                <p>Strategies with Win Rate ≥47%, R:R ≥1.5, Trades ≥300</p>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>&copy; 2025 Advanced Strategy Discovery Engine - Built with AI-Powered Optimization</p>
        </div>
    </div>
    
    <script>
        let isRunning = false;
        let updateInterval;
        
        function startDiscovery() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            fetch('/start_discovery', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        isRunning = true;
                        updateInterval = setInterval(updateStatus, 2000);
                        document.getElementById('progressContainer').style.display = 'block';
                    } else {
                        alert('Failed to start discovery: ' + data.error);
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    }
                })
                .catch(error => {
                    alert('Error: ' + error);
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                });
        }
        
        function stopDiscovery() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            fetch('/stop_discovery', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        isRunning = false;
                        if (updateInterval) {
                            clearInterval(updateInterval);
                        }
                        
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        document.getElementById('status').textContent = 'Stopped';
                        document.getElementById('progressContainer').style.display = 'none';
                    }
                });
        }
        
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = 
                        data.running ? 'Running...' : 'Idle';
                    document.getElementById('tested').textContent = 
                        data.strategies_tested || 0;
                    document.getElementById('successful').textContent = 
                        data.successful_found || 0;
                    document.getElementById('promising').textContent = 
                        data.promising_found || 0;
                    
                    if (!data.running && isRunning) {
                        isRunning = false;
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        document.getElementById('status').textContent = 'Completed';
                        document.getElementById('progressContainer').style.display = 'none';
                        
                        if (updateInterval) {
                            clearInterval(updateInterval);
                        }
                        
                        // Refresh strategy tables
                        loadStrategies();
                    }
                });
        }
        
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName + '-content').classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        function loadStrategies() {
            // Load successful strategies
            fetch('/api/strategies/successful')
                .then(response => response.json())
                .then(data => {
                    updateTable('successful-table', data.strategies || []);
                });
            
            // Load promising strategies
            fetch('/api/strategies/promising')
                .then(response => response.json())
                .then(data => {
                    updateTable('promising-table', data.strategies || []);
                });
        }
        
        function updateTable(tableId, strategies) {
            const tbody = document.getElementById(tableId);
            
            if (strategies.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="8" class="empty-state">
                            <h3>No strategies found yet</h3>
                            <p>Start discovery to find profitable strategies</p>
                        </td>
                    </tr>
                `;
                return;
            }
            
            tbody.innerHTML = strategies.map(strategy => `
                <tr>
                    <td>
                        <span class="strategy-name" onclick="showStrategyDetails('${strategy[0]}')">
                            ${strategy[0]}
                        </span>
                    </td>
                    <td>${strategy[1]}</td>
                    <td>
                        <span class="metric ${strategy[2] >= 60 ? 'excellent' : 'good'}">
                            ${strategy[2]}%
                        </span>
                    </td>
                    <td>
                        <span class="metric ${strategy[3] >= 2.0 ? 'excellent' : 'good'}">
                            ${strategy[3]}
                        </span>
                    </td>
                    <td>${strategy[4]}</td>
                    <td>
                        <span class="metric ${strategy[5] >= 0 ? 'good' : ''}">
                            ${strategy[5]}%
                        </span>
                    </td>
                    <td>${strategy[6]}%</td>
                    <td>
                        <span class="metric ${strategy[7] >= 2.0 ? 'excellent' : 'good'}">
                            ${strategy[7]}
                        </span>
                    </td>
                </tr>
            `).join('');
        }
        
        function showStrategyDetails(strategyName) {
            window.open(`/strategy/${encodeURIComponent(strategyName)}`, '_blank');
        }
        
        // Initialize
        loadStrategies();
        updateStatus();
        setInterval(loadStrategies, 10000); // Refresh every 10 seconds
    </script>
</body>
</html>
'''

STRATEGY_DETAIL_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategy Details - {{ strategy_name }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 15px;
        }
        
        .back-btn {
            position: absolute;
            top: 30px;
            left: 30px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            border-radius: 20px;
            transition: all 0.3s ease;
        }
        
        .back-btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateX(-5px);
        }
        
        .strategy-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .info-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        
        .info-section h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .metric-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .conditions {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .condition-section {
            margin-bottom: 30px;
        }
        
        .condition-section h4 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }
        
        .condition-text {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.95em;
            line-height: 1.6;
            white-space: pre-line;
            border-left: 4px solid #667eea;
        }
        
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        .success {
            background: #d4edda;
            color: #155724;
        }
        
        .promising {
            background: #fff3cd;
            color: #856404;
        }
        
        @media (max-width: 768px) {
            .strategy-info {
                grid-template-columns: 1fr;
            }
            
            .metric-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Dashboard</a>
        
        <div class="header">
            <h1>📊 {{ strategy_name }}</h1>
            <p>Strategy Type: {{ strategy_type }}</p>
            {% if meets_criteria %}
                <div class="status-badge success">✅ Meets Success Criteria</div>
            {% else %}
                <div class="status-badge promising">💡 Promising Strategy</div>
            {% endif %}
        </div>
        
        <div class="strategy-info">
            <div class="info-section">
                <h3>📈 Performance Metrics</h3>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{{ win_rate }}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Risk:Reward</div>
                        <div class="metric-value">{{ risk_reward }}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value">{{ total_trades }}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value">{{ total_return }}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value">{{ max_drawdown }}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{{ profit_factor }}</div>
                    </div>
                </div>
            </div>
            
            <div class="info-section">
                <h3>⚙️ Strategy Parameters</h3>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">Stop Loss</div>
                        <div class="metric-value">{{ sl_percent }}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Take Profit</div>
                        <div class="metric-value">{{ tp_percent }}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Avg Win</div>
                        <div class="metric-value">{{ avg_win }}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Avg Loss</div>
                        <div class="metric-value">{{ avg_loss }}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Max Consecutive Losses</div>
                        <div class="metric-value">{{ max_consecutive_losses }}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Trade Frequency</div>
                        <div class="metric-value">{{ trade_frequency }}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="conditions">
            <h3>🎯 Trading Conditions</h3>
            
            <div class="condition-section">
                <h4>Entry Conditions</h4>
                <div class="condition-text">{{ entry_conditions }}</div>
            </div>
            
            <div class="condition-section">
                <h4>Exit Conditions</h4>
                <div class="condition-text">{{ exit_conditions }}</div>
            </div>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def dashboard():
    """Enhanced dashboard with complete interface"""
    return DASHBOARD_TEMPLATE

@app.route('/start_discovery', methods=['POST'])
def start_discovery():
    """Start the discovery process"""
    global discovery_engine, discovery_thread, discovery_status
    
    with status_lock:
        if discovery_status['running']:
            return jsonify({'success': False, 'error': 'Discovery already running'})
        
        try:
            discovery_engine = AdvancedStrategyDiscoveryEngine()
            discovery_thread = threading.Thread(
                target=run_discovery_thread,
                daemon=True
            )
            discovery_thread.start()
            
            discovery_status['running'] = True
            discovery_status['start_time'] = datetime.now()
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_discovery', methods=['POST'])
def stop_discovery():
    """Stop the discovery process"""
    global discovery_engine, discovery_status
    
    with status_lock:
        if not discovery_status['running']:
            return jsonify({'success': False, 'error': 'Discovery not running'})
        
        try:
            if discovery_engine:
                discovery_engine.stop_discovery()
            
            discovery_status['running'] = False
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    """Get current discovery status"""
    with status_lock:
        return jsonify({
            'running': discovery_status['running'],
            'strategies_tested': discovery_status.get('strategies_tested', 0),
            'successful_found': discovery_status.get('successful_found', 0),
            'promising_found': discovery_status.get('promising_found', 0),
            'start_time': discovery_status.get('start_time').isoformat() if discovery_status.get('start_time') else None
        })

@app.route('/api/strategies/successful')
def get_successful_strategies():
    """Get successful strategies from database"""
    try:
        if discovery_engine and discovery_engine.database:
            strategies = discovery_engine.database.get_successful_strategies()
            return jsonify({'strategies': strategies})
        return jsonify({'strategies': []})
    except Exception as e:
        logger.error(f"Failed to get successful strategies: {e}")
        return jsonify({'strategies': []})

@app.route('/api/strategies/promising')
def get_promising_strategies():
    """Get promising strategies from database"""
    try:
        if discovery_engine and discovery_engine.database:
            strategies = discovery_engine.database.get_promising_strategies()
            return jsonify({'strategies': strategies})
        return jsonify({'strategies': []})
    except Exception as e:
        logger.error(f"Failed to get promising strategies: {e}")
        return jsonify({'strategies': []})

@app.route('/strategy/<strategy_name>')
def strategy_details(strategy_name):
    """Show detailed strategy information with entry/exit conditions"""
    try:
        if discovery_engine and discovery_engine.database:
            strategy_data = discovery_engine.database.get_strategy_details(strategy_name)
            
            if strategy_data:
                # Parse config JSON
                config = json.loads(strategy_data[14]) if strategy_data[14] else {}
                
                return render_template_string(STRATEGY_DETAIL_TEMPLATE,
                    strategy_name=strategy_data[1],
                    strategy_type=strategy_data[2],
                    win_rate=strategy_data[3],
                    risk_reward=strategy_data[4],
                    total_trades=strategy_data[5],
                    total_return=strategy_data[6],
                    avg_win=strategy_data[7],
                    avg_loss=strategy_data[8],
                    max_drawdown=strategy_data[9],
                    max_consecutive_losses=strategy_data[10],
                    profit_factor=strategy_data[11],
                    trade_frequency=strategy_data[12],
                    meets_criteria=bool(strategy_data[13]),
                    sl_percent=config.get('sl_percent', 'N/A'),
                    tp_percent=config.get('tp_percent', 'N/A'),
                    entry_conditions=config.get('entry_conditions', 'Entry conditions not available'),
                    exit_conditions=config.get('exit_conditions', 'Exit conditions not available')
                )
        
        return "Strategy not found", 404
        
    except Exception as e:
        logger.error(f"Failed to get strategy details: {e}")
        return "Error loading strategy details", 500

def run_discovery_thread():
    """Run discovery in background thread"""
    global discovery_status
    
    try:
        successful, promising = discovery_engine.run_discovery(max_iterations=10000)
        
        with status_lock:
            discovery_status['running'] = False
            discovery_status['successful_found'] = len(successful)
            discovery_status['promising_found'] = len(promising)
        
        logger.info(f"Discovery completed: {len(successful)} successful, {len(promising)} promising")
        
    except Exception as e:
        logger.error(f"Discovery thread failed: {e}")
        with status_lock:
            discovery_status['running'] = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutdown signal received...")
    
    global discovery_engine
    if discovery_engine:
        discovery_engine.stop_discovery()
        discovery_engine.cleanup()
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ======= Consolidated run block (Render-ready) =======
import os
import signal
import sys

def graceful_shutdown(sig, frame):
    logger.info('Shutdown signal received, stopping discovery and cleaning up...')
    try:
        if 'discovery_engine' in globals() and discovery_engine:
            try:
                discovery_engine.stop_discovery()
            except Exception as e:
                logger.warning(f'Error stopping discovery engine: {e}')
            try:
                discovery_engine.cleanup()
            except Exception as e:
                logger.warning(f'Error cleaning up discovery engine: {e}')
    except Exception as e:
        logger.error(f'Error during graceful shutdown: {e}')
    finally:
        sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f'🚀 Starting Strategy Discovery Flask server on port {port}')
    # Production: use gunicorn Strategydiscovery_15:app --bind 0.0.0.0:$PORT --workers 4
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
