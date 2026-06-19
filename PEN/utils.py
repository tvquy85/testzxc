import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Tuple

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    atr = pd.Series(tr).rolling(period).mean().values
    atr[np.isnan(atr)] = 0  # Replace NaN with 0
    return atr

def detect_trend(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Detect trend using simple moving average"""
    sma = pd.Series(close).rolling(period).mean().values
    trend = np.zeros_like(close)
    trend[close > sma] = 1  # uptrend
    trend[close < sma] = -1  # downtrend
    trend[np.isnan(trend)] = 0  # Replace NaN with 0
    return trend

def prepare_timeframe_data(
    data: pd.DataFrame,
    timeframe: str,
    input_size: int = 5
) -> np.ndarray:
    """Prepare data for a specific timeframe"""
    # Resample data to target timeframe
    resampled = data.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).fillna(method='ffill')  # Forward fill missing values
    
    # Calculate features
    high = resampled['high'].values
    low = resampled['low'].values
    close = resampled['close'].values
    
    atr = calculate_atr(high, low, close)
    trend = detect_trend(close)
    
    # Combine features
    features = np.column_stack([
        resampled['open'].values,
        high,
        low,
        close,
        resampled['volume'].values,
        atr,
        trend
    ])
    
    # Remove any remaining NaN values
    features = np.nan_to_num(features, 0)
    
    return features

def create_sequences(
    data: np.ndarray,
    seq_length: int,
    target_length: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for training"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - target_length + 1):
        seq = data[i:i + seq_length]
        target = np.sign(data[i + seq_length:i + seq_length + target_length, 3] - data[i + seq_length - 1, 3])  # Compare close prices
        target = target[0]  # Get first target only
        
        # Convert target to class index (0: down, 1: sideway, 2: up)
        if target > 0:
            target = 2
        elif target < 0:
            target = 0
        else:
            target = 1
            
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def explain_prediction(
    model_output: Dict[str, torch.Tensor],
    timeframe_names: List[str]
) -> Dict:
    """Generate explanation for model prediction"""
    prediction = model_output['prediction']
    attention_weights = model_output['attention_weights']
    timeframe_vos = model_output['timeframe_vos']
    
    # Get predicted class
    predicted_class = torch.argmax(prediction).item()
    class_names = ['down', 'sideway', 'up']
    
    # Get most important timeframe
    timeframe_importance = attention_weights.squeeze().tolist()
    most_important_timeframe_idx = np.argmax(timeframe_importance)
    most_important_timeframe = timeframe_names[most_important_timeframe_idx]
    
    # Get feature importance for most important timeframe
    feature_importance = timeframe_vos[most_important_timeframe_idx].squeeze().tolist()
    
    explanation = {
        'prediction': class_names[predicted_class],
        'confidence': prediction[0][predicted_class].item(),
        'most_important_timeframe': most_important_timeframe,
        'timeframe_importance': dict(zip(timeframe_names, timeframe_importance)),
        'feature_importance': {
            'trend': feature_importance[0],
            'volatility': feature_importance[1]
        }
    }
    
    return explanation

def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs['prediction'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    } 