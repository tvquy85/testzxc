import math

def score_utility(action: str, abnormal_return: float, transaction_cost: float = 0.001) -> float:
    """
    Computes a normalized utility score based on the predicted action and the actual abnormal return.
    Normalizes the payoff into [0, 1] range using a sigmoid function.
    """
    if not action or not isinstance(action, str):
        payoff = 0.0
    else:
        action = action.lower().strip()
        if action == 'long':
            payoff = abnormal_return - transaction_cost
        elif action == 'short':
            payoff = -abnormal_return - transaction_cost
        else:
            payoff = 0.0
            
    # Normalize using a sigmoid centered at 0 with a scale factor
    # E.g., a 2% return (0.02) * 50 = 1.0 -> sigmoid(1.0) = 0.73
    # A 5% return (0.05) * 50 = 2.5 -> sigmoid(2.5) = 0.92
    scale = 50.0 
    utility = 1.0 / (1.0 + math.exp(-payoff * scale))
    return utility
