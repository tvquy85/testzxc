import pandas as pd
import json
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            
    df_pred = pd.DataFrame(data)
    
    # Load true returns
    df_samples = pd.read_parquet("data/labels/aligned_samples_h1.parquet")
    df = df_pred.merge(df_samples[['sample_id', 'abnormal_return_h1']], on='sample_id', how='inner')
    
    # Sort by date
    df['window_end_date'] = pd.to_datetime(df['window_end_date'])
    df = df.sort_values('window_end_date')

    # Action Rule
    # P_up = mild_up + strong_up
    # P_down = mild_down + strong_down
    # long if P_up > 0.60
    # short if P_down > 0.60
    # hold otherwise
    
    df['p_up'] = df['p_mild_up'] + df['p_strong_up']
    df['p_down'] = df['p_mild_down'] + df['p_strong_down']
    
    actions = []
    for row in df.itertuples():
        if row.p_up > 0.60:
            actions.append(1) # Long
        elif row.p_down > 0.60:
            actions.append(-1) # Short
        else:
            actions.append(0) # Hold
            
    df['action'] = actions
    
    # Calculate returns with 5bps transaction cost
    tc = 0.0005
    trade_returns = []
    for i in range(len(df)):
        action = df.iloc[i]['action']
        ret = df.iloc[i]['abnormal_return_h1']
        
        if action == 1:
            trade_returns.append(ret - tc)
        elif action == -1:
            trade_returns.append(-ret - tc)
        else:
            trade_returns.append(0.0)
            
    df['strategy_return'] = trade_returns
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    
    # Metrics
    total_return = df['cumulative_return'].iloc[-1] - 1
    
    # annualized sharpe (assuming rough daily freq if grouping by date, but since we trade per sample, we calculate Sharpe per trade then annualize assuming ~252 trades per year)
    mean_ret = df['strategy_return'].mean()
    std_ret = df['strategy_return'].std()
    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(252) # rough annualization
    else:
        sharpe = 0.0
        
    # Max drawdown
    roll_max = df['cumulative_return'].cummax()
    drawdown = df['cumulative_return'] / roll_max - 1.0
    max_drawdown = drawdown.min()
    
    # Win rate
    trades = df[df['action'] != 0]
    if len(trades) > 0:
        win_rate = (trades['strategy_return'] > 0).mean()
    else:
        win_rate = 0.0
        
    metrics = {
        "total_trades": int(len(trades)),
        "long_trades": int((df['action'] == 1).sum()),
        "short_trades": int((df['action'] == -1).sum()),
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate)
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(json.dumps(metrics, indent=2))
    print(f"Backtest metrics saved to {args.output}")

if __name__ == "__main__":
    main()
