import pandas as pd
import json
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, brier_score_loss, log_loss, confusion_matrix

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            
        prob_in_bin = in_bin.sum()
        if prob_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prob_in_bin / len(y_true)
            
    return ece

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} predictions.")

    lbl_map = {
        'Strong Down': 0,
        'Mild Down': 1,
        'Neutral': 2,
        'Mild Up': 3,
        'Strong Up': 4
    }
    
    # Prob matrix
    prob_cols = ['p_strong_down', 'p_mild_down', 'p_neutral', 'p_mild_up', 'p_strong_up']
    
    # Fallback to Neutral if pred_label is unknown
    df['pred_label'] = df['pred_label'].apply(lambda x: x if x in lbl_map else 'Neutral')
    df['true_label'] = df['true_label'].apply(lambda x: x if x in lbl_map else 'Neutral')
    
    y_true = df['true_label'].map(lbl_map).values
    y_pred = df['pred_label'].map(lbl_map).values
    
    # Normalize probabilities
    probs = df[prob_cols].values
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-9)
    
    # Hard metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()
    
    # Soft metrics
    try:
        nll = log_loss(y_true, probs, labels=[0, 1, 2, 3, 4])
    except ValueError:
        nll = float('nan')
        
    # Brier Score and ECE (macro averaged over classes)
    brier_scores = []
    eces = []
    for c in range(5):
        y_true_c = (y_true == c).astype(int)
        y_prob_c = probs[:, c]
        brier_scores.append(brier_score_loss(y_true_c, y_prob_c))
        eces.append(expected_calibration_error(y_true_c, y_prob_c))
        
    brier_macro = np.mean(brier_scores)
    ece_macro = np.mean(eces)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "brier_score_macro": float(brier_macro),
        "ece_macro": float(ece_macro),
        "nll": float(nll),
        "confusion_matrix": cm
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {args.output}")

if __name__ == "__main__":
    main()
