import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, brier_score_loss

def compute_metrics(y_true, y_pred, y_prob=None, labels=None):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    
    if y_prob is not None and labels is not None:
        try:
            # y_prob should be shape (n_samples, n_classes) matching labels order
            # brier score for multiclass: 1/N sum_{i=1}^N sum_{c=1}^C (y_ic - p_ic)^2
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=labels)
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            brier = np.mean(np.sum((y_prob - y_true_bin)**2, axis=1))
            metrics["brier_score_multiclass"] = brier
            
            # Simple ECE (Expected Calibration Error) approximation
            n_bins = 10
            confidences = np.max(y_prob, axis=1)
            predictions = np.argmax(y_prob, axis=1)
            # map string labels back to indices for accuracy checking if y_pred is not indices
            # we'll skip ECE for now unless requested strictly
        except Exception as e:
            metrics["brier_score_multiclass"] = None
            print("Brier score error:", e)
            
    return metrics
