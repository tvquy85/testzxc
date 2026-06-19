import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FlowRewardDataset(Dataset):
    def __init__(self, judge_scores_path: str, tech_features_path: str):
        super().__init__()
        
        # Load parquets
        df_judge = pd.read_parquet(judge_scores_path)
        df_tech = pd.read_parquet(tech_features_path)
        
        # Merge on sample_id
        # Note: df_judge has sample_id and candidate_id. df_tech only has sample_id.
        self.df = df_judge.merge(df_tech, on='sample_id', how='inner')
        
        # Define columns
        self.z1_cols = [
            'infer_p_strong_down', 'infer_p_mild_down', 'infer_p_neutral', 
            'infer_p_mild_up', 'infer_p_strong_up'
        ]
        
        self.tech_cols = [
            'ret_1d', 'ret_5d', 'ret_20d', 'volatility_10d', 'volatility_20d', 
            'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_5', 'SMA_20', 'SMA_60', 
            'price_vs_SMA20', 'price_vs_SMA60', 'Bollinger_width_20', 'Bollinger_position_20', 
            'ATR_14', 'volume_zscore_20', 'gap_pct_last_day', 'market_ret_1d', 'market_ret_5d', 
            'market_ret_20d', 'market_vol_20d', 'market_vol_20d_rank', 
            'relative_strength_vs_market_5d', 'relative_strength_vs_market_20d'
        ]
        
        self.grounding_cols = [
            'financial_soundness_score', 'technical_grounding_score', 
            'news_entailment_rate', 'news_contradiction_rate', 
            'utility_score', 'overall_proxy_score'
        ]
        
        # Fill NaNs
        self.df[self.z1_cols] = self.df[self.z1_cols].fillna(0.0)
        self.df[self.tech_cols] = self.df[self.tech_cols].fillna(0.0)
        self.df[self.grounding_cols] = self.df[self.grounding_cols].fillna(0.0)
        
        # Precompute regime one-hot and sigma
        regime_map = {'low_vol': 0, 'normal_vol': 1, 'high_vol': 2}
        sigma_map = {'low_vol': 0.5, 'normal_vol': 1.0, 'high_vol': 1.5}
        
        self.regimes = self.df['regime_label'].map(regime_map).fillna(1).astype(int).values
        self.sigmas = self.df['regime_label'].map(sigma_map).fillna(1.0).astype(float).values
        
        # Create one-hot arrays
        self.regime_one_hot = np.zeros((len(self.df), 3), dtype=np.float32)
        self.regime_one_hot[np.arange(len(self.df)), self.regimes] = 1.0
        
        # Extract to numpy
        self.z1 = self.df[self.z1_cols].values.astype(np.float32)
        
        tech_vals = self.df[self.tech_cols].values.astype(np.float32)
        grounding_vals = self.df[self.grounding_cols].values.astype(np.float32)
        
        # Concat conditions
        self.cond = np.concatenate([tech_vals, self.regime_one_hot, grounding_vals], axis=1)

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        return {
            'z1': torch.tensor(self.z1[idx]),
            'cond': torch.tensor(self.cond[idx]),
            'sigma': torch.tensor(self.sigmas[idx], dtype=torch.float32),
            'sample_id': self.df.iloc[idx]['sample_id'],
            'candidate_id': self.df.iloc[idx]['candidate_id']
        }
