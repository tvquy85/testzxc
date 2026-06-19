import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os

class StockDataset(Dataset):
    def __init__(self, text_data, price_data, seq_len=10, num_texts=3):
        self.text_data = text_data
        self.price_data = price_data
        self.seq_len = seq_len
        self.num_texts = num_texts
        
    def __len__(self):
        return max(0, len(self.text_data) - self.seq_len)
        
    def __getitem__(self, idx):
        # Lấy dữ liệu văn bản
        texts = self.text_data[idx:idx+self.num_texts]
        # Lấy dữ liệu giá
        prices = self.price_data[idx:idx+self.seq_len]
        # Tính xu hướng giá (1: tăng, 0: giảm)
        target = 1 if prices[-1][0] > prices[-2][0] else 0
        
        return texts, prices, target

def load_acl18_data(data_dir):
    """Tải dữ liệu từ dataset ACL18"""
    # Tải tweets
    tweets_df = pd.read_csv(os.path.join(data_dir, 'acl18/tweets/tweets.csv'))
    # Tải giá
    prices_df = pd.read_csv(os.path.join(data_dir, 'acl18/prices/prices.csv'))
    
    # Xử lý dữ liệu
    text_data = tweets_df['text'].values
    price_data = prices_df[['open', 'high', 'low', 'close']].values
    
    return text_data, price_data

def load_djia_data(data_dir):
    """Tải dữ liệu từ dataset DJIA"""
    # Tải tin tức
    news_df = pd.read_csv(os.path.join(data_dir, 'djia/news/Combined_News_DJIA.csv'))
    # Tải giá
    prices_df = pd.read_csv(os.path.join(data_dir, 'djia/prices/DJIA_table.csv'))
    
    # Xử lý dữ liệu
    text_data = news_df.iloc[:, 1:].values  # Bỏ cột Date
    price_data = prices_df[['Open', 'High', 'Low', 'Close']].values
    
    return text_data, price_data

def prepare_data(data_dir, dataset_name='acl18', batch_size=32):
    """Chuẩn bị dữ liệu cho training"""
    # Tải dữ liệu
    if dataset_name == 'acl18':
        text_data, price_data = load_acl18_data(data_dir)
    else:
        text_data, price_data = load_djia_data(data_dir)
    
    # Khởi tạo tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize văn bản
    tokenized_texts = []
    for text in text_data:
        tokens = tokenizer.encode(text, max_length=10, padding='max_length', truncation=True)
        tokenized_texts.append(tokens)
    
    # Chuyển đổi thành tensor
    text_tensor = torch.tensor(tokenized_texts)
    price_tensor = torch.tensor(price_data, dtype=torch.float32)
    
    # Tạo dataset với seq_len=2, num_texts=1 cho dữ liệu mẫu nhỏ
    dataset = StockDataset(text_tensor, price_tensor, seq_len=2, num_texts=1)
    
    # Tạo dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

if __name__ == "__main__":
    # Test data loading
    data_dir = "data"
    dataloader = prepare_data(data_dir, dataset_name='acl18')
    
    # In ra một batch
    for texts, prices, target in dataloader:
        print("Text shape:", texts.shape)
        print("Price shape:", prices.shape)
        print("Target:", target)
        break 