import numpy as np

def split_data(data, split_ratio=0.8):
    train_size = int(len(data) * split_ratio)
    return data[:train_size], data[train_size:]
