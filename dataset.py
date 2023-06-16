import torch
import pandas as pd
from torch.utils.data import Dataset

class RateDataset(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv("data/ratings.csv")

    def __getitem__(self, index):
        return self.dataset["userId"][index], self.dataset["movieId"][index], self.dataset["rating"][index]

    def __len__(self):
        return self.dataset.__len__()

