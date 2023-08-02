import pandas as pd
from torch.utils.data import Dataset
from collections import Counter

class RateDataset(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv("data/train_data_200_100.csv")

    def __getitem__(self, index):
        return self.dataset["USER_ID"][index], self.dataset["PRD_SEQ"][index], self.dataset["STAR"][index]

    def __len__(self):
        return self.dataset.__len__()


class SDRRateDataset(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv("data/sdp_star_rating.csv")
        self.vocab = Counter(self.dataset["USER_ID"])
        self.vocab = sorted(self.vocab, key=self.vocab.get, reverse=True)
        self.word2idx = {word: ind for ind, word in enumerate(self.vocab)}
        self.idx2word = {ind: word for ind, word in enumerate(self.vocab)}
        self.user_size = len(self.vocab)
        self.encoded_user = [self.word2idx[word] for word in self.dataset["USER_ID"]]

    def __getitem__(self, index):
        return self.encoded_user[index], \
            self.dataset["PRD_SEQ"][index], \
            self.dataset["STAR"][index]

    def __len__(self):
        return self.dataset.__len__()
