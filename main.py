from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RateDataset
from model import BiasedMatrixFactorization

@dataclass
class Configs:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.01

configs = Configs()

dataset = RateDataset()
user_num = dataset[:][0].max() + 1
item_num = dataset[:][1].max() + 1

model = BiasedMatrixFactorization(user_num, item_num)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SparseAdam(model.parameters(), lr=configs.learning_rate)

train_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)
for epoch in range(configs.epochs):
    loop = tqdm(train_loader)
    for bid, batch in enumerate(loop):
        users, items, ratings = batch[0], batch[1], batch[2]
        ratings = ratings.float()
        # forward pass
        preds = model(users, items)
        loss = criterion(preds, ratings)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/30], Loss: {loss.item():.4f}')