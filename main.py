from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import time
import os
from pathlib import Path
import pandas as pd

from dataset import SDRRateDataset
from model import BiasedMatrixFactorization


@dataclass
class Configs:
    # epochs: int = 1000
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01


configs = Configs()

dataset = SDRRateDataset()
user_num = dataset.user_size + 1
item_num = dataset[:][1].max() + 1

model = BiasedMatrixFactorization(user_num, item_num)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SparseAdam(
    model.parameters(), lr=configs.learning_rate)

dir = Path("logs")
file = f"train-{int(time.time())}.log"
model_path = Path("models")
model_file = model_path / "model-sports.pt"

f = open(dir / file, "w")

if os.path.exists(model_file) is False:
    train_loader = DataLoader(
        dataset, batch_size=configs.batch_size, shuffle=True)
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

        print(
            f'Epoch [{epoch + 1}/{configs.epochs}], Loss: {loss.item():.4f}', file=f)

    torch.save(model.state_dict(), model_path / "model.pt")

else:
    model.load_state_dict(torch.load(model_path / "model.pt"))


for i in range(1, 20):
    result: torch.Tensor = model(torch.Tensor([4]).to(
        torch.int64), torch.Tensor([i]).to(torch.int64))
    print(result)
    # print()


def select(item_id: int):
    users = np.arange(1, user_num)
    results = []
    for user_id in users:
        result: torch.Tensor = model(torch.Tensor([user_id]).to(
            torch.int64), torch.Tensor([item_id]).to(torch.int64))
        results.append(result.detach().numpy())
    print(results)


select(5)


user_weights = model.get_parameter("user_factors.weight")
item_weights = model.get_parameter("item_factors.weight")


def consine_similarity(A, B):
    return np.dot(A, B)/(norm(A)*norm(B))


def item_similarity(selected_item_id: int):

    similarity_scores = []

    for item_id in range(selected_item_id, item_num):
        item_weights_np = item_weights.detach().numpy()
        score = consine_similarity(item_weights_np[1],
                                   item_weights_np[item_id])
        similarity_scores.append({"item_id": item_id, "score": score})
        similarity_score_df = pd.DataFrame(similarity_scores)

    print(similarity_score_df.sort_values(
        "score", ascending=False, ignore_index=True))


def predict(user_id: int, item_id: int) -> np.ndarray:
    result: torch.Tensor = model(torch.Tensor([user_id]).to(
        torch.int64), torch.Tensor([item_id]).to(torch.int64))
    return result.detach().numpy()


def recommend_item(item_id: int):
    user_scores = []
    for user_id in range(1, user_num):
        score = predict(user_id, item_id)
        user_scores.append({"user_id": user_id, "score": score})
    df = pd.DataFrame(user_scores)
    print(df.sort_values("score", ascending=False))
    user_id = df.sort_values("score", ascending=False,
                             ignore_index=True)["user_id"][0]
    print(user_id)
    item_scores = []
    for item_id in range(1, item_num):
        score = predict(user_id, item_id)
        item_scores.append({"item_id": item_id, "score": score})
    item_df = pd.DataFrame(item_scores)
    print(item_df.sort_values("score", ascending=False))


def recommend(user_id: int):
    item_scores = []
    for item_id in range(1, item_num):
        score = predict(user_id, item_id)
        item_scores.append({"item_id": item_id, "score": score})
    df = pd.DataFrame(item_scores)
    print(df)
    df.to_csv(f"outputs/results-{user_id}.csv")


# web -> request -> item_id -> server -> recommend_itme(item_id) -> response -> web
recommend_item(1)
item_similarity(1)
recommend(1)
