import torch

class BiasedMatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)
        self.user_biases = torch.nn.Embedding(n_users, 1, sparse=True)
        self.item_biases = torch.nn.Embedding(n_items, 1, sparse=True)

    def forward(self, user, item):
        pred: torch.Tensor = self.user_biases(user) + self.item_biases(item)
        pred += (
            (self.user_factors(user) * self.item_factors(item))
            .sum(dim=1, keepdim=True)
        )
        return pred.squeeze()