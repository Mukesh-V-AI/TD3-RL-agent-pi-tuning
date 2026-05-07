import torch
import torch.nn as nn

# ---------- ACTOR ----------
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256,a_dim), nn.Tanh()
        )

    def forward(self, s):
        return self.net(s)

# ---------- IQN CRITIC ----------
class IQN(nn.Module):
    def __init__(self, s_dim, a_dim, n_quantiles=32):
        super().__init__()
        self.nq = n_quantiles

        self.feature = nn.Sequential(
            nn.Linear(s_dim+a_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU()
        )

        self.quantile = nn.Linear(256, n_quantiles)

    def forward(self, s, a):
        x = torch.cat([s,a], dim=1)
        h = self.feature(x)
        return self.quantile(h)

def quantile_loss(pred, target):
    td = target - pred
    return torch.mean(torch.where(td.abs()<1, 0.5*td**2, td.abs()-0.5))