import torch
import torch.optim as optim
from networks import Actor, IQN, quantile_loss

class CSAC:
    def __init__(self, s_dim, a_dim):

        self.actor = Actor(s_dim, a_dim)
        self.critic = IQN(s_dim, a_dim)
        self.target = IQN(s_dim, a_dim)

        self.actor_opt = optim.Adam(self.actor.parameters(), 3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), 3e-4)

        # Lagrange multipliers
        self.lambda_os = torch.tensor(1.0, requires_grad=True)
        self.lambda_us = torch.tensor(1.0, requires_grad=True)
        self.lambda_ts = torch.tensor(1.0, requires_grad=True)

        self.lambda_opt = optim.Adam(
            [self.lambda_os, self.lambda_us, self.lambda_ts], 1e-3
        )

        self.gamma = 0.99
        self.tau = 0.005

    def act(self, s):
        s = torch.FloatTensor(s).unsqueeze(0)
        return self.actor(s).detach().numpy()[0]

    def update(self, buffer):

        s,a,r,s2,d = buffer.sample()

        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        s2 = torch.FloatTensor(s2)

        # ---------- CRITIC ----------
        q = self.critic(s,a)

        with torch.no_grad():
            a2 = self.actor(s2)
            q2 = self.target(s2,a2)

        loss_c = quantile_loss(q, q2)

        # ---------- PINN (light) ----------
        loss_c += 0.01 * (q**2).mean()

        self.critic_opt.zero_grad()
        loss_c.backward()
        self.critic_opt.step()

        # ---------- ACTOR ----------
        a_new = self.actor(s)
        q_new = self.critic(s, a_new)

        # CVaR (worst-case)
        q_sorted = q_new.sort(dim=1)[0]
        Q = q_sorted[:,:6].mean()

        # constraint values (placeholder batch)
        g_os = torch.mean(q_new)*0 + 0  # replace with env metrics
        g_us = torch.mean(q_new)*0 + 0
        g_ts = torch.mean(q_new)*0 + 0

        loss_a = -Q \
            + self.lambda_os*g_os \
            + self.lambda_us*g_us \
            + self.lambda_ts*g_ts

        self.actor_opt.zero_grad()
        loss_a.backward(retain_graph=True)
        self.actor_opt.step()

        # ---------- LAMBDA UPDATE ----------
        loss_l = -(self.lambda_os*g_os.detach() + self.lambda_us*g_us.detach() + self.lambda_ts*g_ts.detach())

        self.lambda_opt.zero_grad()
        loss_l.backward()
        self.lambda_opt.step()

        self.lambda_os.data.clamp_(0)
        self.lambda_us.data.clamp_(0)
        self.lambda_ts.data.clamp_(0)

        # ---------- TARGET UPDATE ----------
        for p, tp in zip(self.critic.parameters(), self.target.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)