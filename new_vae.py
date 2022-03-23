import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.Linear1 = nn.Linear(input_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out1 = self.relu(self.Linear1(input))
        out2 = self.relu(self.Linear2(out1))
        mu = self.mu(out2)
        sigma = self.sigma(out2)
        zeta = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        z = torch.autograd.Variable(zeta, requires_grad=False) * sigma.exp() + mu
        z = self.sigmoid(z)
        return mu, sigma, z


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=64, device='cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.Linear1 = nn.Linear(latent_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out1 = self.relu(self.Linear1(z))
        out2 = self.sigmoid(self.Linear2(out1))
        return out2


class Vae(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, device='cpu'):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim, device)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim, device)

    def forward(self, input):
        mu, sigma, z = self.encoder(input)
        output = self.decoder(z)
        return mu, sigma, z, output


class Vae_buffer:
    def __init__(self, task_number):
        # 用来记录任务1、任务2的成功轨迹和奖励,做第一层vae
        self.success_trajectory_buffer = []
        # 用来记录任务1、任务2编码得到的z，做第二层vae
        self.z_buffer = []
        for i in range(task_number):
            self.success_trajectory_buffer.append([])
            self.z_buffer.append([])

    def trajectory_padding(self, trajectory, length=100, max_length=160):
        # 对task_info,trajectory,r拼接的长度对齐
        # task_info=(起点，红点，绿点，终点，task_id),trajectory=(s2,s2,...,sT),r=(r1,r3,...,rT-1) 以每轮50步，不够的需要补长
        if trajectory.shape[0] < length:
            trajectory_padding = np.pad(trajectory, (0, length-trajectory.shape[0]), 'constant')
        if trajectory.shape[0] > length:
            trajectory_padding = trajectory[:length]
        return trajectory_padding


class Z_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, device='cpu'):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out1 = self.relu(self.linear1(input))
        out2 = self.sigmoid(self.linear2(out1))
        return out2








