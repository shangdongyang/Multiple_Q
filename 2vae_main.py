import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
import gym
from logger import Logger
from env.two_subgoals import TwoRoom, TwoRoomTest
import os
import matplotlib.pyplot as plt
from new_vae import Vae_buffer, Vae
import xlsxwriter

logger = Logger('./logs')

TRAIN = False
TEST = True
RENDER = False
STATE_DIM = 10
ACTION_DIM = 4
Trajectory_latent_dim = 6
Z_latent_dim = 6
vae_lr = 0.001
NUM_AGENT = 1
Train_Task_Number = 2
Test_Task_Number = 2
# STEP = 200
Train_EPISIODES = 20000  # episodes
Test_EPISIODES = 10000
Trajectory_Epoch = 100
Z_Epoch = 100
PPO_EPOCH = 2
TIME_NUMS = 50  # num_rollout 一轮步长
A_HIDDEN = 32
C_HIDDEN = 32
OPTIM_BATCH = 1
SEQ_NUM = 1
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
print_interval = 1


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(64, 32)):
        super().__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, hidden_size[0])
        self.lstm = nn.LSTM(hidden_size[0], hidden_size[1], batch_first=True)
        self.action_head = nn.Linear(hidden_size[1], action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

        self.init()

    def init(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.action_head.weight, nonlinearity='relu')
        nn.init.constant_(self.action_head.bias, 0.0)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, 1.0)
        nn.init.constant_(self.lstm.bias_hh_l0, 1.0)

    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        x = self.action_head(x)
        prob = F.softmax(x, dim=2)
        return prob, hidden


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=(64, 32)):
        super().__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, hidden_size[0])
        self.lstm = nn.LSTM(hidden_size[0], hidden_size[1], batch_first=True)
        self.value_head = nn.Linear(hidden_size[1], 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

        self.init()

    def init(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.value_head.weight, nonlinearity='relu')
        nn.init.constant_(self.value_head.bias, 0.0)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, 1.0)
        nn.init.constant_(self.lstm.bias_hh_l0, 1.0)

    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        v = self.value_head(x)
        return v, hidden


class PPOLSTM:
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.data = []
        self.p = Actor(state_dim, action_dim)
        self.v = Critic(state_dim)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_l, a_l, r_l, ns_l, fp_l, dm_l, pm_l, hinp_l, houtp_l, hinv_l, houtv_l = \
            [], [], [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, ns, fp, dm, pm, hinp, houtp, hinv, houtv = transition
            s_l.append(s)
            a_l.append([a])
            r_l.append([r])
            ns_l.append(ns)
            fp_l.append([fp])
            dm_l.append([dm])
            pm_l.append([pm])
            hinp_l.append(hinp)
            houtp_l.append(houtp)
            hinv_l.append(hinv)
            houtv_l.append(houtv)
        self.data = []
        return s_l, a_l, r_l, ns_l, fp_l, dm_l, pm_l, hinp_l[0], houtp_l[0], hinv_l[0], houtv_l[0]


def roll_out(model, env, first_state, hiddenp_in, hiddenv_in, time_nums, dtype, device, Z_latent):
    model.p.to('cpu')
    model.v.to('cpu')
    score = 0
    s = first_state
    s_z = np.concatenate((s, Z_latent))
    hp_in = hiddenp_in
    hv_in = hiddenv_in
    done_flag = False
    #
    task_info = first_state
    trajectory_s = []
    trajectory_r = []
    for t in range(time_nums):
        action_list = []
        state_z = torch.from_numpy(s_z).to(dtype).unsqueeze(0).unsqueeze(0)
        prob, hp_out = model.p(state_z, hp_in)
        _, hv_out = model.v(state_z, hv_in)
        prob = prob.view(-1)
        m = Categorical(prob)
        a = m.sample().item()
        for i in range(NUM_AGENT):
            action_list.append(a)
        if RENDER:
            env.render()
        ns, r, done, _ = env.step(action_list)
        ns = ns.reshape(-1)
        ns_z = np.concatenate((ns, Z_latent))
        pad_mask = 1
        done_mask = 0 if done else 1

        # ppo训练数据
        model.put_data((s_z, a, r/100.0, ns_z, prob[a].item(), done_mask, pad_mask, hp_in, hp_out, hv_in, hv_out))
        # vae训练数据
        trajectory_s.append(ns[0:2])
        trajectory_r.append(np.array([r]))

        s_z = ns_z
        hp_in = hp_out
        hv_in = hv_out
        score += r
        done_flag = done
        if done:
            s = env.reset()
            s = s.reshape(-1)
            hx = torch.zeros(1, 1, A_HIDDEN)
            cx = torch.zeros(1, 1, A_HIDDEN)
            hp_in = (hx, cx)
            hv_in = (hx, cx)

            break
    model.p.to(device)
    model.v.to(device)
    trajectory_ss = np.stack(trajectory_s).reshape(-1)
    trajectory_rr = np.stack(trajectory_r).reshape(-1)
    return s, hp_in, hv_in, done_flag, score, task_info, trajectory_ss, trajectory_rr


def advantage_cal(model, states, next_states, dones, rewards, first_hidden_v, second_hidden_v, device, dtype):
    states = torch.tensor(states, dtype=dtype, device=device).unsqueeze(0)
    next_states = torch.tensor(next_states, dtype=dtype, device=device).unsqueeze(0)
    dones = torch.tensor(dones, dtype=dtype, device=device)
    rewards = torch.tensor(rewards, dtype=dtype, device=device)

    v_next, _ = model.v(next_states, second_hidden_v)
    v_next = v_next.squeeze(0)
    td_target = rewards + gamma * v_next * dones
    v, _ = model.v(states, first_hidden_v)
    v = v.squeeze(0)
    delta = (td_target - v).cpu().detach().numpy()

    advantage_list = []
    advantage = 0
    for item in delta[::-1]:
        advantage = gamma * lmbda * advantage + item[0]
        advantage_list.append([advantage])
    advantage_list.reverse()
    td_target = td_target.cpu().detach().numpy().tolist()
    return advantage_list, td_target


def main():
    if TRAIN:
        device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float32
        env = TwoRoom(TIME_NUMS)
        model = PPOLSTM(STATE_DIM + Z_latent_dim, ACTION_DIM)   # (state_dim + Z_latent_dim, action_dim)
        model.p.to(device)
        model.v.to(device)
        epi_i = 0
        score_sum = 0
        s_first = env.reset().reshape(-1)
        hx = torch.zeros(1, 1, A_HIDDEN, dtype=dtype)
        cx = torch.zeros(1, 1, A_HIDDEN, dtype=dtype)
        hinp_first = (hx, cx)
        hinv_first = (hx, cx)
        policy_loss_sum = 0
        value_loss_sum = 0
        tb_log_i = 1
        tb_sum_i = 0
        all_rewards_avg = []
        success = 0
        success_all = []

        # 初始化2个vae和优化器
        trajectory_dim = 105
        vae_trajectory = Vae(input_dim=trajectory_dim, latent_dim=Trajectory_latent_dim, hidden_dim=64, device=device)
        vae_trajectory_optim = torch.optim.Adam(vae_trajectory.parameters(), lr=vae_lr)

        vae_z = Vae(input_dim=Trajectory_latent_dim * 10, latent_dim=Z_latent_dim, hidden_dim=64, device=device)
        vae_z_optim = torch.optim.Adam(vae_z.parameters(), lr=vae_lr)
        mse_loss_function = nn.MSELoss()

        # 初始化vae所需数据buffer, vae_buffer.success_trajectory_buffer, vae_buffer.z_buffer
        # 初始化两个任务的Z_latent
        vae_buffer = Vae_buffer(Train_Task_Number)
        Z_latent1, Z_latent2 = np.zeros(Z_latent_dim), np.zeros(Z_latent_dim)

        # 采样+训练
        while epi_i < Train_EPISIODES:
            if env.c == 1:
                Z_latent = Z_latent1
            elif env.c == 2:
                Z_latent = Z_latent2
            s_first, hinp_first, hinv_first, done_flag, score, task_info, trajectory_s, trajectory_r = \
                roll_out(model, env, s_first, hinp_first, hinv_first, TIME_NUMS, dtype, device, Z_latent)
            if done_flag:
                epi_i += 1
            if score >= 10:
                success += 1
                # trajectory设置为105维，不够的需要补长，超出的进行裁剪
                trajectory = np.concatenate((task_info, trajectory_s, trajectory_r)).reshape(-1)
                trajectory_padding = vae_buffer.trajectory_padding(trajectory, length=trajectory_dim)
                if env.c == 1:
                    vae_buffer.success_trajectory_buffer[0].append(trajectory_padding)
                elif env.c == 2:
                    vae_buffer.success_trajectory_buffer[1].append(trajectory_padding)

            # 如果成功的轨迹大于1000条，开始训练vae_trajectory
            trajectory_num = len(vae_buffer.success_trajectory_buffer[0])+len(vae_buffer.success_trajectory_buffer[1])
            if trajectory_num >= 200 and epi_i % 100 == 0:
                trajectory1 = np.stack((vae_buffer.success_trajectory_buffer[0]))
                trajectory2 = np.stack((vae_buffer.success_trajectory_buffer[1]))
                data = torch.tensor(np.concatenate((trajectory1, trajectory2)), dtype=dtype)
                dataloader = torch.utils.data.DataLoader(data, batch_size=128)
                for epoch in range(Trajectory_Epoch):
                    for i, trajectory_i in enumerate(dataloader):
                        # print("step:{},batch_trajectory:{}".format(i, trajectory_i))
                        vae_trajectory_optim.zero_grad()
                        mu_i, sigma_i, z_i, dec_i = vae_trajectory(trajectory_i)
                        mu_sq = mu_i * mu_i
                        sigma_sq = sigma_i * sigma_i
                        kl_loss = 0.5 * torch.mean(mu_sq + sigma_sq - torch.log(sigma_sq) - 1)
                        recon_loss = mse_loss_function(trajectory_i, dec_i)
                        loss = 2 * kl_loss + recon_loss
                        loss.backward()
                        vae_trajectory_optim.step()

                vae_buffer.z_buffer = [[], []]
                mu1, sigma1, z1 = vae_trajectory.encoder(torch.tensor(trajectory1, dtype=dtype))
                vae_buffer.z_buffer[0] = z1
                mu2, sigma2, z2 = vae_trajectory.encoder(torch.tensor(trajectory2, dtype=dtype))
                vae_buffer.z_buffer[1] = z2

            z_num = len(vae_buffer.z_buffer[0]) + len(vae_buffer.z_buffer[1])
            if z_num >= 200 and epi_i % 20 == 0:
                z1_group = []
                z2_group = []
                z1_data = torch.utils.data.DataLoader(vae_buffer.z_buffer[0], batch_size=10, shuffle=True, drop_last=True)
                z2_data = torch.utils.data.DataLoader(vae_buffer.z_buffer[1], batch_size=10, shuffle=True, drop_last=True)
                z_group = []
                for i, z1_data_i in enumerate(z1_data):
                    z1_group_i = z1_data_i.reshape(-1)
                    z1_group.append(z1_group_i.detach().numpy())
                for i, z2_data_i in enumerate(z2_data):
                    z2_group_i = z2_data_i.reshape(-1)
                    z2_group.append(z2_group_i.detach().numpy())
                z_group = torch.tensor(np.concatenate((z1_group, z2_group)))
                for epoch in range(Z_Epoch):
                    vae_z_optim.zero_grad()
                    mu_z, sigma_z, zz, dec_z = vae_z(z_group)
                    mu_sq = mu_z * mu_z
                    sigma_sq = sigma_z * sigma_z
                    kl_loss = 0.5 * torch.mean(mu_sq + sigma_sq - torch.log(sigma_sq) - 1)
                    recon_loss = mse_loss_function(z_group, dec_z)
                    loss = 2 * kl_loss + recon_loss
                    loss.backward()
                    vae_z_optim.step()

                last_z1 = np.stack((vae_buffer.z_buffer[0].detach().numpy()[-10:])).reshape(-1)
                _, _, last_z1_latent = vae_z.encoder(torch.tensor(last_z1))
                Z_latent1 = last_z1_latent.detach().numpy()

                last_z2 = np.stack((vae_buffer.z_buffer[1].detach().numpy()[-10:])).reshape(-1)
                _, _, last_z2_latent = vae_z.encoder(torch.tensor(last_z2))
                Z_latent2 = last_z2_latent.detach().numpy()

            # prepossess
            score_sum += score
            s_l, a_l, r_l, ns_l, fp_l, dm_l, pm_l, (h1p, c1p), (h2p, c2p), (h1v, c1v), (h2v, c2v) = model.make_batch()
            first_hidden_p = (h1p.to(device).detach(), c1p.to(device).detach())
            second_hidden_p = (h2p.to(device).detach(), c2p.to(device).detach())
            first_hidden_v = (h1v.to(device).detach(), c1v.to(device).detach())
            second_hidden_v = (h2v.to(device).detach(), c2v.to(device).detach())
            ad_l, q_l = advantage_cal(model, s_l, ns_l, dm_l, r_l, first_hidden_v, second_hidden_v, device, dtype)

            s_b = torch.tensor(s_l, dtype=dtype, device=device).unsqueeze(0)
            a_b = torch.tensor(a_l, dtype=torch.long, device=device)
            fp_b = torch.tensor(fp_l, dtype=dtype, device=device)
            pm_b = torch.tensor(pm_l, dtype=dtype, device=device)
            ad_b = torch.tensor(ad_l, dtype=dtype, device=device)
            q_b = torch.tensor(q_l, dtype=dtype, device=device)

            # train
            for _ in range(PPO_EPOCH):
                p, _ = model.p(s_b, first_hidden_p)
                p_b = p.squeeze(0).gather(1, a_b)
                ratio = torch.exp(torch.log(p_b) - torch.log(fp_b))
                surr1 = ratio * ad_b
                surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * ad_b
                policy_loss = -torch.min(surr1, surr2).mean()
                model.p.optimizer.zero_grad()
                policy_loss.backward()
                model.p.optimizer.step()
                policy_loss_sum += policy_loss

                v, _ = model.v(s_b, first_hidden_p)
                v_b = v.squeeze(0)
                value_loss = F.smooth_l1_loss(v_b, q_b)
                model.v.optimizer.zero_grad()
                value_loss.backward()
                model.v.optimizer.step()
                value_loss_sum += value_loss

                tb_sum_i += 1

            # tensorboard and log
            if epi_i % print_interval == 0 and epi_i != 0 and done_flag:
                score_avg = score_sum / print_interval
                all_rewards_avg.append(score_avg)

                print('episode: {} avg score: {:.1f}'.format(epi_i, score_avg))
                policy_loss_sum = 0
                value_loss_sum = 0
                tb_sum_i = 0
                tb_log_i += 1
                score_sum = 0

            if epi_i % 2000 == 0:
                if not os.path.exists('model/2vae_6'):
                    os.makedirs('model/2vae_6')
                torch.save(model, f'model/2vae_6/ppo_lstm_{epi_i}.pth')
                torch.save(vae_trajectory, f'model/2vae_6/vae_trajectory_{epi_i}.pth')
                torch.save(vae_z, f'model/2vae_6/vae_z_{epi_i}.pth')
            if epi_i % 2000 == 0:
                plt.plot(all_rewards_avg)
                if not os.path.exists('image/2vae_6'):
                    os.makedirs('image/2vae_6')
                plt.savefig(os.path.join(f'image/2vae_6/{epi_i}'))
            if epi_i % 20 == 0:
                success_rate = success / 20
                success_all.append(success_rate)
                print('success rate:{}'.format(success_rate))
                success = 0

        torch.cuda.empty_cache()
        plt.plot(all_rewards_avg)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image/2vae_6/rewards'))
        plt.close()

        plt.plot(success_all)
        if not os.path.exists('image/2vae_6'):
            os.makedirs('image/2vae_6')
        plt.savefig(os.path.join('image/2vae_6/success'))
        plt.close()

        workbook = xlsxwriter.Workbook('data/2vae_train/reward10_success_rate.xlsx')
        worksheet = workbook.add_worksheet()
        for i in range(1, success_all.__len__() + 1):
            t = 'A' + str(i)
            worksheet.write(t, float(success_all[i - 1]))
        workbook.close()

    if TEST:
        device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float32
        env = TwoRoomTest(TIME_NUMS)
        model = torch.load('model/2vae_train/ppo_lstm_20000.pth')  # (state_dim + Z_latent_dim, action_dim)
        model.p.to(device)
        model.v.to(device)
        epi_i = 0
        score_sum = 0
        s_first = env.reset().reshape(-1)
        hx = torch.zeros(1, 1, A_HIDDEN, dtype=dtype)
        cx = torch.zeros(1, 1, A_HIDDEN, dtype=dtype)
        hinp_first = (hx, cx)
        hinv_first = (hx, cx)
        tb_log_i = 1
        all_rewards_avg = []
        success = 0
        success_all = []

        # 初始化2个vae和优化器
        trajectory_dim = 105
        vae_trajectory = torch.load('model/2vae_train/vae_trajectory_20000.pth')
        vae_z = torch.load('model/2vae_train/vae_z_20000.pth')

        # 初始化vae所需数据buffer, vae_buffer.success_trajectory_buffer, vae_buffer.z_buffer
        # 初始化两个任务的Z_latent
        vae_buffer = Vae_buffer(Test_Task_Number)
        Z_latent1, Z_latent2 = np.zeros(Z_latent_dim, dtype=float), np.zeros(Z_latent_dim, dtype=float)

        # 采样+训练
        while epi_i < Test_EPISIODES:
            if env.c == 3:
                Z_latent = Z_latent1
            elif env.c == 4:
                Z_latent = Z_latent2
            s_first, hinp_first, hinv_first, done_flag, score, task_info, trajectory_s, trajectory_r = \
                roll_out(model, env, s_first, hinp_first, hinv_first, TIME_NUMS, dtype, device, Z_latent)
            if done_flag:
                epi_i += 1
            if score >= 10:
                success += 1
                # trajectory设置为105维，不够的需要补长，超出的进行裁剪
                trajectory = np.concatenate((task_info, trajectory_s, trajectory_r)).reshape(-1)
                trajectory_padding = vae_buffer.trajectory_padding(trajectory, length=trajectory_dim)
                if env.c == 3:
                    vae_buffer.success_trajectory_buffer[0].append(trajectory_padding)
                elif env.c == 4:
                    vae_buffer.success_trajectory_buffer[1].append(trajectory_padding)

            trajectory1_num = len(vae_buffer.success_trajectory_buffer[0])
            trajectory2_num = len(vae_buffer.success_trajectory_buffer[1])
            if trajectory1_num >= 10 and trajectory2_num >= 10 and epi_i % 20 == 0:
                trajectory1 = np.stack((vae_buffer.success_trajectory_buffer[0]))
                trajectory2 = np.stack((vae_buffer.success_trajectory_buffer[1]))
                vae_buffer.z_buffer = [[], []]
                mu1, sigma1, z1 = vae_trajectory.encoder(torch.tensor(trajectory1, dtype=dtype))
                vae_buffer.z_buffer[0] = z1
                mu2, sigma2, z2 = vae_trajectory.encoder(torch.tensor(trajectory2, dtype=dtype))
                vae_buffer.z_buffer[1] = z2

                last_z1 = np.stack((vae_buffer.z_buffer[0].detach().numpy()[-10:])).reshape(-1)
                _, _, last_z1_latent = vae_z.encoder(torch.tensor(last_z1))
                Z_latent1 = last_z1_latent.detach().numpy()

                last_z2 = np.stack((vae_buffer.z_buffer[1].detach().numpy()[-10:])).reshape(-1)
                _, _, last_z2_latent = vae_z.encoder(torch.tensor(last_z2))
                Z_latent2 = last_z2_latent.detach().numpy()

            score_sum += score
            # tensorboard and log
            if epi_i % print_interval == 0 and epi_i != 0 and done_flag:
                score_avg = score_sum / print_interval
                all_rewards_avg.append(score_avg)

                print('episode: {} avg score: {:.1f}'.format(epi_i, score_avg))
                policy_loss_sum = 0
                value_loss_sum = 0
                tb_sum_i = 0
                tb_log_i += 1
                score_sum = 0

            if epi_i % 20 == 0:
                success_rate = success / 20
                success_all.append(success_rate)
                print('success rate:{}'.format(success_rate))
                success = 0

        torch.cuda.empty_cache()
        plt.plot(success_all)
        if not os.path.exists('image/2vae_test'):
            os.makedirs('image/2vae_test')
        plt.savefig(os.path.join('image/2vae_test/success0'))
        plt.close()

        workbook = xlsxwriter.Workbook('data/2vae_test/reward10_success_rate.xlsx')
        worksheet = workbook.add_worksheet()
        for i in range(1, success_all.__len__() + 1):
            t = 'A' + str(i)
            worksheet.write(t, float(success_all[i - 1]))
        workbook.close()

if __name__ == '__main__':
    main()