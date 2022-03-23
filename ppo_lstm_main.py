import torch
import torch.nn as nn
import torch.nn.functional as F
import xlsxwriter
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
import gym
from logger import Logger
from env.two_subgoals import TwoRoom, TwoRoomTest
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

logger = Logger('./logs')

TRAIN = True
TEST = False
RENDER = False

STATE_DIM = 10
ACTION_DIM = 4
NUM_AGENT = 1
TRAIN_EPISIODES = 20000  # episodes
TEST_EPISIODES = 10000
TIME_NUMS = 100  # num_rollout 一轮步长
A_HIDDEN = 32
C_HIDDEN = 32
EPOCH = 2
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


def roll_out(model, env, first_state, hiddenp_in, hiddenv_in, time_nums, dtype, device):
    model.p.to('cpu')
    model.v.to('cpu')
    score = 0
    s = first_state
    hp_in = hiddenp_in
    hv_in = hiddenv_in
    done_flag = False
    for t in range(time_nums):
        action_list = []
        state = torch.from_numpy(s).to(dtype).unsqueeze(0).unsqueeze(0)
        prob, hp_out = model.p(state, hp_in)
        _, hv_out = model.v(state, hv_in)
        prob = prob.view(-1)
        m = Categorical(prob)
        a = m.sample().item()
        for i in range(NUM_AGENT):
            action_list.append(a)
        if RENDER:
            env.render()
        ns, r, done, _ = env.step(action_list)
        ns = ns.reshape(-1)
        pad_mask = 1
        done_mask = 0 if done else 1

        model.put_data((s, a, r/100.0, ns, prob[a].item(), done_mask, pad_mask, hp_in, hp_out, hv_in, hv_out))

        s = ns
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
    return s, hp_in, hv_in, done_flag, score


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
        model = PPOLSTM(STATE_DIM, ACTION_DIM)   # (state_dim, action_dim)
        writer = SummaryWriter("logs_model")
        graph_inputs1 = torch.rand(1,1,10)
        graph_inputs2 = torch.rand(1,1,32)
        writer.add_graph(model.p, input_to_model=(graph_inputs1, (graph_inputs2, graph_inputs2)))
        writer.add_graph(model.v, input_to_model=(graph_inputs1, (graph_inputs2, graph_inputs2)))
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
        while epi_i < TRAIN_EPISIODES:
            s_first, hinp_first, hinv_first, done_flag, score = \
                roll_out(model, env, s_first, hinp_first, hinv_first, TIME_NUMS, dtype, device)
            if done_flag:
                epi_i += 1
            if score >= 10:
                success += 1
            score_sum += score
            s_l, a_l, r_l, ns_l, fp_l, dm_l, pm_l, (h1p, c1p), (h2p, c2p), (h1v, c1v), (h2v, c2v) = model.make_batch()
            first_hidden_p = (h1p.to(device).detach(), c1p.to(device).detach())
            second_hidden_p = (h2p.to(device).detach(), c2p.to(device).detach())
            first_hidden_v = (h1v.to(device).detach(), c1v.to(device).detach())
            second_hidden_v = (h2v.to(device).detach(), c2v.to(device).detach())
            ad_l, q_l = advantage_cal(model, s_l, ns_l, dm_l, r_l, first_hidden_v, second_hidden_v, device, dtype)

            # prepossess
            s_b = torch.tensor(s_l, dtype=dtype, device=device).unsqueeze(0)
            a_b = torch.tensor(a_l, dtype=torch.long, device=device)
            fp_b = torch.tensor(fp_l, dtype=dtype, device=device)
            pm_b = torch.tensor(pm_l, dtype=dtype, device=device)
            ad_b = torch.tensor(ad_l, dtype=dtype, device=device)
            q_b = torch.tensor(q_l, dtype=dtype, device=device)

            # train
            for _ in range(EPOCH):
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
                # policy_loss_sum = 0
                # value_loss_sum = 0
                tb_sum_i = 0
                tb_log_i += 1
                score_sum = 0

            if epi_i % 2000 == 0:
                if not os.path.exists('model/ppo_lstm_9'):
                    os.makedirs('model/ppo_lstm_9')
                torch.save(model, f'model/ppo_lstm_9/ppo_lstm_{epi_i}.pth')
            if epi_i % 2000 == 0:
                plt.plot(all_rewards_avg)
                if not os.path.exists('image/ppo_lstm_9'):
                    os.makedirs('image/ppo_lstm_9')
                plt.savefig(os.path.join(f'image/ppo_lstm_9/{epi_i}'))
            if epi_i % 20 == 0:
                success_rate = success / 20
                success_all.append(success_rate)
                print('success rate:{}'.format(success_rate))
                success = 0
        
            writer.add_scalar('loss-A', policy_loss, epi_i)
            writer.add_scalar('loss-C', value_loss, epi_i)

        torch.cuda.empty_cache()
        plt.plot(all_rewards_avg)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image/ppo_lstm_9/rewards'))
        plt.close()

        plt.plot(success_all)
        if not os.path.exists('image/ppo_lstm_9'):
            os.makedirs('image/ppo_lstm_9')
        plt.savefig(os.path.join('image/ppo_lstm_9/success'))
        plt.close()

        workbook = xlsxwriter.Workbook('data/ppo_train/reward10_success_rate.xlsx')
        worksheet = workbook.add_worksheet()
        for i in range(1, success_all.__len__() + 1):
            t = 'A' + str(i)
            worksheet.write(t, float(success_all[i - 1]))
        workbook.close()

    if TEST:
        device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float32
        env = TwoRoomTest(TIME_NUMS)
        model = torch.load('model/ppo_lstm_9/ppo_lstm_20000.pth', map_location=torch.device('cpu'))  # (state_dim, action_dim)
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
        success = 0
        success_all = []
        while epi_i < TEST_EPISIODES:
            s_first, hinp_first, hinv_first, done_flag, score = \
                roll_out(model, env, s_first, hinp_first, hinv_first, TIME_NUMS, dtype, device)
            if done_flag:
                epi_i += 1
            if score >= 10:
                success += 1
            score_sum += score

            # tensorboard and log
            if epi_i % print_interval == 0 and epi_i != 0 and done_flag:
                # policy_loss_avg = policy_loss_sum / tb_sum_i
                # value_loss_avg = value_loss_sum / tb_sum_i
                score_avg = score_sum / print_interval

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
        if not os.path.exists('image/ppo_lstm_test'):
            os.makedirs('image/ppo_lstm_test')
        plt.savefig(os.path.join('image/ppo_lstm_test/success_test10'))
        plt.close()

        workbook = xlsxwriter.Workbook('data/ppo_test/reward10_success_rate.xlsx')
        worksheet = workbook.add_worksheet()
        for i in range(1, success_all.__len__() + 1):
            t = 'A' + str(i)
            worksheet.write(t, float(success_all[i - 1]))
        workbook.close()

if __name__ == '__main__':
    main()