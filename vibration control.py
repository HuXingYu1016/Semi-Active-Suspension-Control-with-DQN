import numpy as np
from scipy.signal import cont2discrete, lti, dlti, dstep
import matplotlib.pyplot as plt
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        GPU_num = torch.cuda.current_device()
        self.device = torch.device("cuda:{}".format(GPU_num))

    def forward(self, obs):
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        x = self.fc1(obs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)

        return out

class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size  # Define the maximum size of the replay_buffer
        self.buffer = []  # Define the replay_buffer storage list (storage core)
        self.index = 0  # Define the replay_buffer index
        self.length = 0  # Defines the current length of the replay_buffer (run step)

    def add(self, state, action, reward, next_state):
        # Combine the above data and store it in [data]
        data = (state, action, reward, next_state)

        # Store data
        if self.index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data

        # Index update
        self.index = (self.index + 1) % self.size

        # Length update
        self.length = min(self.length + 1, self.size)

    def sample(self, batch_size, n_steps=1):
        # samples initialization, uniform with PER form to record weights and indexes
        # common replay_buffer, indexes randomly generated, weights all-1 matrix
        samples = {'weights': np.ones(shape=batch_size, dtype=np.float32),
                   'indexes': np.random.choice(self.length - n_steps + 1, batch_size, replace=False)}

        # Data sampling
        sample_data = []
        for i in samples['indexes']:
            data_i = self.buffer[i]
            sample_data.append(data_i)
        return samples, sample_data

class EpsilonGreedy(object):

    def __init__(self, start_epsilon, end_epsilon, decay_step):

        assert 0 <= start_epsilon <= 1
        assert 0 <= end_epsilon <= 1
        assert decay_step >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_step = decay_step
        self.counters = 0
        self.epsilon = start_epsilon

    def compute_epsilon(self):

        if self.counters > self.decay_step:
            epsilon = self.end_epsilon
            self.counters += 1
            return epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            epsilon = self.start_epsilon + epsilon_diff * (self.counters / self.decay_step)
            self.counters += 1
            return epsilon

    def generate_action(self, original_action):

        self.epsilon = self.compute_epsilon()
        if np.random.random() > self.epsilon:  # Greedy
            action = original_action
        else:  # Random
            action = random_action(original_action)
        return action

def random_action(original_action):
    action = np.random.choice(np.arange(action_dim), len(original_action))
    return action

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.soft_update_tau = 0.1
        self.time_counter = 0
        self.warmup_step = 5000
        self.update_interval = 10
        self.target_update_interval = 5000
        self.explorer = EpsilonGreedy(start_epsilon=0.5, end_epsilon=0.01, decay_step=50000)


    def choose_action(self, obs):
        action = self.q_net(obs)

        action = torch.argmax(action, dim=1)

        action = self.explorer.generate_action(action)

        return action


    def learn(self):
        if (self.time_counter <= self.warmup_step) or \
                (self.time_counter % self.update_interval != 0):
            self.time_counter += 1
            return False
        else:
            return True

    def train(self, samples, sample_data):
        elementwise_loss = self.compute_loss(sample_data)

        loss = self.loss_process(elementwise_loss, samples['weights'])

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # Determining whether to update the target network
        if self.time_counter % self.target_update_interval == 0:
            self.soft_update()

        self.time_counter += 1


    def loss_process(self, loss, weight):
        # Calculation of loss based on weights
        weight = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        loss = torch.mean(loss * weight)

        return loss

    def compute_loss(self, data_batch):
        loss = []
        TD_error = []

        for elem in data_batch:
            state, action, reward, next_state = elem
            action = torch.as_tensor(action, dtype=torch.long, device=self.device)

            # Predicted value
            q_predict = self.q_net(state)
            q_predict = q_predict.gather(1, action.unsqueeze(1)).squeeze(1)

            q_predict_save = q_predict.detach().cpu().numpy().reshape(len(q_predict), 1)
            data_useful = np.any(q_predict_save, axis=1)

            # Target value
            q_next = self.target_q_net(next_state)
            q_next = q_next.max(dim=1)[0]
            q_target = reward + self.gamma * q_next

            # TD_error
            TD_error_sample = torch.abs(q_target - q_predict)
            TD_error_sample = torch.mean(TD_error_sample)
            # Count the TD_error of the current sample in the total TD_error
            TD_error.append(TD_error_sample)

            # Loss calculation
            loss_sample = F.smooth_l1_loss(q_predict, q_target)
            # Add the loss of the current sample to the total loss
            loss.append(loss_sample)

        # Further processing of TD_error
        TD_error = torch.stack(TD_error)

        # Combine the loses of different samples in a sample into a tensor
        loss = torch.stack(loss)

        return loss

    def soft_update(self):
        assert 0.0 < self.soft_update_tau < 1.0

        # Parameters update
        for target_param, source_param in zip(self.target_q_net.parameters(),
                                              self.q_net.parameters()):
            target_param.data.copy_((1 - self.soft_update_tau) *
                                    target_param.data + self.soft_update_tau * source_param.data)





lr = 2e-3
num_episodes = 50
hidden_dim = 64
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 1000000
minimal_size = 50000
batch_size = 64
Warmup_Steps = 5000
warmup_count = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

replay_buffer = ReplayBuffer(buffer_size)
state_dim = 2
action_dim = 2
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

def step(action, cf, cr):
    if action[0] == 0:
        cf = 75
    elif action[0] == 1:
        cf = 1500
    if action[1] == 0:
        cr = 75
    elif action[1] == 1:
        cr = 1500
    return cf, cr

for p in range(1, num_episodes + 1):
    mb = 1380
    Ib = 2444
    ls = 0.5
    kf = 17000
    cf = 1500
    lf = 1.25
    kr = 22000
    cr = 1500
    lr = 1.51
    mtf = 81
    ktf = 384000
    mtr = 91
    ktr = 384000
    ms = 12
    ks = 17000
    cs = 100
    H = 0.1
    L = 0.5
    Lp = 5
    v = 10

    S = 5
    dt = 0.001
    T = int(S / dt)
    S_1 = 1
    S_2 = 0.1
    S_3 = 0.5
    S_4 = S - S_1 - 2 * S_2 - S_3
    S_5 = lf + lr
    S_6 = S - S_5 - 2 * S_2 - S_3
    h_1 = []
    h_2 = []

    A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-ks / ms, -cs / ms, ks / ms, cs / ms, ks * ls / ms, cs * ls / ms, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [ks / mb, cs / mb, -(kf + kr + ks) / mb, -(cf + cr + cs) / mb, (kf * lf - kr * lr - ks * ls) / mb,
                   (cf * lf - cr * lr - cs * ls) / mb, kf / mb, cf / mb, kr / mb, cr / mb],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [ks * ls / Ib, cs * ls / Ib, (kf * lf - kr * lr - ks * ls) / Ib,
                   (cf * lf - cr * lr - cs * ls) / Ib, -(kf * lf ** 2 + kr * lr ** 2 + ks * ls ** 2) / Ib,
                   -(cf * lf ** 2 + cr * lr ** 2 + cs * ls ** 2) / Ib, -kf * lf / Ib, -cf * lf / Ib, kr * lr / Ib,
                   cr * lr / Ib],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, kf / mtf, cf / mtf, -kf * lf / mtf, -cf * lf / mtf, -(kf + ktf) / mtf, -cf / mtf, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, kr / mtr, cr / mtr, kr * lr / mtr, cr * lr / mtr, 0, 0, -(kr + ktr) / mtr, -cr / mtr]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [ktf / mtf, 0],
                  [0, 0],
                  [0, ktr / mtr]])
    C = np.eye(10)
    D = np.zeros((10, 2))

    for i in range(T):
        if i <= S_1 / dt:
            h_1.append(0)
        elif S_1 / dt < i <= (S_1 + S_2) / dt:
            h_1.append(0.001 * (i - S_1 / dt))
        elif (S_1 + S_2) / dt < i <= (S_1 + S_2 + S_3) / dt:
            h_1.append(0.1)
        elif (S_1 + S_2 + S_3) / dt < i <= (S_1 + 2 * S_2 + S_3) / dt:
            h_1.append(0.1 - 0.001 * (i - (S_1 + S_2 + S_3) / dt))
        else:
            h_1.append(0)

    for j in range(T):
        if j <= S_5 / dt:
            h_2.append(0)
        elif S_5 / dt < j <= (S_5 + S_2) / dt:
            h_2.append(0.001 * (j - S_5 / dt))
        elif (S_5 + S_2) / dt < j <= (S_5 + S_2 + S_3) / dt:
            h_2.append(0.1)
        elif (S_5 + S_2 + S_3) / dt < j <= (S_5 + 2 * S_2 + S_3) / dt:
            h_2.append(0.1 - 0.001 * (j - (S_5 + S_2 + S_3) / dt))
        else:
            h_2.append(0)

    x_0 = np.zeros(10).transpose()
    state = x_0
    obs = np.array([[state[6], state[7]], [state[8], state[9]]])
    R = []
    for m in range(T):

        a_sum = []
        if warmup_count <= Warmup_Steps:
            action = np.random.choice((np.arange(action_dim)), 2)
        else:
            action = agent.choose_action(obs)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            if isinstance(action, np.ndarray):
                action = action.copy()

        cf_new, cr_new = step(action, cf, cr)
        cf = cf_new
        cr = cr_new

        A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [-ks / ms, -cs / ms, ks / ms, cs / ms, ks * ls / ms, cs * ls / ms, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [ks / mb, cs / mb, -(kf + kr + ks) / mb, -(cf + cr + cs) / mb, (kf * lf - kr * lr - ks * ls) / mb,
                       (cf * lf - cr * lr - cs * ls) / mb, kf / mb, cf / mb, kr / mb, cr / mb],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [ks * ls / Ib, cs * ls / Ib, (kf * lf - kr * lr - ks * ls) / Ib,
                       (cf * lf - cr * lr - cs * ls) / Ib, -(kf * lf ** 2 + kr * lr ** 2 + ks * ls ** 2) / Ib,
                       -(cf * lf ** 2 + cr * lr ** 2 + cs * ls ** 2) / Ib, -kf * lf / Ib, -cf * lf / Ib, kr * lr / Ib,
                       cr * lr / Ib],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, kf / mtf, cf / mtf, -kf * lf / mtf, -cf * lf / mtf, -(kf + ktf) / mtf, -cf / mtf, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, kr / mtr, cr / mtr, kr * lr / mtr, cr * lr / mtr, 0, 0, -(kr + ktr) / mtr, -cr / mtr]])
        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [ktf / mtf, 0],
                      [0, 0],
                      [0, ktr / mtr]])
        C = np.eye(10)
        D = np.zeros((10, 2))

        d_system = cont2discrete((A, B, C, D), dt, method='zoh')
        A = d_system[0]
        B = d_system[1]
        C = d_system[2]
        D = d_system[3]

        h = np.array([h_1[m], h_2[m]])
        x = np.dot(A, x_0) + np.dot(B, h)
        a = (x[1] - x_0[1]) / dt
        reward = - (a**2)
        next_state = x
        next_obs = np.array([[next_state[6], next_state[7]], [next_state[8], next_state[9]]])
        replay_buffer.add(obs, action, reward, next_obs)
        if agent.learn():
            samples, sample_data = replay_buffer.sample(batch_size)
            agent.train(samples, sample_data)

        x_0 = x
        obs = next_obs
        warmup_count += 1

        R.append(reward)

    r_total = np.mean(R)
    if p % 1 == 0:
        print('Training Episode:', p, 'Reward', r_total)

# 测试网络效果
mb = 1380
Ib = 2444
ls = 0.5
kf = 17000
cf = 1500
lf = 1.25
kr = 22000
cr = 1500
lr = 1.51
mtf = 81
ktf = 384000
mtr = 91
ktr = 384000
ms = 12
ks = 17000
cs = 100
H = 0.1
L = 0.5
Lp = 5
v = 10

S = 30
dt = 0.001
T = int(S / dt)
S_1 = 10
S_2 = 0.1
S_3 = 0.5
S_4 = S - S_1 - 2 * S_2 - S_3
S_5 = lf + lr
S_6 = S - S_5 - 2 * S_2 - S_3
h_1 = []
h_2 = []

A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-ks / ms, -cs / ms, ks / ms, cs / ms, ks * ls / ms, cs * ls / ms, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [ks / mb, cs / mb, -(kf + kr + ks) / mb, -(cf + cr + cs) / mb, (kf * lf - kr * lr - ks * ls) / mb,
                   (cf * lf - cr * lr - cs * ls) / mb, kf / mb, cf / mb, kr / mb, cr / mb],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [ks * ls / Ib, cs * ls / Ib, (kf * lf - kr * lr - ks * ls) / Ib,
                   (cf * lf - cr * lr - cs * ls) / Ib, -(kf * lf ** 2 + kr * lr ** 2 + ks * ls ** 2) / Ib,
                   -(cf * lf ** 2 + cr * lr ** 2 + cs * ls ** 2) / Ib, -kf * lf / Ib, -cf * lf / Ib, kr * lr / Ib,
                   cr * lr / Ib],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, kf / mtf, cf / mtf, -kf * lf / mtf, -cf * lf / mtf, -(kf + ktf) / mtf, -cf / mtf, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, kr / mtr, cr / mtr, kr * lr / mtr, cr * lr / mtr, 0, 0, -(kr + ktr) / mtr, -cr / mtr]])

B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [ktf / mtf, 0],
                  [0, 0],
                  [0, ktr / mtr]])
C = np.eye(10)
D = np.zeros((10, 2))
d_system = cont2discrete((A, B, C, D), dt, method='zoh')
A = d_system[0]
B = d_system[1]
C = d_system[2]
D = d_system[3]

for i in range(T):
    R = []
    if i <= S_1 / dt:
        h_1.append(0)
    elif S_1 / dt < i <= (S_1 + S_2) / dt:
        h_1.append(0.001 * (i - S_1 / dt))
    elif (S_1 + S_2) / dt < i <= (S_1 + S_2 + S_3) / dt:
        h_1.append(0.1)
    elif (S_1 + S_2 + S_3) / dt < i <= (S_1 + 2 * S_2 + S_3) / dt:
        h_1.append(0.1 - 0.001 * (i - (S_1 + S_2 + S_3) / dt))
    else:
        h_1.append(0)

for j in range(T):
    if j <= S_5 / dt:
        h_2.append(0)
    elif S_5 / dt < j <= (S_5 + S_2) / dt:
        h_2.append(0.001 * (j - S_5 / dt))
    elif (S_5 + S_2) / dt < j <= (S_5 + S_2 + S_3) / dt:
        h_2.append(0.1)
    elif (S_5 + S_2 + S_3) / dt < j <= (S_5 + 2 * S_2 + S_3) / dt:
        h_2.append(0.1 - 0.001 * (j - (S_5 + S_2 + S_3) / dt))
    else:
        h_2.append(0)

x_0 = np.zeros(10).transpose()
r_1 = []
for m in range(T):
    h = np.array([h_1[m], h_2[m]])
    x = np.dot(A, x_0) + np.dot(B, h)
    a = (x[1] - x_0[1]) / dt
    x_0 = x
    r_1.append(a)

x_0 = np.zeros(10).transpose()
r_2 = []
state = x_0
obs = np.array([[state[6], state[7]], [state[8], state[9]]])
for n in range(T):
    action = agent.choose_action(obs)
    cf_new, cr_new = step(action, cf, cr)
    cf = cf_new
    cr = cr_new

    A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-ks / ms, -cs / ms, ks / ms, cs / ms, ks * ls / ms, cs * ls / ms, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [ks / mb, cs / mb, -(kf + kr + ks) / mb, -(cf + cr + cs) / mb, (kf * lf - kr * lr - ks * ls) / mb,
                   (cf * lf - cr * lr - cs * ls) / mb, kf / mb, cf / mb, kr / mb, cr / mb],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [ks * ls / Ib, cs * ls / Ib, (kf * lf - kr * lr - ks * ls) / Ib,
                   (cf * lf - cr * lr - cs * ls) / Ib, -(kf * lf ** 2 + kr * lr ** 2 + ks * ls ** 2) / Ib,
                   -(cf * lf ** 2 + cr * lr ** 2 + cs * ls ** 2) / Ib, -kf * lf / Ib, -cf * lf / Ib, kr * lr / Ib,
                   cr * lr / Ib],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, kf / mtf, cf / mtf, -kf * lf / mtf, -cf * lf / mtf, -(kf + ktf) / mtf, -cf / mtf, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, kr / mtr, cr / mtr, kr * lr / mtr, cr * lr / mtr, 0, 0, -(kr + ktr) / mtr, -cr / mtr]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [ktf / mtf, 0],
                  [0, 0],
                  [0, ktr / mtr]])
    C = np.eye(10)
    D = np.zeros((10, 2))

    d_system = cont2discrete((A, B, C, D), dt, method='zoh')
    A = d_system[0]
    B = d_system[1]
    C = d_system[2]
    D = d_system[3]

    h = np.array([h_1[n], h_2[n]])
    x = np.dot(A, x_0) + np.dot(B, h)
    a = (x[1] - x_0[1]) / dt
    r_2.append(a)
    x_0 = x
    next_state = x
    obs = np.array([[next_state[6], next_state[7]], [next_state[8], next_state[9]]])

print(r_2)