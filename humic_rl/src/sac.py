# https://github.com/pranz24/pytorch-soft-actor-critic

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
eps = 1e-6

class ReplayBuffer(object):
    def __init__(self, obs_dim, fullstate_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs_state = np.zeros((self.max_size, obs_dim))
        self.full_state = np.zeros((self.max_size, fullstate_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_obs_state = np.zeros((self.max_size, obs_dim))
        self.next_full_state = np.zeros((self.max_size, fullstate_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.obs_state[self.ptr] = state['observation']
        self.full_state[self.ptr] = state['fullstate']
        self.action[self.ptr] = action
        self.next_obs_state[self.ptr] = next_state['observation']
        self.next_full_state[self.ptr] = next_state['fullstate']
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1 - int(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
                {
                    'observation': torch.FloatTensor(self.obs_state[idx]).to(device),
                    'fullstate' : torch.FloatTensor(self.full_state[idx]).to(device)
                },
                torch.FloatTensor(self.action[idx]).to(device),
                {
                    'observation': torch.FloatTensor(self.next_obs_state[idx]).to(device),
                    'fullstate': torch.FloatTensor(self.next_full_state[idx]).to(device)
                },
                torch.FloatTensor(self.reward[idx]).to(device),
                torch.FloatTensor(self.not_done[idx]).to(device)
        )

# class ReplayBuffer(object):
#     def __init__(self, obs_dim, fullstate_dim, action_dim, max_size=int(1e6)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0

#         self.obs_state1 = np.zeros((self.max_size, obs_dim[0], obs_dim[1], obs_dim[2]))
#         self.obs_state2 = np.zeros((self.max_size, obs_dim[3]))
#         self.full_state = np.zeros((self.max_size, fullstate_dim))
#         self.action = np.zeros((self.max_size, action_dim))
#         self.next_obs_state1 = np.zeros((self.max_size, obs_dim[0], obs_dim[1], obs_dim[2]))
#         self.next_obs_state2 = np.zeros((self.max_size, obs_dim[3]))
#         self.next_full_state = np.zeros((self.max_size, fullstate_dim))
#         self.reward = np.zeros((self.max_size, 1))
#         self.not_done = np.zeros((self.max_size, 1))

#     def add(self, state, action, next_state, reward, done):
#         self.obs_state1[self.ptr] = state['observation'][0]
#         self.obs_state2[self.ptr] = state['observation'][1]
#         self.full_state[self.ptr] = state['fullstate']
#         self.action[self.ptr] = action
#         self.next_obs_state1[self.ptr] = next_state['observation'][0]
#         self.next_obs_state2[self.ptr] = next_state['observation'][1]
#         self.next_full_state[self.ptr] = next_state['fullstate']
#         self.reward[self.ptr] = reward
#         self.not_done[self.ptr] = 1 - int(done)

#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
    
#     def sample(self, batch_size):
#         idx = np.random.randint(0, self.size, size=batch_size)

#         return (
#                 {
#                     'observation': (torch.FloatTensor(self.obs_state1[idx]).to(device), torch.FloatTensor(self.obs_state2[idx]).to(device)),
#                     'fullstate' : torch.FloatTensor(self.full_state[idx]).to(device)
#                 },
#                 torch.FloatTensor(self.action[idx]).to(device),
#                 {
#                     'observation': (torch.FloatTensor(self.next_obs_state1[idx]).to(device), torch.FloatTensor(self.next_obs_state2[idx]).to(device)),
#                     'fullstate': torch.FloatTensor(self.next_full_state[idx]).to(device)
#                 },
#                 torch.FloatTensor(self.reward[idx]).to(device),
#                 torch.FloatTensor(self.not_done[idx]).to(device)
#         )




# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale, action_bias, action_div):
        super(GaussianPolicy, self).__init__()

        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        self.mean_l = nn.Linear(256, action_dim)
        self.log_std_l = nn.Linear(256, action_dim)

        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=device)
        self.action_bias = torch.tensor(action_bias, dtype=torch.float32, device=device)
        self.action_div = torch.tensor(action_div, dtype=torch.float32, device=device)

        self.apply(weights_init_)
    
    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        
        mean = self.mean_l(x)
        log_std = self.log_std_l(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample() # for reparameteriation trick(mean + std * N(0, 1))
        
        y_t = torch.tanh(x_t)

        action = ((y_t * self.action_scale) + self.action_bias) / self.action_div

        # Enforcing Action Bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) 
        
        mean = ((mean * self.action_scale) + self.action_bias) / self.action_div

        return action, log_prob, mean

class GaussianPolicyConv(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale, action_bias, action_div):
        super(GaussianPolicyConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 32)
        )

        self.l1 = nn.Linear(32 + obs_dim[-1], 256)
        self.l2 = nn.Linear(256, 256)
        # self.l1 = nn.Linear(obs_dim, 256)
        
        self.mean_l = nn.Linear(256, action_dim)
        self.log_std_l = nn.Linear(256, action_dim)

        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=device)
        self.action_bias = torch.tensor(action_bias, dtype=torch.float32, device=device)
        self.action_div = torch.tensor(action_div, dtype=torch.float32, device=device)

        self.apply(weights_init_)
    
    def forward(self, obs):
        # x = F.relu(self.l1(obs))
        # si = F.relu(self.fc1(obs[:, :512]))
        # si = F.relu(self.fc2(si))
        # x = F.relu(self.l1(torch.cat([si, obs[:, 512:]], 1)))
        si = self.conv(obs[0])
        x = F.relu(self.l1(torch.cat([si, obs[1]], 1)))
        x = F.relu(self.l2(x))
        
        mean = self.mean_l(x)
        log_std = self.log_std_l(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample() # for reparameteriation trick(mean + std * N(0, 1))
        
        y_t = torch.tanh(x_t)

        action = ((y_t * self.action_scale) + self.action_bias) / self.action_div

        # Enforcing Action Bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + eps)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) 
        
        mean = ((mean * self.action_scale) + self.action_bias) / self.action_div

        return action, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

class SAC(object):
    def __init__(
        self, 
        obs_dim,
        fullstate_dim,
        action_dim, 
        action_scale,
        action_bias,
        action_div,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=0.0003,
        target_update_interval=1,
        replay_size=int(1e6),
        batch_size=256,
        automatic_entropy_tuning=True
    ):
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.memory = ReplayBuffer(obs_dim=obs_dim, fullstate_dim=fullstate_dim, action_dim=action_dim, max_size=replay_size)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha # Temperature
        self.lr = lr
        self.batch_size = batch_size

        self.target_update_interval = target_update_interval

        self.critic = QNetwork(fullstate_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.policy = GaussianPolicy(obs_dim, action_dim, action_scale=action_scale, action_bias=action_bias, action_div=action_div).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        if self.automatic_entropy_tuning:
            print("AutoTuning")
            self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

    def select_action(self, obs, evaluate=False):
        state = torch.FloatTensor(obs).to(device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        
        return action.detach().squeeze().cpu().numpy()

    def update_parameters(self, updates):

        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch['observation'])
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch['fullstate'], next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch['fullstate'], action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf_loss =  F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch['observation'])

        qf1_pi, qf2_pi = self.critic(state_batch['fullstate'], pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        if updates % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return self.alpha

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + '_policy.pth')
        torch.save(self.policy_optim.state_dict(), filename + '_policy_optim.pth')
        torch.save(self.critic.state_dict(), filename + '_critic.pth')
        torch.save(self.critic_target.state_dict(), filename + '_critic_target.pth')
        torch.save(self.critic_optim.state_dict(), filename + '_critic_optim.pth')
        
        
    def load(self, filename, evaluate=True):
        if evaluate:
            self.policy.load_state_dict(torch.load(filename + '_policy.pth'))
            self.policy.eval()
        else: # train
            self.policy.load_state_dict(torch.load(filename + '_policy.pth'))
            self.policy_optim.load_state_dict(torch.load(filename + '_policy_optim.pth'))
            self.critic.load_state_dict(torch.load(filename + '_critic.pth'))
            self.critic_target.load_state_dict(torch.load(filename + '_critic_target.pth'))
            self.critic_optim.load_state_dict(torch.load(filename + '_critic_optim.pth'))

            self.policy.train()
            self.critic.train()
            self.critic_target.train()


# class MAReplayBuffer(object):
#     def __init__(self, agent_num, obs_dim, fullstate_dim, action_dim, max_size=int(1e6)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#         self.agent_num = agent_num

#         self.obs_state = np.zeros((self.max_size, obs_dim))
#         self.full_state = np.zeros((self.max_size, fullstate_dim))
#         self.next_obs_state = np.zeros((self.max_size, obs_dim))
#         self.next_full_state = np.zeros((self.max_size, fullstate_dim))
        
#         self.action = []
#         self.reward = []
#         self.not_done = []

#         for _ in range(agent_num):
#             self.action.append(np.zeros((self.max_size, action_dim)))
#             self.reward.append(np.zeros((self.max_size, 1)))
#             self.not_done.append(np.zeros((self.max_size, 1)))

#     def add(self, state, action, next_state, reward, done):
#         self.obs_state[self.ptr] = state['observation']
#         self.full_state[self.ptr] = state['fullstate']
#         self.next_obs_state[self.ptr] = next_state['observation']
#         self.next_full_state[self.ptr] = next_state['fullstate']
        
#         for i in range(self.agent_num):
#             self.action[i][self.ptr] = action[i]
#             self.reward[i][self.ptr] = reward[i]
#             self.not_done[i][self.ptr] = 1 - int(done[i])

#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
    
#     def sample(self, batch_size):
#         idx = np.random.randint(0, self.size, size=batch_size)

#         return (
#                 {
#                     'observation': torch.FloatTensor(self.obs_state[idx]).to(device),
#                     'fullstate' : torch.FloatTensor(self.full_state[idx]).to(device)
#                 },
#                 [torch.FloatTensor(action[idx]).to(device) for action in self.action],
#                 {
#                     'observation': torch.FloatTensor(self.next_obs_state[idx]).to(device),
#                     'fullstate': torch.FloatTensor(self.next_full_state[idx]).to(device)
#                 },
#                 [torch.FloatTensor(reward[idx]).to(device) for reward in self.reward],
#                 [torch.FloatTensor(not_done[idx]).to(device) for not_done in self.not_done]
#         )

class MAReplayBuffer(object):
    def __init__(self, agent_num, obs_dim, fullstate_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.agent_num = agent_num

        self.obs_state0 = np.zeros((self.max_size, obs_dim[0], obs_dim[1], obs_dim[2]))
        self.obs_state1 = np.zeros((self.max_size, obs_dim[3]))
        self.full_state = np.zeros((self.max_size, fullstate_dim))
        self.next_obs_state0 = np.zeros((self.max_size, obs_dim[0], obs_dim[1], obs_dim[2]))
        self.next_obs_state1 = np.zeros((self.max_size, obs_dim[3]))
        self.next_full_state = np.zeros((self.max_size, fullstate_dim))

        self.action = []
        self.reward = []
        self.not_done = []

        for _ in range(agent_num):
            self.action.append(np.zeros((self.max_size, action_dim)))
            self.reward.append(np.zeros((self.max_size, 1)))
            self.not_done.append(np.zeros((self.max_size, 1)))

    def add(self, state, action, next_state, reward, done):
        self.obs_state0[self.ptr] = state['observation'][0]
        self.obs_state1[self.ptr] = state['observation'][1]
        self.full_state[self.ptr] = state['fullstate']
        self.next_obs_state0[self.ptr] = next_state['observation'][0]
        self.next_obs_state1[self.ptr] = next_state['observation'][1]
        self.next_full_state[self.ptr] = next_state['fullstate']

        for i in range(self.agent_num):
            self.action[i][self.ptr] = action[i]
            self.reward[i][self.ptr] = reward[i]
            self.not_done[i][self.ptr] = 1 - int(done[i])
    
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
                {
                    'observation': (torch.FloatTensor(self.obs_state0[idx]).to(device), torch.FloatTensor(self.obs_state1[idx]).to(device)),
                    'fullstate' : torch.FloatTensor(self.full_state[idx]).to(device)
                },
                [torch.FloatTensor(action[idx]).to(device) for action in self.action],
                {
                    'observation': (torch.FloatTensor(self.next_obs_state0[idx]).to(device), torch.FloatTensor(self.next_obs_state1[idx]).to(device)),
                    'fullstate': torch.FloatTensor(self.next_full_state[idx]).to(device)
                },
                [torch.FloatTensor(reward[idx]).to(device) for reward in self.reward],
                [torch.FloatTensor(not_done[idx]).to(device) for not_done in self.not_done]
        )

class MASAC(object):
    def __init__(
        self,
        agent_num,
        obs_dim,
        fullstate_dim,
        action_dim,
        action_scale,
        action_bias,
        action_div,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=0.0003,
        target_update_interval=1,
        replay_size=int(1e6),
        batch_size=256,
        automatic_entropy_tuning=True
    ):
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.memory = MAReplayBuffer(agent_num=agent_num, obs_dim=obs_dim, fullstate_dim=fullstate_dim, action_dim=action_dim, max_size=replay_size)
        
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size

        self.target_update_interval = target_update_interval

        self.agent_num = agent_num
    
        self.alpha = []

        self.critic = []
        self.critic_target = []
        self.critic_optim = []

        self.policy = []
        self.policy_optim = []

        self.target_entropy = []
        self.log_alpha = []
        self.alpha_optim = []

        for i in range(self.agent_num):
            if self.automatic_entropy_tuning:
                print("AutoTuning Agent")
                self.alpha.append(alpha)
                self.critic.append(QNetwork(fullstate_dim, action_dim).to(device))
                self.critic_target.append(copy.deepcopy(self.critic[i]))
                self.critic_optim.append(Adam(self.critic[i].parameters(), lr=self.lr))
                self.policy.append(GaussianPolicyConv(obs_dim, action_dim, action_scale=action_scale, action_bias=action_bias, action_div=action_div).to(device))
                self.policy_optim.append(Adam(self.policy[i].parameters(), lr=self.lr))
                self.target_entropy.append(-torch.prod(torch.Tensor((action_dim,)).to(device)).item())
                self.log_alpha.append(torch.zeros(1, requires_grad=True, device=device))
                self.alpha_optim.append(Adam([self.log_alpha[i]], lr=self.lr))

            else:
                self.alpha.append(alpha)
                self.critic.append(QNetwork(fullstate_dim, action_dim).to(device))
                self.critic_target.append(copy.deepcopy(self.critic[i]))
                self.critic_optim.append(Adam(self.critic[i].parameters(), lr=self.lr))
                self.policy.append(GaussianPolicyConv(obs_dim, action_dim, action_scale=action_scale, action_bias=action_bias, action_div=action_div).to(device))
                self.policy_optim.append(Adam(self.policy[i].parameters(), lr=self.lr))

    def select_action(self, obs, evaluate=False):
        obs0 = torch.FloatTensor(obs[0]).to(device).unsqueeze(0)
        obs1 = torch.FloatTensor(obs[1]).to(device).unsqueeze(0)
        state = (obs0, obs1)

        action = []
        for i in range(self.agent_num):
            self.policy[i].eval()
            if evaluate is False:
                action_, _, _ = self.policy[i].sample(state)
                action.append(action_.detach().squeeze().cpu().numpy())
            else:
                _, _, action_ = self.policy[i].sample(state)
                action.append(action_.detach().squeeze().cpu().numpy())
            self.policy[i].train()
        
        return action

    def update_parameters(self, updates):

        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

        for i in range(self.agent_num):
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy[i].sample(next_state_batch['observation'])
                
                qf1_next_target, qf2_next_target = self.critic_target[i](next_state_batch['fullstate'], next_state_action)

                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha[i] * next_state_log_pi
                
                next_q_value = reward_batch[i] + mask_batch[i] * self.gamma * (min_qf_next_target)
                
            qf1, qf2 = self.critic[i](state_batch['fullstate'], action_batch[i])  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf_loss =  F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)
            
            self.critic_optim[i].zero_grad()
            qf_loss.backward()
            self.critic_optim[i].step()
            
            pi, log_pi, _ = self.policy[i].sample(state_batch['observation'])
            
            qf1_pi, qf2_pi = self.critic[i](state_batch['fullstate'], pi)
            
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            
            policy_loss = ((self.alpha[i] * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            
            self.policy_optim[i].zero_grad()
            policy_loss.backward()
            self.policy_optim[i].step()
            
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[i] * (log_pi + self.target_entropy[i]).detach()).mean()
                
                self.alpha_optim[i].zero_grad()

                alpha_loss.backward()
                
                self.alpha_optim[i].step()
                
                self.alpha[i] = self.log_alpha[i].exp()
                
            if updates % self.target_update_interval == 0:
                for param, target_param in zip(self.critic[i].parameters(), self.critic_target[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return self.alpha

    def save(self, filename):
        for i in range(self.agent_num):
            torch.save(self.policy[i].state_dict(), filename + '_policy{}.pth'.format(i))
            torch.save(self.policy_optim[i].state_dict(), filename + '_policy_optim{}.pth'.format(i))
            torch.save(self.critic[i].state_dict(), filename + '_critic{}.pth'.format(i))
            torch.save(self.critic_target[i].state_dict(), filename + '_critic_target{}.pth'.format(i))
            torch.save(self.critic_optim[i].state_dict(), filename + '_critic_optim{}.pth'.format(i))
        
    def load(self, filename, evaluate=True):
        for i in range(self.agent_num):
            if evaluate:
                self.policy[i].load_state_dict(torch.load(filename + '_policy{}.pth'.format(i)))
                self.policy[i].eval()
                
            else: # train
                self.policy[i].load_state_dict(torch.load(filename + '_policy{}.pth'.format(i)))
                self.policy_optim[i].load_state_dict(torch.load(filename + '_policy_optim{}.pth'.format(i)))
                self.critic[i].load_state_dict(torch.load(filename + '_critic{}.pth'.format(i)))
                self.critic_target[i].load_state_dict(torch.load(filename + '_critic_target{}.pth'.format(i)))
                self.critic_optim[i].load_state_dict(torch.load(filename + '_critic_optim{}.pth'.format(i)))

                self.policy[i].train()
                self.critic[i].train()
                self.critic_target[i].train()