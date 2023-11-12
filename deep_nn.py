# DDPG agent
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # current state_size -> 11
        # current action_size -> 2

BUFFER_SIZE = 10000000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.001
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.lrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.lrelu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        
        x = self.fc3(x)
        x = self.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        self.fc4 = nn.Linear(hidden_size, 1)
        
        self.lrelu = nn.LeakyReLU()

    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], 1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        
        x = self.fc4(x)
        
        return x
    
class DDPG:
    def __init__(self, state_size, action_size, hidden_size, writer_tag="default"):

        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01

        self.actor = Actor(state_size, action_size, hidden_size).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.critic = Critic(state_size, action_size, hidden_size).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.replay_buffer = ReplayBuffer()
        
        # initialize OU Noise
        self.noise = OUNoise(action_size)
        self.writer = SummaryWriter(log_dir=f"runs/{writer_tag}")
		
    # for tensorboard
    def write_summary(self, episode, reward, loss_actor, loss_critic):
        #log values for evaluation curves
        self.writer.add_scalar('Reward', reward, episode)
        self.writer.add_scalar('Loss/Actor', loss_actor, episode)
        self.writer.add_scalar('Loss/Critic', loss_critic, episode)
        

    # def act(self, state, epsilon=0):
    #     # print(state.shape)
    #     state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    #     self.actor.eval()
    #     with torch.no_grad():
    #         action = self.actor(state).cpu().data.numpy()
    #     self.actor.train()
    #     action += epsilon * self.noise.sample()  
    #     return np.clip(action, -1, 1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()

        # Add noise for exploration
        action += self.epsilon * self.noise.sample()
        action = np.clip(action, -1, 1)

        # Decay epsilon
        # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
        return action

    def reset_epsilon(self):
        self.epsilon = 1.0
  
    def train(self):
        if len(self.replay_buffer.buffer) < BATCH_SIZE:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Update critic
        #print(action)
        Q = self.critic(state, action)
        next_action = self.target_actor(next_state)
        next_Q = self.target_critic(next_state, next_action.detach())
        target_Q = reward + GAMMA * next_Q * (1 - done)
        critic_loss = nn.MSELoss()(Q, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_loss_value = critic_loss.item()
        
        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor_loss_value = actor_loss.item()

        # Update target networks
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(TAU * source.data + (1 - TAU) * target.data)
        for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(TAU * source.data + (1 - TAU) * target.data)
        
    def update_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def save_checkpoint(self, filename, episode_count):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode_count': episode_count,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        episode_count = checkpoint.get('episode_count', 0)
        return episode_count

    
class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.max_size = BUFFER_SIZE
        self.ptr = 0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.max_size
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward.reshape(-1, 1), next_state, done.reshape(-1, 1)
	
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state