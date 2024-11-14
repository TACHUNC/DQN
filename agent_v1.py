import math
import random
import sobol_seq
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize

from torch.utils.tensorboard import SummaryWriter

from network import QNetwork, DeepDQN, DeepDQN_enforce
from buffer_replay import ReplayBuffer
from utils import get_grad_norm, auto_clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, 
                 bounds:torch.Tensor,
                 state_dim:int, 
                 action_dim:int,
                 layer1_dim:int=128,
                 layer2_dim:int=64,
                 batch_size:int=64,
                 buffer_size:int=1e5, 
                 tau:float=1e-2,
                 gamma:float=0.99,
    	         learning_rate:float=1e-3, 
                 update_rate:int=300, 
                 n_updates:int=64,
                 n_step_bootstrapping:int=1,
                 n_multistart:int=8,
                 seed:int=0,
                 writer:SummaryWriter=SummaryWriter):
        
        """ Parameters:
            -----------
            - state_size (int): dimension of each state
            - action_size (int): dimension of each action
            - replay_memory size (int): size of the replay memory buffer (typically 5e4 to 5e6)
            - batch_size (int): size of the memory batch used for model updates (typically 32, 64 or 128)
            - gamma (float): paramete for setting the discoun ted value of future rewards (typically .95 to .995)
            - learning_rate (float): specifies the rate of model learing (typically 1e-4 to 1e-3))
            - seed (int): random seed for initializing training point. """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.buffer_size = int(buffer_size)
        self.tau = tau
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.update_rate = update_rate
        self.n_updates = n_updates
        self.n_multistart = n_multistart
        self.seed = random.seed(seed)
        self.bounds = bounds
        self.batch_bounds = [(self.bounds[i][0].item(), self.bounds[i][1].item()) for i in range(self.action_dim)]
        
        # Q network and optimizer
        self.network = DeepDQN(state_dim, action_dim, layer1_dim, layer2_dim, seed).to(device)
        self.target_network = DeepDQN(state_dim, action_dim, layer1_dim, layer2_dim, seed).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size=buffer_size,batch_size=batch_size,
                                   device=device, seed=self.seed,gamma=self.gamma,
                                   nstep=n_step_bootstrapping)
        
        # autoclipping
        self.auto_clip = auto_clip()
        # writer
        self.writer = writer
        
        self.t_step = 0
        # initalize parameters for epsilon decay
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.decay_rate = 1e-5

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(self.n_updates):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    def action_selection(self, states, network:nn.Module, eps_greedy:bool, time_step:int=0)->tuple[torch.Tensor, torch.Tensor]:
        batch_size = self.batch_size

        if states.dim() == 1:
            states = states.reshape(1,-1)
            batch_size = 1

        network.eval()
        # Shape: [batch_size, n_multistart, action_dim]
        init_action_batch   = self.param_init(self.bounds, self.n_multistart, batch_size) 
        best_actions_batch  = []
        best_q_values_batch = []        
        
        for batch, (state,init_action_multi) in enumerate(zip(states,init_action_batch)): 
            state = state.unsqueeze(0)
            best_action  = None
            best_q_value = -math.inf

            if eps_greedy and random.random() < self.eps_decay(time_step):
                # only works for normalized action
                best_action = torch.FloatTensor(1, self.action_dim).uniform_(-1, 1)
                best_q_value = network(state, best_action)
            else:
                for multi, init_action in enumerate(init_action_multi):
            
                    def objective(action_np):
                        action_tensor = torch.from_numpy(action_np).to(torch.float32).view(1, self.action_dim)
                        q_value = network(state, action_tensor) 
                        return -q_value.item()  

                    result = minimize(objective, init_action, method='SLSQP', bounds=self.batch_bounds,tol=1e-8, options={'maxiter': 5000})
                    
                    q_value_optimized = -result.fun if result.success else -math.inf
                    
                    if q_value_optimized > best_q_value:
                        best_q_value = q_value_optimized
                        best_action = result.x

                if best_q_value == -math.inf: print('failure')
                   
                best_action = torch.from_numpy(best_action).to(torch.float32).view(1, self.action_dim)
            
            best_actions_batch.append(best_action)
            best_q_values_batch.append([best_q_value])

        network.train()
        best_actions_batch = torch.stack(best_actions_batch)
        best_q_values_batch = torch.tensor(best_q_values_batch)
        
        return best_q_values_batch, best_actions_batch
    
    @staticmethod
    def param_init(bounds, n_multistart, batch_size=1):
        bounds_ = np.array(bounds)
        lb = bounds_[:,0]
        ub = bounds_[:,1]
        num_dim = bounds_.shape[0]
        
        # Initialize x_init array for all batches
        x_init = np.zeros((batch_size, n_multistart, num_dim))

        for b in range(batch_size):
            sobol_points = sobol_seq.i4_sobol_generate(num_dim, n_multistart)
            scaled_points = lb + (ub - lb) * sobol_points
            x_init[b] = scaled_points
        return x_init
        
    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            Q_ , _  = self.action_selection(next_states, self.target_network, False)
                    
        Q_network   = self.network(states, actions)
        Q_target    = rewards + (self.gamma * Q_  * (1 - dones))        
        # loss        = F.huber_loss(Q_network,Q_target)
        loss        = F.mse_loss(Q_network,Q_target)
        

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        total_norm_before = get_grad_norm(self.network)
        
        # gradient clip norm
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
        self.auto_clip(self.network)
        
        total_norm_after = get_grad_norm(self.network)
        clipping_ratio = total_norm_before / total_norm_after

        self.optimizer.step()

        # update target network
        self.soft_update(self.network, self.target_network, self.tau)
        # update to the tensorboard
        self.writer.add_scalar('MSE_LOSS', loss, self.time_step)  
        self.writer.add_scalar('Grad_norm_before', total_norm_before, self.time_step)  
        self.writer.add_scalar('Grad_norm_after', total_norm_after, self.time_step)  
        self.writer.add_scalar('Clipping_ratio', clipping_ratio, self.time_step)  
        

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save(self, checkpoint_path):
        torch.save(self.network.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.network.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True))
        self.target_network.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True))
    
    def predict(self,state,deterministic=True):
        self.network.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)
            q_value, action = self.action_selection(state,self.network,False)
        self.network.train()
        return action.squeeze().numpy().reshape((self.action_dim,)), q_value.squeeze().numpy().reshape((1,))
    
    def eps_decay(self, time_step):
        self.time_step = time_step
        return max(self.epsilon_end, self.epsilon_start - self.decay_rate * time_step)