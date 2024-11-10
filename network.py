import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layer1_dim=128, layer2_dim=64, seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim + action_dim, layer1_dim)
        self.fc2 = nn.Linear(layer1_dim, layer2_dim)
        self.fc3 = nn.Linear(layer2_dim, 1)

    def forward(self, state, action):
        x = torch.cat((state,action),1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class DeepDQN(nn.Module):
    def __init__(self, state_dim, action_dim, layer1_dim=128, layer2_dim=64, seed=0):
        super(DeepDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_dim + action_dim, layer1_dim)
        self.bn1 = nn.BatchNorm1d(layer1_dim)
        self.fc2 = nn.Linear(layer1_dim, layer1_dim)
        self.bn2 = nn.BatchNorm1d(layer1_dim)
        self.fc3 = nn.Linear(layer1_dim, layer2_dim)
        self.bn3 = nn.BatchNorm1d(layer2_dim)
        self.fc4 = nn.Linear(layer2_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat((state,action),1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = torch.tanh(self.fc4(x))
        return x
    
class DeepDQN_enforce(nn.Module):
    def __init__(self, state_dim, action_dim, layer1_dim=128, layer2_dim=64, seed=0):
        super(DeepDQN_enforce, self).__init__()
        self.seed = torch.manual_seed(seed)
        concat_dim = state_dim + action_dim 
        
        self.fc1 = nn.Linear(concat_dim, layer1_dim)
        self.bn1 = nn.BatchNorm1d(layer1_dim)
        self.fc2 = nn.Linear(concat_dim + layer1_dim, layer1_dim)
        self.bn2 = nn.BatchNorm1d(layer1_dim)
        self.fc3 = nn.Linear(concat_dim+layer1_dim, layer2_dim)
        self.bn3 = nn.BatchNorm1d(layer2_dim)
        self.fc4 = nn.Linear(layer2_dim, 1)
    
    def forward(self, state, action):
        input_ = torch.cat((state,action),1)
        x = F.relu(self.fc1(input_))
        x = self.bn1(x)
        x = F.relu(self.fc2(torch.cat([input_, x], dim=1)))
        x = self.bn2(x)
        x = F.relu(self.fc3(torch.cat([input_, x], dim=1)))
        x = self.bn3(x)
        x = torch.tanh(self.fc4(x))
        return x
    