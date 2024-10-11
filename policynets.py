import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class CNNPolicyNetwork(nn.Module):
    def __init__(self, world_size, obs_channels, act_dim):
        super(CNNPolicyNetwork, self).__init__()

        """
        BEWARE: pooling ignore pixels that dont fit !!! hence 5x5 pooled by 2x2 kernel with 2 stride -> 2x2
        """

        self.conv1 = nn.Conv2d(obs_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)          
        self.pool = nn.MaxPool2d(2, 2)        
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)                 
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)                   
        self.pool2 = nn.MaxPool2d(2, 2)
        
        flat_size = (world_size // 2) // 2
        self.flat_dim = 64 * flat_size * flat_size
        self.fc1 = nn.Linear(self.flat_dim, 64)
        self.fc2 = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = x.view(-1, self.flat_dim)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
