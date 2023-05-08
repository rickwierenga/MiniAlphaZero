# ResidualBlock is based on https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py

import torch.nn as nn

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride = 1):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels))
    self.relu = nn.ReLU()
    self.out_channels = out_channels

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    out += residual
    out = self.relu(out)
    return out


class Network(nn.Module):
  def __init__(self, board_size, policy_shape, num_layers):
    """ board_size: (num_rows, num_columns) """
    super(Network, self).__init__()

    self.num_rows, self.num_columns = board_size
    self.policy_shape = policy_shape
    self.policy_size = 1
    for dim in self.policy_shape:
      self.policy_size *= dim

    # initial block
    self.num_channels = 64
    self.conv = conv3x3(3, self.num_channels)
    self.bn = nn.BatchNorm2d(self.num_channels)
    self.relu = nn.ReLU(inplace=True)

    # residual blocks
    self.blocks = []
    for _ in range(num_layers):
      block = ResidualBlock(self.num_channels, self.num_channels, stride=1)
      self.blocks.append(block)

    # policy head
    self.policy_head = nn.Sequential(
      nn.Conv2d(self.num_channels, 2, kernel_size=1, stride=1),
      nn.BatchNorm2d(2),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(2 * self.num_rows * self.num_columns, self.policy_size),
      nn.Softmax(dim=1)
    )

    # value head
    self.value_head = nn.Sequential(
      nn.Conv2d(self.num_channels, 1, kernel_size=1, stride=1),
      nn.BatchNorm2d(1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(self.num_rows * self.num_columns, 256),
      nn.ReLU(),
      nn.Linear(256, 1),
      nn.Tanh()
    )

  def forward(self, encoded_board):
    x = encoded_board
    
    # batch size, num_channels, num_rows, num_columns
    x = x.view(-1, 3, self.num_rows, self.num_columns)

    # initial block
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)

    # residual blocks
    for block in self.blocks:
      x = block(x)

    # policy and value head
    x_value = self.value_head(x)
    x_policy = self.policy_head(x)
    x_policy = x_policy.reshape(self.policy_shape)
    return x_policy, x_value
