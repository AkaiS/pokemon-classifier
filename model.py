import torch.nn as nn
from torch.nn import functional as F

class ResidualConvolutionalNN(nn.Module):
  def __init__(self, num_labels):
    super().__init__()
    self.blocks = nn.Sequential(
      Conv1(),
      ConvN(64, 64, 3, 3, 1),
      ConvN(64, 128, 3, 4, 2),
      ConvN(128, 256, 3, 6, 2),
      ConvN(256, 512, 3, 3, 2),
    )
    self.av_l = nn.AvgPool2d(7, stride=1)
    self.fc = nn.Linear(512, num_labels)
  
  def forward(self, x, target=None):
    x = self.blocks(x)
    x = self.av_l(x)
    x = x.view(x.shape[0], -1)
    logits = self.fc(x)
    if target is None:
      loss = None
    else:
      loss = F.cross_entropy(logits, target)
    return logits, loss

# First convolutional layer is static and unique (pools) relative to subsequent layers
class Conv1(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(3, 64, 7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(3, 2, padding=1),
    )

  def forward(self, x):
    x = self.net(x)
    return x

# For convolutional layers 2 - 5
class ConvN(nn.Module):
  def __init__(self, cin, cout, k_size, num_rc_blocks, stride):
    super().__init__()
    self.net = nn.Sequential(
      ResidConvBlock(cin, cout, k_size, stride),
      *[ResidConvBlock(cout, cout, k_size) for _ in range(num_rc_blocks - 1)]
    )
  
  def forward(self, x):
    x = self.net(x)
    return x

# A block of convolutions with a residual connection, not flexible to deeper models represented in paper
class ResidConvBlock(nn.Module):
  def __init__(self, cin, cout, k_size, stride=1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(cin, cout, k_size, padding=1, stride=stride),
      nn.BatchNorm2d(cout),
      nn.ReLU(),
      nn.Conv2d(cout, cout, k_size, padding=1),
    )
    self.bn = nn.BatchNorm2d(cout)
    self.relu = nn.ReLU()
    self.linear_proj = None if cin == cout else nn.Conv2d(cin, cout, 1, stride)
  
  def forward(self, x):
    if self.linear_proj is not None:
      return self.relu(self.bn(self.linear_proj(x) + self.net(x)))
    else:
      return self.relu(self.bn(x + self.net(x)))