import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
		self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
		self.conv4 = nn.Conv2d(40, 80, kernel_size=3)
		self.conv5 = nn.Conv2d(80, 160, kernel_size=3)
		self.conv5_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(2560, 50)
		self.fc2 = nn.Linear(50, nclasses)
		
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(F.relu(self.conv3(x)), 2)
		x = F.relu(self.conv4(x))
		x = self.conv5_drop(F.max_pool2d(F.relu(self.conv5(x)), 2))
		x = x.view(-1, 2560)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=-1)