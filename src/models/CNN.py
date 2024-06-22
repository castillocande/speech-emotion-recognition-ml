import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 input channel (grayscale), 8 output channels, 3x3 kernel size
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)   # 2x2 average pooling
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 8 input channels, 16 output channels, 3x3 kernel size
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)   # 2x2 average pooling
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 16 input channels, 32 output channels, 3x3 kernel size
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)    # 2x2 average pooling
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 2048)  # 32 channels after pooling, 28x28 image size assumed here
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 8)  # 8 classes for output

    def forward(self, x):
        x = F.relu(self.conv1(x)) # input: 224*224
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.avgpool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)  # Softmax activation for multi-class classification
        return x
