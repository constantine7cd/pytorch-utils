import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class TestNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1a.weight)

        self.conv1b = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1b.weight)

        self.conv1c = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1c.weight)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(in_channels=128, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2a.weight)

        self.conv2b = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2b.weight)

        self.conv2c = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2c.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv2d(in_channels=256, out_channels=512,
                                kernel_size=3)
        nn.init.kaiming_normal_(self.conv3a.weight)

        self.conv3b = nn.Conv2d(in_channels=512, out_channels=256,
                                kernel_size=1)
        nn.init.kaiming_normal_(self.conv3b.weight)

        self.conv3c = nn.Conv2d(in_channels=256, out_channels=128,
                                kernel_size=1)
        nn.init.kaiming_normal_(self.conv3c.weight)

        self.pool3 = nn.AvgPool2d(kernel_size=5)

        self.flatten = Flatten()

        self.dense = nn.Linear(in_features=128, out_features=1)
        nn.init.kaiming_normal_(self.dense.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv1a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv1b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv1c(x), negative_slope=0.1)
        x = self.pool1(x)

        x = F.leaky_relu(self.conv2a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2c(x), negative_slope=0.1)
        x = self.pool2(x)

        x = F.leaky_relu(self.conv3a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3c(x), negative_slope=0.1)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x


class PiModel_v1(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1a.weight)

        self.conv1b = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1b.weight)

        self.conv1c = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1c.weight)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout2d()

        self.conv2a = nn.Conv2d(in_channels=128, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2a.weight)

        self.conv2b = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2b.weight)

        self.conv2c = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2c.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout2 = nn.Dropout2d()

        self.conv3a = nn.Conv2d(in_channels=256, out_channels=512,
                                kernel_size=3)
        nn.init.kaiming_normal_(self.conv3a.weight)

        self.conv3b = nn.Conv2d(in_channels=512, out_channels=256,
                                kernel_size=1)
        nn.init.kaiming_normal_(self.conv3b.weight)

        self.conv3c = nn.Conv2d(in_channels=256, out_channels=128,
                                kernel_size=1)
        nn.init.kaiming_normal_(self.conv3c.weight)

        self.pool3 = nn.AvgPool2d(kernel_size=5)

        self.flatten = Flatten()

        self.dense = nn.Linear(in_features=128, out_features=1)
        nn.init.kaiming_normal_(self.dense.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv1a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv1b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv1c(x), negative_slope=0.1)
        x = self.pool1(x)

        x = self.dropout1(x)

        x = F.leaky_relu(self.conv2a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2c(x), negative_slope=0.1)
        x = self.pool2(x)

        x = self.dropout2(x)

        x = F.leaky_relu(self.conv3a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3c(x), negative_slope=0.1)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x


class PiModel_v2(nn.Module):

    def __init__(self, p_drop=0.5):
        super().__init__()

        self.p_drop = p_drop

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1a.weight)

        self.conv1b = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1b.weight)

        self.conv1c = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1c.weight)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout2d(p=p_drop)

        self.conv2a = nn.Conv2d(in_channels=128, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2a.weight)

        self.conv2b = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2b.weight)

        self.conv2c = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2c.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout2 = nn.Dropout2d(p=p_drop)

        self.conv3a = nn.Conv2d(in_channels=256, out_channels=512,
                                kernel_size=3)
        nn.init.kaiming_normal_(self.conv3a.weight)

        self.conv3b = nn.Conv2d(in_channels=512, out_channels=256,
                                kernel_size=1)
        nn.init.kaiming_normal_(self.conv3b.weight)

        self.conv3c = nn.Conv2d(in_channels=256, out_channels=128,
                                kernel_size=1)
        nn.init.kaiming_normal_(self.conv3c.weight)

        self.pool3 = nn.AvgPool2d(kernel_size=5)

        self.flatten = Flatten()

        self.dense = nn.Linear(in_features=128, out_features=1)
        nn.init.kaiming_normal_(self.dense.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv1a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv1b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv1c(x), negative_slope=0.1)
        x = self.pool1(x)

        x = self.dropout1(x)

        x = F.leaky_relu(self.conv2a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2c(x), negative_slope=0.1)
        x = self.pool2(x)

        x = self.dropout2(x)

        x = F.leaky_relu(self.conv3a(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3b(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3c(x), negative_slope=0.1)
        x = self.pool3(x)

        encoding = self.flatten(x)
        x = self.dense(encoding)

        return encoding, x
