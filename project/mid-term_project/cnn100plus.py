import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

devicee = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        # self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        # self.bn7 = nn.BatchNorm2d(1024)
        # self.conv8 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)
        # self.bn8 = nn.BatchNorm2d(2048)
        self.fc1 = nn.Linear(512 * 8 * 8, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 100)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(self.bn2(F.relu(self.conv2(x))), 2)
        x = self.bn3(F.relu(self.conv3(x)))
        x = F.max_pool2d(self.bn4(F.relu(self.conv4(x))), 2)
        x = self.bn5(F.relu(self.conv5(x)))
        x = F.max_pool2d(self.bn6(F.relu(self.conv6(x))), 2)
        # print(x.shape)
        x = x.view(-1, 512 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载数据集
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor()])
train_dataset = ImageFolder('./face_classification_100/train_sample', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_dataset = ImageFolder('./face_classification_100/dev_sample', transform=transform)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
test_dataset = ImageFolder('./face_classification_100/test_sample', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义损失函数和优化器
net = Net().to(devicee)
criterion = nn.CrossEntropyLoss().to(devicee)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):  # 多次循环数据集
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # 获取输入数据
        inputs, labels = data
        inputs = inputs.to(devicee)
        labels = labels.to(devicee)
        # print(inputs.shape)
        # print(labels.shape)

        # 梯度清零
        optimizer.zero_grad()

        # 正向传递
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # 反向传递和优化
        loss.backward()
        optimizer.step()

        # 输出统计信息
        running_loss += loss.item()
        if i % 10 == 9:
            # 每10个小批次打印一次统计信息
            print('[%d, %5d] loss: %.5f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

correct = 0
total = 0
with torch.no_grad():
    for data in dev_loader:
        inputs, labels = data
        inputs = inputs.to(devicee)
        labels = labels.to(devicee)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the dev images: %d %%' % (100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(devicee)
        labels = labels.to(devicee)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


