import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

devicee = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 512) 
        self.fc2 = nn.Linear(512, 128) 
        self.fc3 = nn.Linear(128, 10) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        print(x.shape)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

# 加载数据集
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
train_dataset = ImageFolder('./face_classification_10/train_sample', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataset = ImageFolder('./face_classification_10/dev_sample', transform=transform)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_dataset = ImageFolder('./face_classification_10/test_sample', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义损失函数和优化器
net = Net().to(devicee)
criterion = nn.CrossEntropyLoss().to(devicee)
optimizer = optim.SGD(net.parameters(), lr=0.001)
train_loss = []

# 训练模型
for epoch in range(150):  # 多次循环数据集
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
        train_loss.append(loss.item())
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

plt.plot(train_loss)
plt.title('training loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()