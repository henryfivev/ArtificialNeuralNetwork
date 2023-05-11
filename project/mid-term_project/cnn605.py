import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

devicee = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 39%

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),nn.Conv2d(3, 16, 5, 1, 1,), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 1), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 128, 5, 1, 1), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(512)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 2048, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(2048)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(2048, 8192, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(8192)
        )
        # self.conv7 = nn.Sequential(
        #     nn.Conv2d(4096, 8192, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(8192)
        # )
        # self.conv8 = nn.Sequential(
        #     nn.Conv2d(8192, 16384, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(16384)
        # )
        # self.conv9 = nn.Sequential(
        #     nn.Conv2d(16384, 32768, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),nn.BatchNorm2d(32768)
        # )
        self.fc1 = nn.Linear(8192*3*3, 1024)
        self.fc2 = nn.Linear(1024, 500)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# 加载数据集
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = ImageFolder("./face_classification_500/train_sample", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataset = ImageFolder("./face_classification_500/dev_sample", transform=transform)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_dataset = ImageFolder("./face_classification_500/test_sample", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义损失函数和优化器
net = Net().to(devicee)
criterion = nn.CrossEntropyLoss().to(devicee)
optimizer = optim.SGD(net.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
train_loss = []

# 训练模型
for epoch in range(10):  # 多次循环数据集
    net.train()
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
            print("[%d, %5d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    scheduler.step()
    net.eval()
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
    print("Accuracy of the network on the dev images: %d %%" % (100 * correct / total))

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

    print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))

plt.plot(train_loss)
plt.title("training loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
