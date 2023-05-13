import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# from torch.optim.lr_scheduler import ReduceLROnPlateau

devicee = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 56 50
# 相较于609参考网络上的架构

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 96, 7, 2, 1,),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(96),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(256)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(384),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
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
        self.fc1 = nn.Linear(512 * 6*6, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 500)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图层
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return x


# 加载数据集
transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
train_dataset = ImageFolder(
    "./face_classification_500/train_sample", transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_dataset = ImageFolder("./face_classification_500/dev_sample", transform=transform)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
test_dataset = ImageFolder("./face_classification_500/test_sample", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义损失函数和优化器
net = Net().to(devicee)
criterion = nn.CrossEntropyLoss().to(devicee)
optimizer = optim.SGD(net.parameters(), lr=0.007)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.7,
    patience=10,
    verbose=False,
    threshold=0.0001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-08,
)
train_loss = []
dev_acc = []
test_acc = []


# 训练模型
for epoch in range(100):  # 多次循环数据集
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
            scheduler.step(running_loss / 10)
            print("[%d, %5d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    print("lr = ", optimizer.state_dict()["param_groups"][0]["lr"])
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
    dev_acc.append((100 * correct / total))

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
    test_acc.append((100 * correct / total))

torch.save(net.state_dict(), 'model.pkl')

plt.plot(train_loss)
plt.title("training loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

plt.plot(dev_acc)
plt.title("dev acc")
plt.xlabel("Iterations")
plt.ylabel("dev acc")
plt.show()

plt.plot(test_acc)
plt.title("test acc")
plt.xlabel("Iterations")
plt.ylabel("test acc")
plt.show()
