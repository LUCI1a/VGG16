from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import multiprocessing
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
n=4# 第几张图片
target_class = 2
count = 0
target_image = None
for image, label in testset:
    if label == target_class:
        count += 1
        if count == n:
            target_image = image.unsqueeze(0)
            break

input_image = target_image.clone().detach().requires_grad_(True)

original_image = input_image.detach().squeeze().permute(1, 2, 0).numpy()
original_image = (original_image * 0.5 + 0.5).clip(0, 1)
plt.imshow(original_image)
plt.title(f"Original Image for Class {target_class}")
plt.axis('off')
plt.show()

model = ResNet18(num_classes=10)
model.load_state_dict(torch.load('resnet18_new.pth'))
model.eval()
optimizer = torch.optim.Adam([input_image], lr=0.1)

epsilon_1 = 0.1  # 对应于激活最大化的学习率
epsilon_2 = 0.1  # 对应于总变差正则化的学习率

# 定义总变差（TV）函数
def total_variation(image):
    diff1 = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    diff2 = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    return diff1 + diff2


# 进行迭代
for step in range(300):
    optimizer.zero_grad()
    output = model(input_image)

    if isinstance(output, tuple):
        output = output[0]
    target_activation = output[0, target_class]
    activation_gradient = torch.autograd.grad(target_activation, input_image)[0]    # 计算激活值的梯度
    tv_gradient = torch.autograd.grad(total_variation(input_image), input_image)[0]    # 计算总变差的梯度
    input_image.data = input_image.data + epsilon_1 * activation_gradient - epsilon_2 * tv_gradient
    input_image.data.clamp_(0, 1)
    loss = -target_activation + epsilon_2 / epsilon_1 * total_variation(input_image)
    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

optimized_image = input_image.detach().squeeze().permute(1, 2, 0).numpy()
optimized_image = (optimized_image * 0.5 + 0.5).clip(0, 1)
plt.imshow(optimized_image)
plt.title(f"Optimized Image for Class {target_class} ")
plt.axis('off')
plt.show()
