from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x6 = torch.flatten(x5, 1)
        x7 = self.fc(x6)

        return x7, [x, x1, x2, x3, x4]


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


model = ResNet18(10)
model.load_state_dict(torch.load('resnet18.pth'))
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
def analyze_and_save_heatmap(model, dataloader, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    for idx, (img, label) in enumerate(dataloader):
        print(f"Processing image {idx + 1} with label: {label.item()}")
        img = img[0].unsqueeze(0)
        original_output = model(img)
        original_first_block = original_output[1][2].view(original_output[1][2].size(0), -1) #可修改输出位置
        sensitivity_matrix = np.zeros((img.size(2), img.size(3)))
        #展示原图
        # img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # plt.imshow(img_np)
        # plt.axis("off")
        # plt.title("Original Image")
        # plt.show()
        for i in range(img.size(2)):
            for j in range(img.size(3)):
                noise = torch.zeros(1, 3, 32, 32)
                noise[:, 0, i, j] = torch.randn(1, 1) * 0.1 - 0.05  #可修改在指定通道加噪声
                perturbed_img = img + noise
                with torch.no_grad():
                    perturbed_output = model(perturbed_img)
                perturbed_first_block = perturbed_output[1][2].view(perturbed_output[1][2].size(0), -1) #可修改输出位置
                diff_norm = torch.norm(original_first_block - perturbed_first_block).item()
                sensitivity_matrix[i, j] = diff_norm
        final_sensitivity_matrix = np.zeros((32, 32))
        for i in range(0, 32, 4):
            for j in range(0, 32, 4):
                block = sensitivity_matrix[i:i + 4, j:j + 4]
                avg_value = block.mean()
                final_sensitivity_matrix[i:i + 4, j:j + 4] = avg_value
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        plt.figure(figsize=(8, 6))
        sns.heatmap(final_sensitivity_matrix, annot=False, cmap='coolwarm', cbar=True, square=True)
        plt.title(f"Sensitivity Heatmap for Image {idx}")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        output_image_path = os.path.join(output_dir, f"sensitivity_heatmap_{idx}.png")
        plt.savefig(output_image_path, bbox_inches='tight')
        plt.close()
        print(f"Saved sensitivity heatmap for image {idx} to {output_image_path}")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    indices = [i for i, (_, label) in enumerate(testset) if label == 0]
    filtered_testset = Subset(testset, indices)
    filtered_testloader = DataLoader(filtered_testset, batch_size=1, shuffle=False, num_workers=2)
    output_dir = './RESNET18_2'
    analyze_and_save_heatmap(model, filtered_testloader, output_dir)
