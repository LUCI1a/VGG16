from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
import os


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, 4096),  # CIFAR-10 输入尺寸 32x32, Pool 后尺寸 1x1
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        out = out5.view(out5.size(0), -1)
        out = self.classifier(out)
        return out, [out1, out2, out3, out4, out5]

model = VGG16()
model.load_state_dict(torch.load('vgg16.pth'))
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
        original_first_block = original_output[1][3].view(original_output[1][3].size(0), -1)  #更改输出位置
        sensitivity_matrix = np.zeros((img.size(2), img.size(3)))

        for i in range(img.size(2)):
            for j in range(img.size(3)):
                noise = torch.zeros(1, 3, 32, 32)
                noise[:, 0, i, j] = torch.randn(1, 1) * 0.1 - 0.05   #选择颜色通道添加噪声（0red，1green，2blue）
                perturbed_img = img + noise
                with torch.no_grad():
                    perturbed_output = model(perturbed_img)
                perturbed_first_block = perturbed_output[1][3].view(perturbed_output[1][3].size(0), -1) #更改输出位置
                diff_norm = torch.norm(original_first_block - perturbed_first_block).item()
                sensitivity_matrix[i, j] = diff_norm
        final_sensitivity_matrix = np.zeros((32, 32))
        for i in range(0, 32, 4):
            for j in range(0, 32, 4):
                block = sensitivity_matrix[i:i + 4, j:j + 4]
                avg_value = block.mean()
                final_sensitivity_matrix[i:i + 4, j:j + 4] = avg_value
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
    indices = [i for i, (_, label) in enumerate(testset) if label == 1]
    filtered_testset = Subset(testset, indices)
    filtered_testloader = DataLoader(filtered_testset, batch_size=1, shuffle=False, num_workers=2)
    output_dir = './blue_lable0'
    analyze_and_save_heatmap(model, filtered_testloader, output_dir)


