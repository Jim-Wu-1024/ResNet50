import torch
import torch.nn as nn

from typing import Optional


class Bottleneck(nn.Module):
    # Expansion factor for the Bottleneck structure
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 ):
        super().__init__()
        # 1x1 Conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 Conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 Conv (Dimensions Expansion)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # Save input for the residual connection

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # Residual connection
        x += identity
        x = self.relu(x)
        return x
    

class ResNet(nn.Module):  # layers: [3, 4, 6, 3] --> ResNet50
    def __init__(self, block: Bottleneck, layers: list, image_channels: int, num_classes: int):
        super().__init__() 
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Maxpooling 
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
    
    def forward(self, x):
        # input: [bs, 3, 224, 224]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # [bs, 64, 112, 112]
        x = self.pool(x)  # [bs, 64, 56, 56]

        x = self.layer1(x)  # [bs, 256, 56, 56]
        x = self.layer2(x)  # [bs, 512, 28, 28]
        x = self.layer3(x)  # [bs, 1024, 14, 14]
        x = self.layer4(x)  # [bs, 2048, 7, 7]

        x = self.avgpool(x)  # [bs, 2048, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # [bs, 1000]
        return x
    
    def _make_layer(self, block: Bottleneck, num_residual_blocks: int, out_channels: int, stride: int):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                    stride=stride),
                                    nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * 4

        for _ in range(1, num_residual_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def build_ResNet50(image_channels=3, num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6,3], image_channels, num_classes)


if __name__ == '__main__':
    input = torch.randn(2, 3, 244, 244)
    model = build_ResNet50()
    output = model(input)
    print(output.shape)