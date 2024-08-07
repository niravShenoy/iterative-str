import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.builder import get_builder
from args import args

# Source: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            dconv = builder.conv1x1(
                inplanes, planes * self.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * self.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out 



class ResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = builder.conv3x3(3, self.in_planes, stride=1)
        self.bn1 = builder.batchnorm(16)
        self.layer1 = self._make_layer(builder, block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(builder, block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 64, num_blocks[2], stride=2)
        self.linear = builder.conv1x1(64, num_classes)


    def _make_layer(self, builder, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        out = self.linear(out).squeeze()
        return out

def ResNet20(input_shape, num_classes):
    return ResNet(get_builder(), BasicBlock, [3, 3, 3], num_classes)


class ResNetWidth(nn.Module):

    def __init__(self, builder, block, num_blocks, width, num_classes=10):
        super(ResNetWidth, self).__init__()
        self.in_planes = width

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = builder.conv3x3(3, width, stride=1)
        self.bn1 = builder.batchnorm(width)
        self.layer1 = self._make_layer(builder, block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(builder, block, width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(builder, block, width, num_blocks[2], stride=2)
        self.linear = builder.conv1x1(width, num_classes)


    def _make_layer(self, builder, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        out = self.linear(out).squeeze()
        return out

# defining a resnet with constant width
def ResNetWidth20(input_shape, num_classes, width=16):
    return ResNetWidth(get_builder(), BasicBlock, [3, 3, 3], width, num_classes)
