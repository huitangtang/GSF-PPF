'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.feature_dim = 512
        self.block_expansion = block.expansion
        self.channels = self.feature_dim * self.block_expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)        
        
        # initialize weights
        self.apply(initialize_weights)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)

        return out


class VanillaClassifier(nn.Module):

    def __init__(self, feature_dim, block_expansion, num_classes):
        super(VanillaClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim * block_expansion, num_classes)
        
        # initialize weights
        self.apply(initialize_weights)

    def forward(self, x, mode=None):
        x = self.fc(x)

        return x


class StochasticClassifier(nn.Module):

    def __init__(self, feature_dim, block_expansion, num_classes):
        super(StochasticClassifier, self).__init__()
        self.mean = nn.Linear(feature_dim * block_expansion, num_classes)
        self.std = nn.Linear(feature_dim * block_expansion, num_classes)
        
        # initialize weights
        self.apply(initialize_weights)

    def forward(self, x, mode='train'):
        if mode == 'test':
            weight = self.mean.weight
            bias = self.mean.bias
        else:            
            e_weight = torch.randn(self.mean.weight.data.size()).cuda()
            e_bias = torch.randn(self.mean.bias.data.size()).cuda()
            
            weight = self.mean.weight + self.std.weight * e_weight
            bias = self.mean.bias + self.std.bias * e_bias
        
        out = torch.matmul(x, weight.t()) + bias

        return out


F_GETTERS = {'vanilla': VanillaClassifier, 'stochastic': StochasticClassifier}
        

def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])
    
    
def build_preact_resnet(args, **kwargs):
    print("==> creating model '{}' ".format(args.arch))
    model = {}
    if args.arch == 'resnet18':
        model['G'] = PreActResNet18()
    elif args.arch == 'resnet34':
        model['G'] = PreActResNet34()
    elif args.arch == 'resnet50':
        model['G'] = PreActResNet50()
    elif args.arch == 'resnet101':
        model['G'] = PreActResNet101()
    elif args.arch == 'resnet152':
        model['G'] = PreActResNet152()
    else:
        raise ValueError('Unrecognized model architecture', args.arch)
    model['F'] = F_GETTERS[args.classifier_type](model['G'].feature_dim, model['G'].block_expansion, args.num_classes)
    
    return model