import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['CPNet50', 'CPNet101', 'CPNet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, out_planes=0, stride=1):
        super(Bottleneck, self).__init__()
        out_planes = self.expansion*planes if out_planes == 0 else out_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class CPNet(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, dropout=0.75):
        super(CPNet, self).__init__()
        self.in_planes = 64
        self.classes = num_classes
        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer( 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.lateral1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral4 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.global1 = self._make_global(scale_factor=1)
        self.global2 = self._make_global(scale_factor=0.5)
        self.global3 = self._make_global(scale_factor=0.25)
        self.global4 = self._make_global(scale_factor=0.125)

        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(2048, num_classes)
        self.fc2 = nn.Linear(1024, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride=stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    def _make_global(self, scale_factor):
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.AvgPool2d(7)
        )

    def _upsample_smooth_add(self, x, smooth, y):
        up = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        return smooth(up) + F.relu(y)

    def forward(self, x):
        # Top-down
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c = self.avgpool(c5)
        # Bottom-up
        p5 = self.lateral1(c5)
        p4 = self._upsample_smooth_add(p5, self.smooth1, self.lateral2(c4))
        p3 = self._upsample_smooth_add(p4, self.smooth2, self.lateral3(c3))
        p2 = self._upsample_smooth_add(p3, self.smooth3, self.lateral4(c2))
        # GlobalNet
        g5 = self.global1(p5)
        g4 = self.global2(p4)
        g3 = self.global3(p3)
        g2 = self.global4(p2)
        g = torch.cat([g5,g4,g3,g2], 1)
        
        c = c.view(c.size(0), -1)
        g = g.view(g.size(0), -1)
        if self.dropout > 0:
            c = self.drop1(c)
            g = self.drop2(g)
        c = self.fc1(c)
        g = self.fc2(g)
        return c, g


def CPNet50(pretrained=False, **kargs):
    model =  CPNet([3,4,6,3], **kargs)
    if pretrained == True:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model

def CPNet101(pretrained=False, **kargs):
    model = CPNet([3,4,23,3], **kargs)
    if pretrained == True:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model

def CPNet152(pretrained=False, **kargs):
    model =  CPNet([3,8,36,3], **kargs)
    if pretrained == True:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model


def test():
    net = CPNet50(num_classes=17)
    ys = net(torch.randn(1,3,224,224))
    for y in ys:
        print(y.size())

if __name__ == '__main__':
    test()
