import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

class FP_ResNet(nn.Module):
    def __init__(self, num_class, base_model='resnet101', dropout=0.8):
        super(FP_ResNet, self).__init__()
        self.dropout = dropout

        self._prepare_base_model(base_model)
        self._prepare_fp(num_class)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # self.avgpool0 = nn.AvgPool2d(7)
        self.avgpool1 = nn.AvgPool2d(7)
        self.avgpool2 = nn.AvgPool2d(7)
        # self.avgpool3 = nn.AvgPool2d(7)

    def _prepare_base_model(self, base_model):
        if 'resnet' not in base_model:
            import torchvision.models as models
            model_names = sorted( name for name in models.__dict__ if \
                                    name.islower() and \
                                    not name.startswith("__") and \
                                    callable(models.__dict__[name]) and \
                                    'resnet' in name)
            raise ValueError('Unknown base_model. supported: {}'.format(' | '.join(model_names)))
        
        self.base_model = getattr(torchvision.models, base_model)(True)

    def _prepare_fp(self, num_class):
        feature_dim = 5120 #self.base_model.fc.in_features
        if self.dropout == 0:
            self.base_model.fc = nn.Linear(feature_dim, num_class)
        else:
            self.base_model.fc = nn.Dropout(p=self.dropout)
            self.new_fc = nn.Linear(feature_dim, num_class)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def _interpolate(self, x):
        return F.interpolate(x, [7,7], mode='bilinear')

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        c2 = self.base_model.layer1(x)
        c3 = self.base_model.layer2(c2)
        c4 = self.base_model.layer3(c3)
        c5 = self.base_model.layer4(c4)
        
        p5 = self.toplayer(c5)                              # 256*7*7
        p4 = self._upsample_add(p5, self.latlayer1(c4))     # 256*14*14
        p3 = self._upsample_add(p4, self.latlayer2(c3))     # 256*28*28
        p2 = self._upsample_add(p3, self.latlayer3(c2))     # 256*56*56
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # logit5 = self.avgpool0(p5)
        # logit5 = logit5.view(logit5.size(0), -1)
        logit4 = self.avgpool1(p4)
        logit4 = logit4.view(logit4.size(0), -1)
        logit3 = self.avgpool2(p3)
        logit3 = logit3.view(logit3.size(0), -1)
        # logit2 = self.avgpool3(p2)
        # logit2 = logit2.view(logit2.size(0), -1)
        
        x = torch.cat([logit4, logit3], dim=1)
        x = self.base_model.fc(x)
        if self.dropout > 0:
            x = self.new_fc(x)

        return x

def test():
    net = FP_ResNet(101)
    output = net(Variable(torch.randn(10,3,224,224)))
    print(output.size())


if __name__ == '__main__':
   test()
