import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class hswish(nn.Module):
    def __init__(self):
        super(hswish, self).__init__()

    def forward(self, x):
        return x * F.relu6(x+3, inplace=True) / 6.0

class hsigmoid(nn.Module):
    def __init__(self):
        super(hsigmoid, self).__init__()
        
    def forward(self, x):
        return F.relu6(x+3, inplace=True) / 6.0

class SE(nn.Module):
    def __init__(self, in_c):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(in_c, in_c//4),
            nn.ReLU(inplace=True),
            nn.Linear(in_c//4, in_c),
            hsigmoid()
        )   

    def forward(self, x):
        b, c, _, _ = x.size()
        se_x = self.avg_pool(x)
        se_x = se_x.view(b, c) 
        se_x = self.fc(se_x)
        se_x = se_x.view(b, c, 1, 1)
        return se_x * x 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
      
class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_s, stride, bn=True, NL="HS"):
        super(conv2d, self).__init__()
        padding_size = int((kernel_s - 1) / 2)

        #----conv2d
        layers = []
        layers += [nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_s, stride=stride, padding=padding_size, bias=False)]

        #-----bn
        if bn:
            layers += [nn.BatchNorm2d(out_c)]

        #-----non linear layer
        if NL=="HS":
            nonlinear_layer = hswish()
            layers += [nonlinear_layer]
        elif NL == "RE":
            nonlinear_layer = nn.ReLU(inplace=True)
            layers += [nonlinear_layer]

        self.layers = nn.Sequential(*layers)        

    def forward(self, x):
        x = self.layers(x)
        return x

class Bneck(nn.Module):
    def __init__(self, in_c, out_c, kernel_s, stride, exp_size, expand_ratio=1, NL='RE', se=False):
        super(Bneck, self).__init__() 
        self.expand_ratio = expand_ratio
        padding_size = int((kernel_s - 1) / 2)
        self.use_res = (stride == 1 and in_c == out_c)

        if NL=="RE":
            nonlinear_layer = nn.ReLU(inplace=True)
        elif NL == "HS":
            nonlinear_layer = hswish()

        #---se module
        if se:
            se_identity_layer = SE(exp_size)
        else:
            se_identity_layer = Identity()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            nonlinear_layer, 

            #----depthwise
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_s, stride=stride, padding=padding_size, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            #---SE module
            se_identity_layer, 
            nonlinear_layer,

            #---pointwise
            nn.Conv2d(exp_size, out_c, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        if self.use_res:
            return x + self.layers(x)
        else:
            return self.layers(x)

class mobileNet_L(nn.Module):
    def __init__(self, n_classes=1000):
        super(mobileNet_L, self).__init__()
        self.layers = nn.Sequential(
            conv2d(in_c=3, out_c=16, kernel_s=3, stride=2, bn=True, NL='HS'), #0
            Bneck(in_c=16, kernel_s=3, exp_size=16,  out_c=16, se=False, NL='RE', stride=1), #1
            Bneck(in_c=16, kernel_s=3, exp_size=64,  out_c=24, se=False, NL='RE', stride=2), #2
            Bneck(in_c=24, kernel_s=3, exp_size=72,  out_c=24, se=False, NL='RE', stride=1), #3
            Bneck(in_c=24, kernel_s=5, exp_size=72,  out_c=40, se=True , NL='RE', stride=2), #4
            Bneck(in_c=40, kernel_s=5, exp_size=120, out_c=40, se=True , NL='RE', stride=1), #5
            Bneck(in_c=40, kernel_s=5, exp_size=120, out_c=40, se=True , NL='RE', stride=1), #6
            Bneck(in_c=40, kernel_s=3, exp_size=240, out_c=80, se=False, NL='HS', stride=2), #7
            Bneck(in_c=80, kernel_s=3, exp_size=200, out_c=80, se=False, NL='HS', stride=1), #8
            Bneck(in_c=80, kernel_s=3, exp_size=184, out_c=80, se=False, NL='HS', stride=1), #9
            Bneck(in_c=80, kernel_s=3, exp_size=184, out_c=80, se=False, NL='HS', stride=1), #10
            Bneck(in_c=80, kernel_s=3, exp_size=480, out_c=112, se=True, NL='HS', stride=1), #11
            Bneck(in_c=112, kernel_s=3, exp_size=672, out_c=112, se=True, NL='HS', stride=1), #12
            Bneck(in_c=112, kernel_s=5, exp_size=672, out_c=160, se=True, NL='HS', stride=2), #13
            Bneck(in_c=160, kernel_s=5, exp_size=960, out_c=160, se=True, NL='HS', stride=1), #14
            Bneck(in_c=160, kernel_s=5, exp_size=960, out_c=160, se=True, NL='HS', stride=1), #15
            conv2d(in_c=160, out_c=960, kernel_s=1, stride=1, bn=True, NL='HS'), #16
        )        
        self.head_layers = nn.Sequential(
            conv2d(in_c=960, out_c=1280, kernel_s=1, stride=1, bn=False, NL='HS'),
            conv2d(in_c=1280, out_c=n_classes, kernel_s=1, stride=1, bn=False, NL='')
        )

    def forward(self, x):
        x = self.layers(x)
        x = F.avg_pool2d(x, kernel_size=7)
        x = self.head_layers(x).squeeze()
        return x

if __name__=="__main__":
    x = torch.rand((1, 3, 224, 224)) 
    net = mobileNet_L(1000) 
    x = net(x)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    print(x.size()) 
