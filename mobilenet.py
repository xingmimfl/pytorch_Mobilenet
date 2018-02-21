import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                relu=True, bn=True, same_padding=False, bias=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DepthwiseSepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, strides):
        super(DepthwiseSepConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=3, stride=strides, groups=in_channels,
                                        padding=1, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=1, stride=1, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)

        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x 

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, width_multiplier=1, Training=False):
        """
        num_classes: number of predicted classes.
        Training: whether or not the model is being trained.
        """
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, same_padding=True),
            DepthwiseSepConv2d(32, 64, 1),
            DepthwiseSepConv2d(64, 128, 2),
            DepthwiseSepConv2d(128, 128, 1),
            DepthwiseSepConv2d(128, 256, 2),
            DepthwiseSepConv2d(256, 256, 1),
            DepthwiseSepConv2d(256, 512, 2),
        
            DepthwiseSepConv2d(512, 512, 1),
            DepthwiseSepConv2d(512, 512, 1),
            DepthwiseSepConv2d(512, 512, 1),
            DepthwiseSepConv2d(512, 512, 1),
            DepthwiseSepConv2d(512, 512, 1),
        
            DepthwiseSepConv2d(512, 1024, 2),
            DepthwiseSepConv2d(1024, 1024, 1),
            nn.AvgPool2d(kernel_size=7)
        )
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        
    def forward(self,x):
        x = self.features(x) 
        #print "x.size():\t", x.size()
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

if __name__=="__main__":
    net = MobileNet()
    print net

