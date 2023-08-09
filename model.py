import torch
import torch.nn as nn
from activate_fuction import new
from torchsummary import summary
import numpy as np
import torch.nn.functional as F

Activate = new

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class Encoder0(nn.Module):
    def __init__(self, input_nc=1):
        super(Encoder0, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.DB1(x)
        return x

#new densefuse implementation
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1,self).__init__()

        self.Conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.Activate  = Activate(inplace=True)
        self.Conv_d = nn.Conv2d(32, 16, 3, 1, 1)
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(16,16,3,1,1),
            'DenseConv2': nn.Conv2d(32,16,3,1,1),
            'DenseConv3': nn.Conv2d(48,16,3,1,1)
        })

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Activate(x)
        x_d = self.Conv_d(x)
        for i in range(len(self.layers)):
            out = self.Activate(self.layers['DenseConv' + str(i + 1)](x_d))
            x_d = torch.cat([x_d, out], 1)

        print(x_d.shape)
        return x_d


class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2,self).__init__()
        self.Conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.Conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.Conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.Conv4 = nn.Conv2d(128, 64, 3, 2, 1)
        self.Upsample = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Upsample(x)
        return x


class Fusion_method(nn.Module):
    def __init__(self):
        super(Fusion_method,self).__init__()
        
    def addition(self, x, y):
        print(x.shape)
        print(y.shape)
        out = (x + y)/2
        return out

    def max(self, x, y):
        out = torch.maximum(x, y)
        return out

    def forward(self, x, y):
        out = self.addition(x,y)
        return out


class Decoder0(nn.Module):
    def __init__(self, output_nc=1):
        super(Decoder0,self).__init__()
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # decoder
        # self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def forward(self, x):
        # x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.conv5(x)

        return output

#new decoder implementation
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv3', nn.Conv2d(64,32,3,1,1))
        self.layers.add_module('Act3' , Activate(inplace=True))
        self.layers.add_module('Conv4', nn.Conv2d(32,16,3,1,1))
        self.layers.add_module('Act4' , Activate(inplace=True))
        self.layers.add_module('Conv5', nn.Conv2d(16,1,3,1,1))

    def forward(self, x):
        return self.layers(x)

class Fusionmodel(nn.Module):
    
    def __init__(self):
        super(Fusionmodel,self).__init__()
        self.encoder1 = Encoder1()
        self.encoder2 = Encoder2()
        self.fusion = Fusion_method()
        self.decoder = Decoder0()
        
    def forward(self,x,y):
        x = self.encoder1(x)
        y = self.encoder2(y)
        out = self.fusion(x,y)
        out = self.decoder(out)
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Fusionmodel().to(device)
    summary(model,input_size=[(1, 128, 128), (1, 128, 128)])

