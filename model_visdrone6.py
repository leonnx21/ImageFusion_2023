import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class Encoder0(nn.Module):
    def __init__(self):
        super(Encoder0,self).__init__()
        self.ConvE01 = nn.Conv2d(3, 8, 3, 1, 1)
        self.Activate  = nn.ReLU()
        self.Conv_E02 = nn.Conv2d(8, 4, 3, 1, 1)
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(4,4,3,1,1),
            'DenseConv2': nn.Conv2d(8,4,3,1,1),
            'DenseConv3': nn.Conv2d(12,4,3,1,1)
        })

    def forward(self, x):
        x = self.ConvE01(x)
        x = self.Activate(x)
        x_d = self.Conv_E02(x)
        for i in range(len(self.layers)):
            out = self.Activate(self.layers['DenseConv' + str(i + 1)](x_d))
            x_d = torch.cat([x_d, out], 1)
        # print(x_d.shape)
        return x_d

class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('ConvE11', nn.Conv2d(3, 8, 3, 1 , 1, padding_mode='replicate'))
        self.layers.add_module('MaxpoolE11', nn.MaxPool2d(3, stride=1, padding=1))
        self.layers.add_module('ActE11' , nn.ReLU())
        self.layers.add_module('ConvE12', nn.Conv2d(8, 16, 3, 1, 1, padding_mode='replicate'))
        self.layers.add_module('MaxpoolE12', nn.MaxPool2d(3,stride=1,padding=1))
        self.layers.add_module('ActE12' , nn.ReLU())
        # self.layers.add_module('ConvIF4', nn.Conv2d(64, 128, 3, 1, 1, padding_mode='replicate'))
        # self.layers.add_module('MaxpoolIF4', nn.MaxPool2d(3,stride=1,padding=1))
        # self.layers.add_module('ActIF4' , nn.ReLU())
        # self.layers.add_module('ConvIF5', nn.Conv2d(128, 64, 3, 1, 1, padding_mode='replicate'))
        # self.layers.add_module('MaxpoolIF5', nn.MaxPool2d(3,stride=1,padding=1))
        # self.layers.add_module('ActIF5' , nn.ReLU())

        # self.Upsample = nn.Upsample(scale_factor=1,mode='bilinear',align_corners=True)


    def forward(self, x):
        out = self.layers(x)
        # out = self.Upsample(x)
        return out

class Fusion_method(nn.Module):
    def __init__(self):
        super(Fusion_method,self).__init__()

    def addition(self, vis1, vis2, ir1, ir2):
        vis = (vis1+vis2)/2
        ir = (ir1+ir2)/2
        out = (vis+ir)/2
        return out

    def multiply(self, vis1, vis2, ir1, ir2):
        vis = (vis1*vis2)
        ir = (ir1*ir2)
        out = (vis*ir)
        return out

    def max_fc(self, vis1, vis2, ir1, ir2):
        vis = torch.maximum(vis1, vis2)
        ir = torch.maximum(ir1, ir2)
        out = torch.maximum(vis, ir)
        return out

    def forward(self, vis1, vis2, ir1, ir2):
        out= self.multiply(vis1, vis2, ir1, ir2)
        return out

class Cat_method(nn.Module):
    def __init__(self):
        super(Cat_method,self).__init__()

    def forward(self, vis1, vis2, ir1, ir2):
        out = torch.cat((vis1, vis2, ir1, ir2), 1)
        return out


class Mix_method(nn.Module):
    def __init__(self):
        super(Mix_method,self).__init__()

    def rand_mix_func(self, vis1, vis2, ir1, ir2):
        input = torch.cat((vis1, vis2, ir1, ir2), 1)
        torch.manual_seed(10)
        permute = torch.randperm(256)
        out = input[:, permute]
        out0, out1, out2, out3 = torch.split(out, 32, dim=1)

        return out0, out1, out2, out3

    def channel_shuffle(self, vis1, vis2, ir1, ir2):
        out = torch.cat((vis1, vis2, ir1, ir2), 1)
        batchsize, num_channels, height, width = out.size()
        group = 4
        channels_per_group = num_channels // group

        # reshape
        out = out.view(batchsize, group, channels_per_group, height, width)

        out = torch.transpose(out, 1, 2).contiguous()

        # flatten
        out = out.view(batchsize, num_channels, height, width)
        out0, out1, out2, out3 = torch.split(out, 16, dim=1)

        return out0, out1, out2, out3

    def forward(self, vis1, vis2, ir1, ir2):
        out0, out1, out2, out3 = self.channel_shuffle(vis1, vis2, ir1, ir2)
        return out0, out1, out2, out3


class Silo_fusion(nn.Module):
    def __init__(self):
        super(Silo_fusion, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('ConvE21', nn.Conv2d(16, 8, 3, 1 , 1, padding_mode='replicate'))
        self.layers.add_module('ActE21' , nn.ReLU())
        self.layers.add_module('ConvE22', nn.Conv2d(8, 4, 3, 1 , 1, padding_mode='replicate'))
        self.layers.add_module('ActE22' , nn.ReLU())

    def forward(self, x):
        out = self.layers(x)
        return out

class Decoder0(nn.Module):
    def __init__(self):
        super(Decoder0,self).__init__()
        self.layers = nn.Sequential()
        # self.layers.add_module('ConvD2', nn.Conv2d(64,128,3,1,1))
        # self.layers.add_module('ActD2' , nn.ReLU())
        self.layers.add_module('ConvD3', nn.Conv2d(16,32,3,1,1))
        self.layers.add_module('ActD3' , nn.ReLU())   
        self.layers.add_module('ConvD4', nn.Conv2d(32,16,3,1,1))
        self.layers.add_module('ActD4' , nn.ReLU())     
        self.layers.add_module('ConvD5', nn.Conv2d(16,8,3,1,1))
        self.layers.add_module('ActD5' , nn.ReLU())
        self.layers.add_module('ConvD6', nn.Conv2d(8,3,3,1,1))
        
    def forward(self, x):
        return self.layers(x)

class Fusionmodel(nn.Module):
    
    def __init__(self):
        super(Fusionmodel,self).__init__()
        self.encodervis1 = Encoder0()
        self.encodervis2 = Encoder1()
        self.encoderir1 = Encoder0()
        self.encoderir2 = Encoder1()
        self.mix = Mix_method()
        self.silo_fusion0 = Silo_fusion()
        self.silo_fusion1 = Silo_fusion()
        self.silo_fusion2 = Silo_fusion()
        self.silo_fusion3 = Silo_fusion()
        self.cat = Cat_method()
        self.decoder0 = Decoder0()
        
    def forward(self,x,y):
        vis1 = self.encodervis1(x)
        vis2 = self.encodervis2(x)
        ir1 = self.encoderir1(y)
        ir2 = self.encoderir2(y)
        out0, out1, out2, out3 = self.mix(vis1, vis2, ir1, ir2)
        out0 =  self.silo_fusion0(out0)
        out1 =  self.silo_fusion1(out1)
        out2 =  self.silo_fusion2(out2)
        out3 =  self.silo_fusion3(out3)
        out = self.cat(out0, out1, out2, out3)
        out = self.decoder0(out)
        return out

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Fusionmodel().to('cpu')
    summary(model,input_data=[(3, 256, 256), (3, 256, 256)], col_names=["input_size", "output_size", "num_params", "kernel_size"], depth=4,  device = 'cpu')



