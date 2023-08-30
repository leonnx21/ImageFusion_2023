import torch
import torch.nn as nn
# from activate_fuction import new
from torchsummary import summary
import torch.nn.functional as F

#new densefuse implementation
class Encoder0(nn.Module):
    def __init__(self):
        super(Encoder0,self).__init__()
        self.Conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.Activate  = nn.ReLU()
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
        # print(x_d.shape)
        return x_d


class EncoderIF(nn.Module):
    def __init__(self):
        super(EncoderIF,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('ConvIF2', nn.Conv2d(3, 32, 3, 1 , 1))
        self.layers.add_module('MaxpoolIF2', nn.MaxPool2d(4, stride=1, padding=2))
        self.layers.add_module('ActIF2' , nn.ReLU())
        self.layers.add_module('ConvIF3', nn.Conv2d(32, 64, 3, 1, 1))
        self.layers.add_module('MaxpoolIF3', nn.MaxPool2d(4,stride=1,padding=2))
        self.layers.add_module('ActIF3' , nn.ReLU())
        self.layers.add_module('ConvIF4', nn.Conv2d(64, 128, 3, 1, 1))
        self.layers.add_module('MaxpoolIF4', nn.MaxPool2d(4,stride=1,padding=1))
        self.layers.add_module('ActIF4' , nn.ReLU())
        self.layers.add_module('ConvIF5', nn.Conv2d(128, 1, 3, 1, 1))
        self.layers.add_module('MaxpoolIF5', nn.MaxPool2d(4,stride=1,padding=1))
        self.layers.add_module('ActIF5' , nn.ReLU())

    def forward(self, x):
        return self.layers(x)



class EncoderVIS(nn.Module):
    def __init__(self):
        super(EncoderVIS,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('ConvVI2', nn.Conv2d(3, 32, 3, 1 , 1))
        self.layers.add_module('MaxpoolVI2', nn.MaxPool2d(4, stride=1, padding=2))
        self.layers.add_module('ActVI2' , nn.ReLU())

        self.layers.add_module('ConvVI3', nn.Conv2d(32, 64, 3, 1, 1))
        self.layers.add_module('MaxpoolVI3', nn.MaxPool2d(4,stride=1,padding=2))
        self.layers.add_module('ActVI3' , nn.ReLU())

        self.layers.add_module('ConvVI4', nn.Conv2d(64, 128, 3, 1, 1))
        self.layers.add_module('MaxpoolVI4', nn.MaxPool2d(4,stride=1,padding=1))
        self.layers.add_module('ActVI4' , nn.ReLU())

        self.layers.add_module('ConvVI5', nn.Conv2d(128, 1, 3, 1, 1))
        self.layers.add_module('MaxpoolVI5', nn.MaxPool2d(4,stride=1,padding=1))
        self.layers.add_module('ActVI5' , nn.ReLU())

    def forward(self, x):
        return self.layers(x)

class Fusion_method(nn.Module):
    def __init__(self):
        super(Fusion_method,self).__init__()
        
    def addition(self, vis1, vis2, ir1, ir2):
        vis = (vis1 + vis2)/2
        ir = (ir1 + ir2)/2
        out = (ir + vis)/2
        return out

    def max_fc(self, vis1, vis2, ir1, ir2):
        vis = torch.maximum(vis1, vis2)
        ir = torch.maximum(ir1, ir2)
        out = (vis+ir)/2

        return out

    def forward(self, vis1, vis2, ir1, ir2):
        out = self.max_fc(vis1, vis2, ir1, ir2)
        return out

class No_fusion(nn.Module):
    def __init__(self):
        super(No_fusion,self).__init__()
        
    def keep(self, x):
        # print(x.shape)
        # print(y.shape)
        return x

    def forward(self, x):
        return x

class Decoder0(nn.Module):
    def __init__(self):
        super(Decoder0,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('ConvD2', nn.Conv2d(64,64,3,1,1))
        self.layers.add_module('ActD2' , nn.ReLU())
        self.layers.add_module('ConvD3', nn.Conv2d(64,32,3,1,1))
        self.layers.add_module('ActD3' , nn.ReLU())        
        self.layers.add_module('ConvD4', nn.Conv2d(32,16,3,1,1))
        self.layers.add_module('ActD4' , nn.ReLU())
        self.layers.add_module('ConvD5', nn.Conv2d(16,3,3,1,1))
        
    def forward(self, x):
        return self.layers(x)

class Fusionmodel(nn.Module):
    
    def __init__(self):
        super(Fusionmodel,self).__init__()
        self.encoder0 = Encoder0()
        self.encodervis = EncoderVIS()
        self.encoderif = EncoderIF()
        self.fusion = Fusion_method()
        self.decoder0 = Decoder0()
        
    def forward(self,x,y):
        vis1 = self.encoder0(x)
        vis2 = self.encodervis(x)
        ir1 = self.encoder0(y)
        ir2 = self.encoderif(y)
        z1 = self.fusion(vis1, vis2, ir1, ir2)
        z2 = self.decoder0(z1)
        return z2

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Fusionmodel().to(device)
    summary(model,input_size=[(3, 128, 128), (3, 128, 128)])

