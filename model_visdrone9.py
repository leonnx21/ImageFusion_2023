import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class Encoder0(nn.Module):
    def __init__(self):
        super(Encoder0,self).__init__()
        self.Conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.Activate  = nn.ReLU()
        self.Conv_d = nn.Conv2d(8, 4, 3, 1, 1)
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(4,4,3,1,1),
            'DenseConv2': nn.Conv2d(8,4,3,1,1),
            'DenseConv3': nn.Conv2d(12,4,3,1,1)
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

class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('ConvIF2', nn.Conv2d(3, 8, 3, 1 , 1, padding_mode='replicate'))
        self.layers.add_module('MaxpoolIF2', nn.MaxPool2d(3, stride=1, padding=1))
        self.layers.add_module('ActIF2' , nn.ReLU())
        self.layers.add_module('ConvIF3', nn.Conv2d(8, 16, 3, 1, 1, padding_mode='replicate'))
        self.layers.add_module('MaxpoolIF3', nn.MaxPool2d(3,stride=1,padding=1))
        self.layers.add_module('ActIF3' , nn.ReLU())
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


# class ResidualBlock(nn.Module):
#     """
#     A residual block, comprising two convolutional blocks with a residual connection across them.
#     """

#     def __init__(self):
#         """
#         :param kernel_size: kernel size
#         :param n_channels: number of input and output channels (same because the input must be added to the output)
#         """
#         super(ResidualBlock, self).__init__()

#         # The first convolutional block
#         self.layers = nn.Sequential()
#         self.layers.add_module('ConvIF2', nn.Conv2d(3, 32, 3, 1 , 1))
#         self.layers.add_module('ActIF2' , nn.ReLU())
#         self.layers.add_module('ConvIF3', nn.Conv2d(32, 64, 3, 1 , 1))
#         self.layers.add_module('ActIF3' , nn.ReLU())


#     def forward(self, input):
#         """
#         Forward propagation.

#         :param input: input images, a tensor of size (N, n_channels, w, h)
#         :return: output images, a tensor of size (N, n_channels, w, h)
#         """
#         residual = input  # (N, n_channels, w, h)
#         output = self.layers(input)  # (N, n_channels, w, h)
#         output = output + residual  # (N, n_channels, w, h)

#         return output


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

    def No_mix_func(self, vis1, vis2, ir1, ir2):
        out = torch.cat((vis1, vis2, ir1, ir2), 1)
        return out

    def rand_mix_func(self, vis1, vis2, ir1, ir2):
        input = torch.cat((vis1, vis2, ir1, ir2), 1)
        torch.manual_seed(10)
        permute = torch.randperm(256)
        out = input[:, permute]
        return out

    def channel_shuffle(self, vis1, vis2, ir1, ir2):
        out = torch.cat((vis1, vis2, ir1, ir2), 1)
        batchsize, num_channels, height, width = out.size()
        group = 64
        channels_per_group = num_channels // group

        # reshape
        out = out.view(batchsize, group, channels_per_group, height, width)

        out = torch.transpose(out, 1, 2).contiguous()

        # flatten
        out = out.view(batchsize, num_channels, height, width)

        return out

    def forward(self, vis1, vis2, ir1, ir2):
        out = self.multiply(vis1, vis2, ir1, ir2)
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
        self.fusion = Fusion_method()
        self.decoder0 = Decoder0()
        
    def forward(self,x,y):
        vis1 = self.encodervis1(x)
        vis2 = self.encodervis2(x)
        ir1 = self.encoderir1(y)
        ir2 = self.encoderir2(y)
        z1 = self.fusion(vis1, vis2, ir1, ir2)
        z2 = self.decoder0(z1)
        return z2

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Fusionmodel().to('cpu')
    summary(model,input_data=[(3, 256, 256), (3, 256, 256)], col_names=["input_size", "output_size", "num_params", "kernel_size"], depth=4,  device = 'cpu')



