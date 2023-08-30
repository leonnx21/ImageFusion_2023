import torch
import torch.nn as nn
# from activate_fuction import new
from torchsummary import summary
import numpy as np
import torch.nn.functional as F

import functools
import torch.nn.functional as F
# import common

# Activate = new

# # Convolution operation
# class ConvLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
#         super(ConvLayer, self).__init__()
#         reflection_padding = int(np.floor(kernel_size / 2))
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#         self.dropout = nn.Dropout2d(p=0.5)
#         self.is_last = is_last

#     def forward(self, x):
#         out = self.reflection_pad(x)
#         out = self.conv2d(out)
#         if self.is_last is False:
#             # out = F.normalize(out)
#             out = F.relu(out, inplace=True)
#             # out = self.dropout(out)
#         return out


# # Dense convolution unit
# class DenseConv2d(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(DenseConv2d, self).__init__()
#         self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

#     def forward(self, x):
#         out = self.dense_conv(x)
#         out = torch.cat([x, out], 1)
#         return out


# # Dense Block unit
# class DenseBlock(torch.nn.Module):
#     def __init__(self, in_channels, kernel_size, stride):
#         super(DenseBlock, self).__init__()
#         out_channels_def = 16
#         denseblock = []
#         denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
#                        DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
#                        DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
#         self.denseblock = nn.Sequential(*denseblock)

#     def forward(self, x):
#         out = self.denseblock(x)
#         return out


# class Encoder0(nn.Module):
#     def __init__(self):
#         super(Encoder0, self).__init__()
#         denseblock = DenseBlock
#         nb_filter = [16, 64, 32, 16]
#         kernel_size = 3
#         stride = 1

#         # encoder
#         self.conv1 = ConvLayer(3, nb_filter[0], kernel_size, stride)
#         self.DB1 = denseblock(nb_filter[0], kernel_size, stride)
#         # self.conv2 = nn.Conv2d(64,1,3,1,1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.DB1(x)
#         # x = self.conv2(x)
#         return x

#new densefuse implementation
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1,self).__init__()
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


# class Encoder2(nn.Module):
#     def __init__(self):
#         super(Encoder2,self).__init__()
#         self.Conv1 = nn.Conv2d(3, 32, 3, 1 , 1)
#         self.Conv2 = nn.Conv2d(32, 64, 3, 1, 1)
#         self.Conv3 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.Conv4 = nn.Conv2d(128, 64, 3, 1, 1)
#         # self.Upsample = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)

#     def forward(self, x):
#         x = self.Conv1(x)
#         x = self.Conv2(x)
#         x = self.Conv3(x)
#         x = self.Conv4(x)
#         # x = self.Upsample(x)
#         return x

# class Convsharpen(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
#         super(Convsharpen, self).__init__()
#         self.out_channels = out_channels
#         self.conv_op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding )

#     def kernel_func(self):
#         kernel = np.array([], dtype='float32')
#         # for i in range(out_channels):
#         #     kernel = np.append(kernel, [[-1, -1, -1], [-1, i, -1], [-1, -1, -1]])

#         # kernel = kernel.reshape((1, out_channels, 3, 3))
#         kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
#         kernel = kernel.reshape((1, 1, 3, 3))
#         kernel = torch.tensor(kernel)

#         return kernel

#     def forward(self, x):
#         kernel = self.kernel_func()
    
#         with torch.no_grad():
#             self.conv_op.weight = nn.Parameter(kernel)

#         out = self.conv_op(x)
#         return out


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

# class Encoder5(nn.Module):
#     def __init__(self):
#         super(Encoder5,self).__init__()
#         self.layers = nn.Sequential()
#         self.layers.add_module('Conv2', nn.Conv2d(3, 32, 3, 1 , 1))
#         self.layers.add_module('Act2' , Activate())
#         self.layers.add_module('Conv3', nn.Conv2d(32, 64, 3, 1, 1))
#         self.layers.add_module('Act3' , Activate())
#         self.layers.add_module('Conv4', nn.Conv2d(64, 128, 3, 1, 1))
#         self.layers.add_module('Act4' , Activate())
#         self.layers.add_module('Conv5', nn.Conv2d(128, 64, 3, 1, 1))
#         self.layers.add_module('Act5' , Activate())

#     def forward(self, x):
#         return self.layers(x)


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


# class Decoder0(nn.Module):
#     def __init__(self, output_nc=3):
#         super(Decoder0,self).__init__()
#         nb_filter = [3, 64, 32, 16]
#         kernel_size = 3
#         stride = 1

#         # decoder
#         self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
#         self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
#         self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
#         self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

#     def forward(self, x):
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         output = self.conv5(x)

#         return output


#new decoder implementation
# class Decoder1(nn.Module):
#     def __init__(self):
#         super(Decoder1,self).__init__()
#         self.layers = nn.Sequential()
#         self.layers.add_module('ConvD2', nn.Conv2d(64,64,3,1,1))
#         self.layers.add_module('ActD2' , Activate())
#         self.layers.add_module('ConvD3', nn.Conv2d(64,32,3,1,1))
#         self.layers.add_module('ActD3' , Activate())
#         self.layers.add_module('ConvD4', nn.Conv2d(32,16,3,1,1))
#         self.layers.add_module('ActD4' , Activate())
#         self.layers.add_module('ConvD5', nn.Conv2d(16,3,3,1,1))

#     def forward(self, x):
#         return self.layers(x)


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2,self).__init__()
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

#============================================================================


# class EDSR(nn.Module):
#     def __init__(self, conv=common.default_conv):
#         super(EDSR, self).__init__()

#         n_resblock = 16
#         n_feats = 64
#         kernel_size = 3 
#         scale = 1
#         act = nn.ReLU(True)
#         rgb_range = 255
#         res_scale = 1
#         n_colors = 3

#         rgb_mean = (0.4488, 0.4371, 0.4040)
#         rgb_std = (1.0, 1.0, 1.0)
#         self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        
#         # define head module
#         m_head = [conv(n_colors, n_feats, kernel_size)]

#         # define body module
#         m_body = [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=res_scale
#             ) for _ in range(n_resblock)
#         ]
#         m_body.append(conv(n_feats, n_feats, kernel_size))

#         # define tail module
#         m_tail = [
#             common.Upsampler(conv, scale, n_feats, act=False),
#             nn.Conv2d(
#                 n_feats, n_colors, kernel_size,
#                 padding=(kernel_size//2)
#             )
#         ]

#         self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

#         self.head = nn.Sequential(*m_head)
#         self.body = nn.Sequential(*m_body)
#         self.tail = nn.Sequential(*m_tail)

#     def forward(self, x):
#         x = self.sub_mean(x)
#         x = self.head(x)

#         res = self.body(x)
#         res += x

#         x = self.tail(res)
#         x = self.add_mean(x)

#         return x 


#============================================================================

class Fusionmodel(nn.Module):
    
    def __init__(self):
        super(Fusionmodel,self).__init__()
        self.encoder0 = Encoder1()
        self.encodervis = EncoderVIS()
        self.encoderif = EncoderIF()
        self.fusion = Fusion_method()
        self.decoder1 = Decoder2()
        
    def forward(self,x,y):
        vis1 = self.encoder0(x)
        vis2 = self.encodervis(x)
        ir1 = self.encoder0(y)
        ir2 = self.encoderif(y)

        # z1 = self.fusion(ir2)
        z1 = self.fusion(vis1, vis2, ir1, ir2)
        z2 = self.decoder1(z1)
        return z2


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = Fusionmodel().to(device)
    summary(model,input_size=[(3, 128, 128), (3, 128, 128)])

