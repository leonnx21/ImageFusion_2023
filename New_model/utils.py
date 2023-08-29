# -*- coding: utf-8 -*-
import string
from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.transforms as transforms
from glob import glob
import os
from PIL import Image
import numpy as np
from pathlib import Path  

from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset

_tensor = transforms.ToTensor()
_pil_rgb = transforms.ToPILImage('RGB')
_pil_gray = transforms.ToPILImage()
device = 'cuda'


def make_dataset(root: str, sub1: str, sub2: str) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rgb_path, gt_path)
    """
    dataset = []

    # Get all the filenames from RGB folder
    vis_fnames = sorted(os.listdir(os.path.join(root, sub1)))
    
    # Compare file names from GT folder to file names from RGB:
    for ir_fname in sorted(os.listdir(os.path.join(root, sub2))):

            if ir_fname in vis_fnames:
                # if we have a match - create pair of full path to the corresponding images
                vis_path = os.path.join(root, sub1, ir_fname)
                ir_path = os.path.join(root, sub2, ir_fname)

                item = (vis_path, ir_path)
                # append to the list dataset
                dataset.append(item)
            else:
                continue

    return dataset


class CustomVisionDataset(VisionDataset):
    def __init__(self,
                 root,
                 subfolder1,
                 subfolder2,
                 loader=default_loader):
        super().__init__(root)

        # Prepare dataset
        samples = make_dataset(self.root, subfolder1, subfolder2)

        self.loader = loader
        self.samples = samples
        # list of RGB images
        self.vis_samples = [s[1] for s in samples]
        # list of GT images
        self.ir_samples = [s[1] for s in samples]
        
        self.transform = transforms.Compose([
            # transforms.CenterCrop(800),
            # transforms.CenterCrop((640,512)),
            transforms.Resize(256),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # getting our paths to images
        vis_path, ir_path = self.samples[index]
        
        # import each image using loader (by default it's PIL)
        vis_sample = self.loader(vis_path)
        ir_sample = self.loader(ir_path)
        
        vis_sample = vis_sample.convert('L')
        ir_sample = ir_sample.convert('L')
        # ir_sample = np.stack((ir_sample,)*3, axis=-1)
        # ir_sample = Image.fromarray(ir_sample.astype('uint8'))

        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        vis_sample = self.transform(vis_sample)
        ir_sample = self.transform(ir_sample)
        
        #get name for sure
        vis_name = Path(vis_path).name
        ir_name = Path(ir_path).name

        # now we return the right imported pair of images (tensors)
        return vis_sample, ir_sample, vis_name, ir_name

    def __len__(self):
        return len(self.samples)


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

# def load_img(img_path, img_type='gray'):
#     img = Image.open(img_path)
#     if img_type=='gray':
#         img = img.convert('L')
#     return _tensor(img).unsqueeze(0)

# def view_img(arr):
#     arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
#     plt.imshow(arr_.T)
#     plt.show()

# class Strategy(nn.Module):
#     def __init__(self, mode='add', window_width=1):
#         super().__init__()
#         self.mode = mode
#         if self.mode == 'l1':
#             self.window_width = window_width

#     def forward(self, y1, y2):
#         if self.mode == 'add':
#             return (y1+y2)/2

#         if self.mode == 'l1':
#             ActivityMap1 = y1.abs()
#             ActivityMap2 = y2.abs()

#             kernel = torch.ones(2*self.window_width+1,2*self.window_width+1)/(2*self.window_width+1)**2
#             kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
#             kernel = kernel.expand(y1.shape[1],y1.shape[1],2*self.window_width+1,2*self.window_width+1)
#             ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=self.window_width)
#             ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=self.window_width)
#             WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
#             WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
#             return WeightMap1*y1+WeightMap2*y2

# def fusion(x1,x2,model,mode='l1', window_width=1):
#     with torch.no_grad():
#         fusion_layer  = Strategy(mode,window_width).to(device)
#         feature1 = model.encoder(x1)
#         feature2 = model.encoder(x2)
#         feature_fusion = fusion_layer(feature1,feature2)
#         return model.decoder(feature_fusion).squeeze(0).detach().cpu()

# class Test:
#     def __init__(self):
#         pass

#     def load_imgs(self, img1_path,img2_path, device):
#         img1 = load_img(img1_path,img_type=self.img_type).to(device)
#         img2 = load_img(img2_path,img_type=self.img_type).to(device)
#         return img1, img2

#     def save_imgs(self, save_path,save_name, img_fusion):
#         mkdir(save_path)
#         save_path = os.path.join(save_path,save_name)
#         img_fusion.save(save_path)

# class test_gray(Test):
#     def __init__(self):
#         self.img_type = 'rgray'

#     def get_fusion(self,img1_path,img2_path,model,
#                    save_path = './test_result/', save_name = 'none', mode='l1',window_width=1):
#         img1, img2 = self.load_imgs(img1_path,img2_path,device)

#         img_fusion = fusion(x1=img1,x2=img2,model=model,mode=mode,window_width=window_width)
#         img_fusion = _pil_gray(img_fusion)

#         self.save_imgs(save_path,save_name, img_fusion)
#         return img_fusion

# class test_rgb(Test):
#     def __init__(self):
#         self.img_type = 'rgb'

#     def get_fusion(self,img1_path,img2_path,model,
#                    save_path = './test_result/', save_name = 'none', mode='l1',window_width=1):
#         img1, img2 = self.load_imgs(img1_path,img2_path,device)

#         img_fusion = _pil_rgb(torch.cat(
#                              [fusion(img1[:,i,:,:][:,None,:,:],
#                              img2[:,i,:,:][:,None,:,:], model,
#                              mode=mode,window_width=window_width)
#                              for i in range(3)],
#                             dim=0))

#         self.save_imgs(save_path,save_name, img_fusion)
#         return img_fusion


# def test(test_path, model, img_type='gray', save_path='./test_result/',mode='l1',window_width=1):
#     img_list = glob(test_path+'*')
#     img_num = len(img_list)/2
#     suffix = img_list[0].split('.')[-1]
#     img_name_list = list(set([img_list[i].split('\\')[-1].split('.')[0].strip(string.digits) for i in range(len(img_list))]))

#     if img_type == 'gray':
#         fusion_phase = test_gray()
#     elif img_type == 'rgb':
#         fusion_phase = test_rgb()

#     for i in range(int(img_num)):
#         img1_path = test_path+img_name_list[0]+str(i+1)+'.'+suffix
#         img2_path = test_path+img_name_list[1]+str(i+1)+'.'+suffix
#         save_name = 'fusion'+str(i+1)+'_'+img_type+'_'+mode+'.'+suffix
#         fusion_phase.get_fusion(img1_path,img2_path,model,
#                    save_path = save_path, save_name = save_name, mode=mode,window_width=window_width)



def gradient(input):
    # conv_op = nn.Conv2d(3,1,3,1,0, bias=False)
    # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # kernel = np.concatenate((kernel, kernel, kernel))
    # kernel = kernel.reshape(1, 3, 3, 3)
    # conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    # edge_detect = conv_op(input)

    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=0)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    kernel = kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    edge_detect = conv_op(input)


    # print(edge_detect.shape)
    # imgs = edge_detect.cpu().detach().numpy()
    # print(img.shape)
    # import cv2
    # for i in range(64):
    #     # print(i,imgs.shape)
    #     img = imgs[i, :, :]
    #     img = img.squeeze()
    #     min = np.amin(img)
    #     max = np.amax(img)
    #     img = (img - min) / (max - min)
    #     img = img * 255
    #     # print(img.shape)
    #
    #     cv2.imwrite('gradient/gradinet' + str(i) + '.jpg', img)

    return edge_detect

# # def contrast(img1):
# #     m, n = img1.shape
# #     img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
# #     rows_ext, cols_ext = img1_ext.shape
# #     b = 0.0
# #     for i in range(1, rows_ext - 1):
# #         for j in range(1, cols_ext - 1):
# #             b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
# #                   (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)
# #
# #     cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)  # 对应上面48的计算公式
# #     return cg


def hist_similar(x,y):
    t_min = torch.min(torch.cat((x, y), 1)).item()
    t_max = torch.max(torch.cat((x, y), 1)).item()
    return (torch.norm((torch.histc(x, 255, min=t_min, max=t_max)-torch.histc(y, 255, min=t_min, max=t_max)),2))/255
