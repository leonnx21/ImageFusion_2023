from glob import glob
from model import Fusionmodel

import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import os
from torchvision.utils import save_image
from utils import mkdir, CustomVisionDataset, gradient, hist_similar
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss_network import LossNetwork
from pytorch_msssim import ms_ssim as msssim
import torch.nn as nn


def test():
    # from channel_fusion import channel_f as channel_fusion
    # from utils import mkdir,Strategy 
    #testing remote code

    # _tensor = transforms.ToTensor()
    # _pil_gray = transforms.ToPILImage()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda'

    model = Fusionmodel().to(device)
    checkpoint = torch.load('New_model/train_model_retest_result/Training_best.pkl')
    model.load_state_dict(checkpoint['weight'])

    root = 'New_model/Original_test_image'
    sub1 = 'VIS'
    sub2 = 'IR'    
    dataset = CustomVisionDataset(root, sub1, sub2)
    test_dataloader = DataLoader(dataset, shuffle=False)

    # root = '/storage/locnx/SampleVisDrone/SampleVisDroneTest'
    # sub1 = 'VIS'
    # sub2 = 'IR'    
    # dataset = CustomVisionDataset(root, sub1, sub2)
    # test_dataloader = DataLoader(dataset, shuffle=False)

    # root = '/storage/locnx/SampleVisDrone/SampleVisDroneTest2'
    # sub1 = 'VIS'
    # sub2 = 'IR'    
    # dataset = CustomVisionDataset(root, sub1, sub2)
    # test_dataloader = DataLoader(dataset, shuffle=False)

    save_dir = "New_model/output_final" 
    mkdir(save_dir)

    steps = len(test_dataloader)
    print("step:",steps)

    MSE_fun = nn.MSELoss()
    SSIM_fun = msssim
    CrossEntropyLoss = nn.CrossEntropyLoss()

    with torch.no_grad():
        loss_network = LossNetwork()
        loss_network.to(device)
    loss_network.eval()


    def test(model):

        tqdms = tqdm(range(int(steps)))
        s_time = time.time()
        imgs_T = iter(test_dataloader)

        for index in tqdms:
            imgs = next(imgs_T)

            img_vis = imgs[0].to(device)
            img_ir = imgs[1].to(device)
            vis_name = imgs[2]
            ir_name = imgs[3]

            outimage = model(img_vis,img_ir)
            outimage2 = outimage[0]

            name = 'addition'

            # print("input shape:", img1.shape)
            # print("output shape:", out.shape)
           
            ssim_loss1 = 1 - SSIM_fun(img_vis, outimage)
            mse_loss1 = MSE_fun(img_vis,outimage)
            grd_loss1 = MSE_fun(gradient(img_vis), gradient(outimage))
            hist_loss1 = hist_similar(img_vis, outimage.detach()) * 0.001
            std_loss1 = torch.abs(outimage.std() - img_vis.std())
            entropy1 = CrossEntropyLoss(img_vis, outimage)

            ssim_loss2 = 1 - SSIM_fun(img_ir, outimage)
            mse_loss2 = MSE_fun(img_ir,outimage)
            grd_loss2 = MSE_fun(gradient(img_ir), gradient(outimage))
            hist_loss2 = hist_similar(img_ir, outimage.detach()) * 0.001
            std_loss2 = torch.abs(outimage.std() - img_ir.std())
            entropy2 = CrossEntropyLoss(img_ir, outimage)


            msssim_loss = (ssim_loss2 + ssim_loss2)/2
            mse_loss = (mse_loss1 + mse_loss2)/2
            std_loss = (std_loss1 + std_loss2)/2
            grd_loss = (grd_loss1 + grd_loss2)/2
            ent_loss = (entropy1 + entropy2)/2

            e_time = time.time() - s_time
            save_name = save_dir+'/'+name+'/fusion'+str(vis_name)+"-"+str(ir_name)+'.jpg'
            mkdir(save_dir+'/'+name)
            # img_fusion = _pil_gray(out)
            save_image(outimage2, save_name)
            print("pic:[%d] %.4fs %s"%(index,e_time,save_name))
            print('stat":[%.5f %.5f %.5f %.5f]' % (msssim_loss, mse_loss, std_loss, grd_loss))

    with torch.no_grad():
        test(model)

if __name__ == '__main__':
    print("test initiated")
    test()
    print("success testing")