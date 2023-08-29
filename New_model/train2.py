from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from PIL import Image

from model import Fusionmodel
from ssim import SSIM 
from utils import mkdir, gradient, hist_similar, CustomVisionDataset
import os
import time
from loss_network import LossNetwork
from pytorch_msssim import ms_ssim as msssim


def train():
    save_name = "Training_"
    train_path = 'New_model/train_model_retest_result'
    epochs = 10
    batch_size = 128
    print("epochs",epochs,"batch_size",batch_size)
    device = 'cuda'
    lr = 1e-3
    
    root = '/storage/locnx/COCO2014'
    sub1 = '100files'
    sub2 = '100files'
    dataset = CustomVisionDataset(root, sub1, sub2)
    train_dataloader = DataLoader(dataset, shuffle=False)
 
    # root = '/storage/locnx/SampleVisDrone/SampleVisDroneTest2'
    # sub1 = 'VIS'
    # sub2 = 'IR'
    # dataset = CustomVisionDataset(root, sub1, sub2)
    # train_dataloader = DataLoader(dataset, shuffle=False)

    # root = '/storage/locnx/VisDrone/train'
    # sub1 = 'trainimg'
    # sub2 = 'trainimgr'
    # dataset = CustomVisionDataset(root, sub1, sub2)
    # train_dataloader = DataLoader(dataset, shuffle=False)
	


    fusion_model = Fusionmodel().to(device)
    print(fusion_model)
    optimizer = optim.Adam(fusion_model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
        verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)

    MSE_fun = nn.MSELoss()
    SSIM_fun = msssim
    CrossEntropyLoss = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss_network = LossNetwork()
        loss_network.to(device)
    loss_network.eval()


    # Training
    mse_train = []
    ssim_train = []
    loss_train = []
    mse_val = []
    ssim_val = []
    loss_val = []
    gradient_train = []
    mkdir(train_path)
    min_loss = 100
    print('============ Training Begins [epochs:{}] ==============='.format(epochs))
    steps = len(train_dataloader)
    print("step:",steps)
    steps = steps
    s_time = time.time()
    loss = torch.zeros(1)

    msssim_los_all1 = []
    grd_loss_all1 = []
    hist_loss_all1 = []
    mse_loss_all1 = []
    std_loss_all1 = []
    # perceptual_loss_all1 = []

    msssim_los_all2 = []
    grd_loss_all2 = []
    hist_loss_all2 = []
    mse_loss_all2 = []
    std_loss_all2 = []
    # perceptual_loss_all2 = []

    for iteration in range(epochs):

        scheduler.step(loss.item())
        imgs_T = iter(train_dataloader)

        tqdms = tqdm(range(int(steps)))

        for index in tqdms:
            img_vis_ir = next(imgs_T)
            img_vis = img_vis_ir[0].to(device)
            img_ir = img_vis_ir[1].to(device)
            
            optimizer.zero_grad()

            img_re = fusion_model(img_vis, img_ir)

            ssim_loss1 = 1 - SSIM_fun(img_vis, img_re)
            mse_loss1 = MSE_fun(img_vis,img_re)
            grd_loss1 = MSE_fun(gradient(img_vis), gradient(img_re))
            hist_loss1 = hist_similar(img_vis, img_re.detach()) * 0.001
            std_loss1 = torch.abs(img_re.std() - img_vis.std())
            # std_loss1 = hist_loss1

            ssim_loss2 = 1 - SSIM_fun(img_ir, img_re)
            mse_loss2 = MSE_fun(img_ir,img_re)
            grd_loss2 = MSE_fun(gradient(img_ir), gradient(img_re))
            hist_loss2 = hist_similar(img_ir, img_re.detach()) * 0.001
            std_loss2 = torch.abs(img_re.std() - img_ir.std())
            # std_loss2 = hist_loss2


            # with torch.no_grad():
            #     x = img_vis.detach()
            #     features1 = loss_network(x)
            #     features_re1 = loss_network(img_re)

            #     y = img_ir.detach()
            #     features2 = loss_network(y)
            #     features_re2 = loss_network(img_re)

            # with torch.no_grad():
            #     f_x_vi11 = features1[1].detach()
            #     f_x_vi21 = features1[2].detach()
            #     f_x_ir31 = features1[3].detach()
            #     f_x_ir41 = features1[4].detach()

            #     f_x_vi12 = features2[1].detach()
            #     f_x_vi22 = features2[2].detach()
            #     f_x_ir32 = features2[3].detach()
            #     f_x_ir42 = features2[4].detach()

            # perceptual_loss1 = MSE_fun(features_re1[1], f_x_vi11)+MSE_fun(features_re1[2], f_x_vi21) + \
            #                  MSE_fun(features_re1[3], f_x_ir31)+MSE_fun(features_re1[4], f_x_ir41)
            
            # perceptual_loss2 = MSE_fun(features_re2[1], f_x_vi12)+MSE_fun(features_re2[2], f_x_vi22) + \
            #                  MSE_fun(features_re2[3], f_x_ir32)+MSE_fun(features_re2[4], f_x_ir42)

            # std_loss1 = std_loss1
            # perceptual_loss1 = perceptual_loss1*1000

            # std_loss2 = std_loss2
            # perceptual_loss2 = perceptual_loss2*1000

            msssim_los_all1.append(ssim_loss1.item())
            grd_loss_all1.append(grd_loss1.item())
            hist_loss_all1.append(hist_loss1.item())
            mse_loss_all1.append(mse_loss1.item())
            std_loss_all1.append(std_loss1.item())
            # perceptual_loss_all1.append(perceptual_loss1.item())

            msssim_los_all2.append(ssim_loss2.item())
            grd_loss_all2.append(grd_loss2.item())
            hist_loss_all2.append(hist_loss2.item())
            mse_loss_all2.append(mse_loss2.item())
            std_loss_all2.append(std_loss2.item())
            # perceptual_loss_all2.append(perceptual_loss1.item())


            loss1 = ssim_loss1 #+ mse_loss1 + grd_loss1 + std_loss1 #+ perceptual_loss1
            loss2 = ssim_loss2 #+ mse_loss2 + grd_loss2 + std_loss2 #+ perceptual_loss2
            loss = (loss1 + loss2)/2

            loss.backward()
            optimizer.step()

            e_time = time.time()-s_time
            last_time = epochs*int(steps)*(e_time/(iteration*int(steps)+index+1))-e_time

            msssim_loss = (ssim_loss2 + ssim_loss2)/2
            mse_loss = (mse_loss1 + mse_loss2)/2
            std_loss = (std_loss1 + std_loss2)/2
            grd_loss = (grd_loss1 + grd_loss2)/2
            # perceptual_loss = (perceptual_loss1 + perceptual_loss2)/2

            tqdms.set_description('%d MSGP[%.5f %.5f %.5f %.5f] T[%d:%d:%d] lr:%.4f '%
              (iteration,msssim_loss.item(),mse_loss.item(),std_loss.item(),grd_loss.item(),last_time/3600,last_time/60%60,
               last_time%60,optimizer.param_groups[0]['lr']*1000))

        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save({'weight': fusion_model.state_dict(), 'epoch': iteration, 'batch_index': index},
                    os.path.join(train_path, save_name+'best.pkl'))
            print('[%d] - Best model is saved -' % (iteration))

        if (iteration+1) % 10 ==0 and iteration != 0:
            torch.save( {'weight': fusion_model.state_dict(), 'epoch':iteration, 'batch_index': index},
                       os.path.join(train_path,save_name+'model_weight_new.pkl'))
            print('[%d] - model is saved -'%(iteration))

if __name__ == '__main__':
    train()
    print("success training")