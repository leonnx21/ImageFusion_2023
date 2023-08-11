from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from PIL import Image


from model_origin import Fusionmodel
from ssim import SSIM 
from utils_origin import mkdir, gradient, hist_similar, view_img, CustomVisionDataset
import os
import time
from loss_network import LossNetwork


def train():
    save_name = "Training_"
    train_path = './train_origin_result/'
    epochs = 20
    batch_size = 64
    print("epochs",epochs,"batch_size",batch_size)
    device = 'cuda'
    lr = 1e-3

    dataset = CustomVisionDataset('/storage/locnx/train2014')
    train_dataloader = DataLoader(dataset, shuffle=False)

    fusion_model = Fusionmodel().to(device)
    print(fusion_model)
    optimizer = optim.Adam(fusion_model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
        verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)

    MSE_fun = nn.MSELoss()
    SSIM_fun = SSIM()
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

    grd_loss_all = []
    hist_loss_all = []
    mse_loss_all = []
    perceptual_loss_all = []

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

            mse_loss = MSE_fun(img_ir,img_re)
            grd_loss = MSE_fun(gradient(img_ir), gradient(img_re))
            hist_loss = hist_similar(img_ir, img_re.detach()) * 0.001
            std_loss = torch.abs(img_re.std() - img_ir.std())
            std_loss = hist_loss

            with torch.no_grad():
                x = img_ir.detach()
            features = loss_network(x)
            features_re = loss_network(img_re)

            with torch.no_grad():
                f_x_vi1 = features[1].detach()
                f_x_vi2 = features[2].detach()
                f_x_ir3 = features[3].detach()
                f_x_ir4 = features[4].detach()

            perceptual_loss = MSE_fun(features_re[1], f_x_vi1)+MSE_fun(features_re[2], f_x_vi2) + \
                             MSE_fun(features_re[3], f_x_ir3)+MSE_fun(features_re[4], f_x_ir4)

            std_loss = std_loss
            perceptual_loss = perceptual_loss*1000

            grd_loss_all.append(grd_loss.item())
            hist_loss_all.append(hist_loss.item())
            mse_loss_all.append(mse_loss.item())
            perceptual_loss_all.append(perceptual_loss.item())

            loss = mse_loss +grd_loss + std_loss+ perceptual_loss
            loss.backward()
            optimizer.step()

            e_time = time.time()-s_time
            last_time = epochs*int(steps)*(e_time/(iteration*int(steps)+index+1))-e_time

            tqdms.set_description('%d MSGP[%.5f %.5f %.5f %.5f] T[%d:%d:%d] lr:%.4f '%
              (iteration,mse_loss.item(),std_loss.item(),grd_loss.item(),perceptual_loss.item(),last_time/3600,last_time/60%60,
               last_time%60,optimizer.param_groups[0]['lr']*1000))

        if min_loss > loss.item():
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