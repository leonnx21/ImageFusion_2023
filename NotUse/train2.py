from types import CellType
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from PIL import Image

from model import Fusionmodel
# from ssim import SSIM 
from utils import mkdir, CustomVisionDataset_train
import os
import time
from loss_network import LossNetwork

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure, ErrorRelativeGlobalDimensionlessSynthesis

def train():
    save_name = "Training_"
    train_path = 'New_model/train_COCO_result'
    epochs = 10
    batch_size = 16
    print("epochs",epochs,"batch_size",batch_size)
    device = 'cuda'
    lr = 1e-3
    
    # root = '/storage/locnx/COCO2014'
    # sub1 = 'train2014'
    # sub2 = 'train2014'
    # dataset = CustomVisionDataset_train(root, sub1, sub2)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
    root = '/storage/locnx/COCO2014'
    sub1 = '100files'
    sub2 = '100files'
    dataset = CustomVisionDataset_train(root, sub1, sub2)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
    # root = '/storage/locnx/SampleVisDrone/SampleVisDroneTest2'
    # sub1 = 'VIS'
    # sub2 = 'IR'
    # dataset = CustomVisionDataset_train(root, sub1, sub2)
    # train_dataloader = DataLoader(dataset, shuffle=False)

    # root = '/storage/locnx/VisDrone/train'
    # sub1 = 'trainimg'
    # sub2 = 'trainimgr'
    # dataset = CustomVisionDataset_train(root, sub1, sub2)
    # train_dataloader = DataLoader(dataset, shuffle=False)

    fusion_model = Fusionmodel().to(device)
    print(fusion_model)
    optimizer = optim.Adam(fusion_model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
        verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)


    #Loss functions
    # MSE_fun = nn.MSELoss()
    SSIM_func = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    MS_SSIM_func = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR_func = PeakSignalNoiseRatio(data_range=1.0).to(device)
    # UQI_func = UniversalImageQualityIndex().to(device)
    # EGRAS_func = ErrorRelativeGlobalDimensionlessSynthesis().to(device)
    # CE_loss_func = nn.CrossEntropyLoss(reduction='mean')

    # with torch.no_grad():
    #     loss_network = LossNetwork()
    #     loss_network.to(device)
    # loss_network.eval()


    # Training
 
    mkdir(train_path)
    min_loss = 1000
    print('============ Training Begins [epochs:{}] ==============='.format(epochs))
    steps = len(train_dataloader)
    print("step:",steps)
    steps = steps
    s_time = time.time()
    loss = torch.zeros(1)

    # ssim_los_all1 = []
    # ssim_los_all2 = []
    # ms_ssim_los_all1 = []
    # ms_ssim_los_all2 = []
    # psnr_loss_all1 = []
    # psnr_loss_all2 = []
    # uqi_loss_all1 = []
    # uqi_loss_all2 = []
    # egras_loss_all1 = []
    # egras_loss_all2 = []
    # ce_loss_all1 = []
    # ce_loss_all2 = []

    # loss_all = []

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

            ssim_loss1 = (1 - SSIM_func(img_re, img_vis))*100
            ms_ssim_loss1 = (1- MS_SSIM_func(img_re, img_vis))*100
            psnr_loss1 = -PSNR_func(img_re, img_vis)
            # # uqi_loss1 = 1 - UQI_func(img_re, img_vis)
            # egras_loss1 = EGRAS_func(img_re, img_vis)
            # ce_loss1 = CE_loss_func(img_vis,img_re)*100

            ssim_loss2 = (1 - SSIM_func(img_re, img_ir))*100
            ms_ssim_loss2 = 1- MS_SSIM_func(img_re, img_ir)
            psnr_loss2 = -PSNR_func(img_re, img_ir)
            # # uqi_loss2 = 1 - UQI_func(img_re, img_ir)
            # egras_loss2 = EGRAS_func(img_re, img_ir)
            # ce_loss2 = CE_loss_func(img_vis,img_re)*100

            # ssim_los_all1.append(ssim_loss1.item())
            # ms_ssim_los_all1.append(ms_ssim_loss1.item())
            # psnr_loss_all1.append(psnr_loss1.item())
            # # uqi_loss_all1.append(uqi_loss1.item())
            # egras_loss_all1.append(egras_loss1.item())
            # ce_loss_all1.append(ce_loss1.item())

            # ssim_los_all2.append(ssim_loss2.item())
            # ms_ssim_los_all2.append(ms_ssim_loss2.item())
            # psnr_loss_all2.append(psnr_loss2.item())
            # # uqi_loss_all2.append(uqi_loss2.item())
            # egras_loss_all2.append(egras_loss2.item())
            # ce_loss_all2.append(ce_loss2.item())

            loss = (ssim_loss1 + ssim_loss2 + ms_ssim_loss1 + ms_ssim_loss2 + psnr_loss1 + psnr_loss2)/2

            # loss_all.append(loss_total.item())

            loss.backward()
            optimizer.step()

            e_time = time.time()-s_time

            last_time = epochs*int(steps)*(e_time/(iteration*int(steps)+index+1))-e_time

            ssim_loss_avg = (ssim_loss1 + ssim_loss2)/2
            ms_ssim_loss_avg = (ms_ssim_loss1 + ms_ssim_loss2)/2
            psnr_loss_avg = (psnr_loss1 + psnr_loss2)/2
            # # uqi_loss_avg = (uqi_loss1 + uqi_loss2)/2
            # egras_loss_avg = (egras_loss1 + egras_loss2)/2
            # ce_loss_avg = (ce_loss1 + ce_loss2)/2

            tqdms.set_description('%d Metrics [%.5f = %.5f + %.5f + %.5f] T[%d:%d:%d] lr:%.4f '%
              (iteration, loss.item(), ssim_loss_avg.item(), ms_ssim_loss_avg.item(), psnr_loss_avg.item(), last_time/3600,last_time/60%60,
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