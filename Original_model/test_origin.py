from glob import glob
from model_origin import Fusionmodel
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import os
from torchvision.utils import save_image
from utils import mkdir, load_img, CustomVisionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def test():
    # from channel_fusion import channel_f as channel_fusion
    # from utils import mkdir,Strategy

    # _tensor = transforms.ToTensor()
    # _pil_gray = transforms.ToPILImage()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = 'cuda'

    model = Fusionmodel().to(device)
    checkpoint = torch.load('Original_model/models/1e2/Final_epoch_20__1e2.pkl')
    model.load_state_dict(checkpoint)
    save_dir = "Original_model/output2" 
    mkdir(save_dir)

    root = '/storage/locnx/SampleVisDrone/SampleVisDroneTest'
    sub1 = 'VIS'
    sub2 = 'IR'    
    dataset = CustomVisionDataset(root, sub1, sub2)
    test_dataloader = DataLoader(dataset, shuffle=False)


    # dataset = CustomVisionDataset('Original_test_image', 'VIS', 'IR')
    # test_dataloader = DataLoader(dataset, shuffle=False)

    steps = len(test_dataloader)
    print("step:",steps)

    def test(model):

        tqdms = tqdm(range(int(steps)))
        s_time = time.time()
        imgs_T = iter(test_dataloader)

        for index in tqdms:
            imgs = next(imgs_T)

            vis_img = imgs[0].to(device)
            ir_img = imgs[1].to(device)
            vis_name = imgs[2]
            ir_name = imgs[3]

            out = model(vis_img,ir_img)
            outimage = out[0]

            name = 'addition'

            # print("input shape:", img1.shape)
            # print("output shape:", out.shape)

            e_time = time.time() - s_time
            save_name = save_dir+"/"+name+'/fusion'+str(vis_name)+"-"+str(ir_name)+'.jpg'
            mkdir(save_dir+"/"+name)
            # img_fusion = _pil_gray(out)
            save_image(outimage, save_name)
            print("pic:[%d] %.4fs %s"%(index,e_time,save_name))

    with torch.no_grad():
        test(model)

if __name__ == '__main__':
    print("test initiated")
    test()
    print("success testing")