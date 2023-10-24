import torch
from utils import CustomVisionDataset_test, mkdir
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np
import csv

def loader_function(root, sub1, sub2):
    Data_set = CustomVisionDataset_test(root, sub1, sub2)
    Loader = torch.utils.data.DataLoader(Data_set, shuffle=False)
    return Loader

def test_fusion(name, model, model_path, benchmark_function, test_dataloader,save_dir, device):
    
    #load model
    model.load_state_dict(torch.load(model_path,map_location=device))

    model.eval()

    benchmark_list = []

    with torch.no_grad():
        for i, tdata in enumerate(test_dataloader):
            
            vis_image, ir_image, vis_name, ir_name = tdata
            vis_image = vis_image.to(device)
            ir_image = ir_image.to(device)

            toutputs = model(vis_image, ir_image)

            # toutputs_gray =  np.mean(toutputs.squeeze(0).detach().cpu().numpy(), axis=0)
            # toutputs_gray = torch.from_numpy(toutputs_gray)
            # print(toutputs_gray.size())

            #report loss        
            ssim_avg, ms_ssim_avg, psnr_avg, egras_avg = benchmark_function(vis_image, ir_image, toutputs, device)
            
            benchmark_list.append([i, name, vis_name, ssim_avg.item(), ms_ssim_avg.item(), psnr_avg.item(), egras_avg.item()])

            print(i, name, vis_name, ssim_avg.item(), ms_ssim_avg.item(), psnr_avg.item(), egras_avg.item())

            save_name = save_dir+ir_name[0]
            mkdir(save_dir)

            save_image(toutputs, save_name)

    # print(benchmark_list)
    fields = ['no', 'model', 'filename', 'ssim_avg', 'ms_ssim_avg', 'psnr_avg', 'egras_avg']
    with open('benchmark_result.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile) 
        
        # writing the fields 
        csvwriter.writerow(fields) 
        
        # writing the data rows 
        csvwriter.writerows(benchmark_list)

        csvfile.close()

