import torch
from model import Fusionmodel
from utils import CustomVisionDataset_test, mkdir
from loss_fn import loss_function
from torchvision.utils import save_image

def loader_function(root, sub1, sub2):
    Data_set = CustomVisionDataset_test(root, sub1, sub2)
    Loader = torch.utils.data.DataLoader(Data_set, shuffle=False)
    return Loader

def test_fusion(model_path, test_dataloader,save_dir, device):
    
    #load model
    saved_model = Fusionmodel().to(device)
    saved_model.load_state_dict(torch.load(model_path))

    saved_model.eval()

    with torch.no_grad():
        for i, tdata in enumerate(test_dataloader):
            
            vis_image, ir_image, vis_name, ir_name = tdata
            vis_image = vis_image.to(device)
            ir_image = ir_image.to(device)

            toutputs = saved_model(vis_image, ir_image)

            #report loss        
            tloss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg = loss_function(vis_image, ir_image, toutputs, device)
            
            print(tloss.item(), ssim_loss_avg.item(), ms_ssim_loss_avg.item(), psnr_loss_avg.item())

            save_name = save_dir+str(vis_name)+"-"+str(ir_name)+'.jpg'
            mkdir(save_dir)

            save_image(toutputs, save_name)

if __name__ == '__main__':

    device = 'cuda'
    model_path = "New_model/trained/model_20230830_113427_19"
    test_loader = loader_function( 'New_model/Original_test_image', 'VIS', 'IR')
    save_dir =  "New_model/Output_test/"

    test_fusion(model_path, test_loader,save_dir, device)

