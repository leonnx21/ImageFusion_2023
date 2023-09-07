import torch
import os
from model_visdrone import Fusionmodel
from utils import CustomVisionDataset_test, mkdir
from loss_fn import loss_function
from torchvision.utils import save_image
from test import loader_function, test_fusion

if __name__ == '__main__':

    device = 'cuda'

    model = Fusionmodel().to(device)

    model_path = "VISDRONE/trained_model/model_best_val_20230905_223657_8"

    test_loader1 = loader_function( 'Original_test_image', 'VIS', 'IR')
    save_dir1 =  "VISDRONE/Output_test/"
    test_fusion(model, model_path, test_loader1, save_dir1, device)

    test_loader2 = loader_function( '/storage/locnx/SampleVisDrone/SampleVisDroneTest', 'VIS', 'IR')
    save_dir2 =  "VISDRONE/Output_test1/"
    test_fusion(model, model_path, test_loader2, save_dir2, device)

