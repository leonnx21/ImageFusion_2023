from model_visdrone9 import Fusionmodel
from test import loader_function, test_fusion
from benchmark import CustomBenchmark
import torch

if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'

    model = Fusionmodel().to(device)
    loss_function = CustomBenchmark().to(device)
    name = 'CBD9'

    model_path = "Trained/CBD9/trained_model/best_val/model_best_val_20231005_065448_145"

    test_loader1 = loader_function( 'Original_test_image', 'VIS', 'IR')
    save_dir1 =  "Tested/CBD9/Output_test/"
    test_fusion(name, model, model_path, loss_function, test_loader1, save_dir1, device)

    # test_loader2 = loader_function( '/storage/locnx/VisDrone/test/Test2/', 'VIS', 'IR')
    # save_dir2 =  "Tested/CBD9/Output_test1/"
    # test_fusion(name, model, model_path, loss_func
    # tion, test_loader2, save_dir2, device)

    test_loader3 = loader_function( '/storage/locnx/VisDrone/val/', 'VIS2', 'IR2')
    save_dir3 =  "Tested/CBD9/Output_test1/"
    test_fusion(name, model, model_path, loss_function, test_loader3, save_dir3, device)
