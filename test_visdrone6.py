from model_visdrone6 import Fusionmodel
from test import loader_function, test_fusion
from benchmark import CustomBenchmark
import torch

if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'

    model = Fusionmodel().to(device)
    loss_function = CustomBenchmark().to(device)
    name = 'CBD6'

    model_path = "Trained/CBD6/trained_model/best_val/model_best_val_20230927_151951_127"

    test_loader1 = loader_function( 'Original_test_image', 'VIS', 'IR')
    save_dir1 =  "Tested/CBD6/Output_test/"
    test_fusion(name, model, model_path, loss_function, test_loader1, save_dir1, device)

    # test_loader2 = loader_function( '/storage/locnx/VisDrone/test/Test2/', 'VIS', 'IR')
    # save_dir2 =  "Tested/CBD6/Output_test1/"
    # test_fusion(name, model, model_path, loss_function, test_loader2, save_dir2, device)

    test_loader2 = loader_function( '/storage/locnx/VisDrone/val/', 'VIS2', 'IR2')
    save_dir2 =  "Tested/CBD6/Output_test1/"
    test_fusion(name, model, model_path, loss_function, test_loader2, save_dir2, device)
