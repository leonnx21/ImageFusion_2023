from model_visdrone2 import Fusionmodel
from test import loader_function, test_fusion
import torch

if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    model = Fusionmodel().to(device)

    model_path = "CBD/trained_model/model_train_20230914_140847_0"

    test_loader1 = loader_function( 'Original_test_image', 'VIS', 'IR')
    save_dir1 =  "CBD/Output_test/"
    test_fusion(model, model_path, test_loader1, save_dir1, device)

    test_loader2 = loader_function( '/storage/locnx/SampleVisDrone/SampleVisDroneTest', 'VIS', 'IR')
    save_dir2 =  "CBD/Output_test1/"
    test_fusion(model, model_path, test_loader2, save_dir2, device)

