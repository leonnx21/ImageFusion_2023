from model_visdrone import Fusionmodel
from test import loader_function, test_fusion

if __name__ == '__main__':

    device = 'cpu'

    model = Fusionmodel().to(device)

    model_path = "COCO2014/trained_model/model_best_val_20230910_111121_0"

    test_loader1 = loader_function( 'Original_test_image', 'VIS', 'IR')
    save_dir1 =  "COCO2014_3/Output_test/"
    test_fusion(model, model_path, test_loader1, save_dir1, device)

    test_loader2 = loader_function( '/storage/locnx/SampleVisDrone/SampleVisDroneTest', 'VIS', 'IR')
    save_dir2 =  "COCO2014_3/Output_test1/"
    test_fusion(model, model_path, test_loader2, save_dir2, device)

