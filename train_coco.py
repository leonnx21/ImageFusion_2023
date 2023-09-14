import torch
from model_visdrone import Fusionmodel
from utils import CustomVisionDataset_train
from train import train

def loader_function(root, sub1, sub2, batchsize=8, num_samples=0):
    Data_set = CustomVisionDataset_train(root, sub1, sub2)
    
    if num_samples != 0:
        random_sampler = torch.utils.data.RandomSampler(Data_set, num_samples=num_samples)
        Loader = torch.utils.data.DataLoader(Data_set, batch_size=batchsize, sampler=random_sampler)
    else:
        Loader = torch.utils.data.DataLoader(Data_set, batch_size=batchsize, shuffle=True)
    return Loader

if __name__ == '__main__':

    device = 'cuda'
    model = Fusionmodel().to(device)
    lr = 1e-3
    epoch = 20
    batchsize = 4
    
    #report batch
    report_freq = 10   

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    training_loader = loader_function( '/storage/locnx/COCO2014/', '1000files', '1000files', batchsize=batchsize, num_samples=0)
    validation_loader = loader_function('/storage/locnx/COCO2014/', '100vals', '100vals', batchsize=batchsize, num_samples=0)
    train_path = './COCO2014_3/trained_model/'
    writer_path = './COCO2014_3/stats/'
    name = 'COCO2014_3'

    # training_loader = loader_function('/storage/locnx/COCO2014/', 'train2014', 'train2014', batchsize=batchsize, num_samples=0)
    # validation_loader = loader_function('/storage/locnx/COCO2014/', 'val2014', 'val2014', batchsize=batchsize, num_samples=2000)
    # train_path = './COCO2014_3/trained_model/'
    # writer_path = './COCO2014_3/stats/'
    # name = 'COCO2014_3'

    train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, epoch, device, report_freq)

    