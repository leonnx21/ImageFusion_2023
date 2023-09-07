import torch
from model_visdrone import Fusionmodel
from train import train
from utils import CustomVisionDataset_train_visdrone

################################################################
#IMPORTANT

#Note: [ export LD_LIBRARY_PATH=/home/locnx/anaconda3/lib/ ] for compatibility tensorboard writer

################################################################


def loader_function(root, sub1, sub2, batchsize=8, num_samples=0):
    Data_set = CustomVisionDataset_train_visdrone(root, sub1, sub2)
    
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
    epoch = 200
    batchsize = 4
    
    #report batch
    report_freq = 10   

    train_sample = 0 #0 for all samples
    validation_sample = 2000

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    training_loader = loader_function( '/storage/locnx/VisDrone/train/', 'trainimg', 'trainimgr', batchsize=batchsize, num_samples=train_sample)
    validation_loader = loader_function('/storage/locnx/VisDrone/val/', 'valimg', 'valimgr', batchsize=batchsize, num_samples=validation_sample)
    train_path = './VISDRONE/trained_model/'
    writer_path = './VISDRONE/stats/'
    name = 'VISDRONE'


    train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, epoch, device, report_freq)

    