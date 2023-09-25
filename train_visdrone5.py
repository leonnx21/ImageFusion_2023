import torch
from model_visdrone5 import Fusionmodel
from train import train
from utils import CustomVisionDataset_train
from loss_fn2 import CustomLoss

################################################################
#IMPORTANT

#Note: [ export LD_LIBRARY_PATH=/home/locnx/anaconda3/lib/ ] for compatibility tensorboard writer

################################################################


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
    epoch = 200
    batchsize = 4
    
    #report batch
    report_freq = 100   

    train_sample = 0 #0 for all samples
    validation_sample = 2000

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_function = CustomLoss().to(device)

    # training_loader = loader_function( '/storage/locnx/COCO2014/', '1000files', '1000files', batchsize=batchsize, num_samples=0)
    # validation_loader = loader_function('/storage/locnx/COCO2014/', '100vals', '100vals', batchsize=batchsize, num_samples=0)
    # train_path = './VISDRONE_3/trained_model/'
    # writer_path = './VISDRONE_3/stats/'
    # name = 'VISDRONE_3'

    training_loader = loader_function( '/storage/locnx/CBD/train/', 'VIS', 'IR', batchsize=batchsize, num_samples=train_sample)
    validation_loader = loader_function('/storage/locnx/CBD/val/', 'VIS', 'IR', batchsize=batchsize, num_samples=validation_sample)
    train_path = './Trained/CBD5/trained_model/'
    writer_path = './Trained/CBD5/stats/'
    name = 'CBD5'

    train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, loss_function, epoch, device, report_freq)

    