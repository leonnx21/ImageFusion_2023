import torch
from model_visdrone6 import Fusionmodel
from train import train
from utils import CustomVisionDataset_train, loader_function
from loss_fn2 import CustomLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from RandomSplit_data import random_split


################################################################
#IMPORTANT

#Note: [ export LD_LIBRARY_PATH=/home/locnx/anaconda3/lib/ ] for compatibility tensorboard writer

################################################################


if __name__ == '__main__':

    device = 'cuda'
    model = Fusionmodel().to(device)
    lr = 0.0001
    epoch = 200
    batchsize = 8
    
    #report batch
    report_freq = 1000  

    training_loader, validation_loader = loader_function( '/storage/locnx/CBD/train/', 'VIS', 'IR', batchsize=batchsize)

    train_path = './Trained/CBD6/trained_model/'
    writer_path = './Trained/CBD6/stats/'
    name = 'CBD6'

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    loss_function = CustomLoss().to(device)

    #traing new or resume    
    training_new = True

    if training_new == True:
        train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, loss_function, epoch, device, report_freq)
    else:
        #provide path to resume 
        checkpoint = torch.load("Trained/CBD4/trained_model/all_train/model_train_20230927_132202_6")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_number = checkpoint['epoch']
        epoch = epoch - epoch_number
        best_tloss = checkpoint['best_tloss']
        best_vloss = checkpoint['best_vloss']

        train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, loss_function, epoch, device, report_freq, epoch_number, best_tloss, best_vloss)




    