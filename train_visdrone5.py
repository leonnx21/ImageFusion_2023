import torch
from model_visdrone5 import Fusionmodel
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
    lr = 1e-4
    epoch = 200
    batchsize = 8
    
    #report batch
    report_freq = 100 

    training_loader, validation_loader = loader_function( '/storage/locnx/CBD/train/', 'VIS', 'IR', batchsize=batchsize)

    train_path = './Trained/CBD5/trained_model/'
    writer_path = './Trained/CBD5/stats/'
    name = 'CBD5'

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    loss_function = CustomLoss().to(device)

    #traing new or resume    
    training_new = False

    if training_new == True:
        train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, loss_function, epoch, device, report_freq)
    else:
        #provide path to resume 
        checkpoint = torch.load("Trained/CBD5/trained_model/all_train/model_train_20230927_151947_86")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_number = checkpoint['epoch']
        epoch = epoch - epoch_number
        best_tloss = checkpoint['best_tloss']
        best_vloss = checkpoint['best_vloss']
        #can mannualy added
        timestamp = "20230927_151947" 
        
        train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, loss_function, epoch, device, report_freq, epoch_number, best_tloss, best_vloss, timestamp)



    