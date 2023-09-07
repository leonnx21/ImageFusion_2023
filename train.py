import os
import torch
from utils import mkdir, CustomVisionDataset_train
from loss_fn import loss_function
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

################################################################
#IMPORTANT

#Note: [ export LD_LIBRARY_PATH=/home/locnx/anaconda3/lib/ ] for compatibility tensorboard writer

################################################################

# PyTorch TensorBoard support

# Optimizers specified in the torch.optim package
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(name, model, training_loader, epoch_index, tb_writer, optimizer, device, report_freq):
    running_loss = 0.
    last_loss = 0.

    running_loss_ssim = 0.
    last_loss_ssim = 0.

    running_loss_msssim = 0.
    last_loss_msssim = 0.

    running_loss_psnr = 0.
    last_loss_psnr = 0.

    running_loss_egras = 0.
    last_loss_egras = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        vis_image, ir_image, vis_name, ir_name = data
        vis_image = vis_image.to(device)
        ir_image = ir_image.to(device)
    
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(vis_image, ir_image)

        # Compute the loss and its gradients
        loss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg, egras_loss_avg = loss_function(vis_image, ir_image, outputs, device)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_loss_ssim += ssim_loss_avg.item()
        running_loss_msssim += ms_ssim_loss_avg.item()
        running_loss_psnr += psnr_loss_avg.item()
        running_loss_egras += egras_loss_avg.item()
        
        if i % report_freq == (report_freq-1):
            last_loss = running_loss / report_freq # loss per batch
            last_loss_ssim = running_loss_ssim / report_freq
            last_loss_msssim = running_loss_msssim / report_freq
            last_loss_psnr = running_loss_psnr / report_freq  
            last_loss_egras = running_loss_egras / report_freq

            print(name + ' Epoch {} Batch {} train_loss: {}'.format(epoch_index + 1, i + 1, last_loss))
            
            tb_x = epoch_index * len(training_loader) + i + 1 
            tb_writer.add_scalar('Loss/Train', last_loss, tb_x)
            tb_writer.add_scalars('Loss/Detail',{'Loss' : last_loss, 'Loss_ssim' : last_loss_ssim, 'Loss_msssim' : last_loss_msssim, 'Loss_psnr' : last_loss_psnr, 'Egras_loss': last_loss_egras}, tb_x)
            
            running_loss = 0.
            running_loss_ssim = 0.
            running_loss_msssim = 0.
            running_loss_psnr = 0.
            running_loss_egras = 0.

        # print("success training")

    return last_loss

def validation_function(name, model, validation_loader, device, epoch_number, report_freq):
    running_vloss = 0.
    
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vis_image, ir_image, vis_name, ir_name = vdata
            vis_image = vis_image.to(device)
            ir_image = ir_image.to(device)

            voutputs = model(vis_image, ir_image)
            vloss, ssim_loss_avg, ms_ssim_loss_avg, psnr_loss_avg, egras_loss_avg = loss_function(vis_image, ir_image, voutputs, device)            
            
            running_vloss += vloss.item()

            if i % report_freq == (report_freq-1):
                print(name+" Epoch {} validation Batch {}".format(epoch_number+1,i+1))

    avg_vloss = running_vloss / (i + 1)

    return avg_vloss

def train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, epoch, device, report_freq):
    mkdir(train_path)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(writer_path+'fusion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = epoch

    best_loss = 1000000.
    best_vloss = 1000000.

    for epoch in range(EPOCHS):
        print(name + ' EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        
        model.train(True)
        avg_loss = train_one_epoch(name, model, training_loader, epoch_number, writer, optimizer, device, report_freq)
        
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
    
        model.eval()
        avg_vloss = validation_function(name, model, validation_loader, device, epoch_number,  report_freq)

        print(name + ' LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss            
            model_vpath = os.path.join(train_path,'model_best_val_{}_{}'.format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_vpath)
            print("model saved: "+model_vpath)

        if avg_loss < best_loss:
            best_loss = avg_loss            
            model_tpath = os.path.join(train_path,'model_best_train_{}_{}'.format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_tpath)
            print("model saved: "+model_tpath)

        epoch_number += 1

# if __name__ == '__main__':

#     device = 'cuda'
#     model = Fusionmodel().to(device)
#     lr = 1e-3
#     epoch = 20
#     batchsize = 4
    
#     #report batch
#     report_freq = 10   

#     optimizer = torch.optim.Adam(model.parameters(), lr = lr)

#     # training_loader = loader_function( '/storage/locnx/COCO2014/', '1000files', '1000files', batchsize=batchsize, num_samples=0)
#     # validation_loader = loader_function('/storage/locnx/COCO2014/', '100vals', '100vals', batchsize=batchsize, num_samples=0)
#     # train_path = './COCO2014/trained_model/'
#     # writer_path = './COCO2014/stats/'

#     training_loader = loader_function('/storage/locnx/COCO2014/', 'train2014', 'train2014', batchsize=batchsize, num_samples=0)
#     validation_loader = loader_function('/storage/locnx/COCO2014/', 'val2014', 'val2014', batchsize=batchsize, num_samples=2000)
#     train_path = './COCO2014/trained_model/'
#     writer_path = './COCO2014/stats/'

#     train(model, train_path, writer_path, training_loader, validation_loader, optimizer, epoch, device, report_freq)

    