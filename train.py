import os
import torch
from utils import mkdir
from loss_fn import loss_function
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

################################################################
#IMPORTANT

#Note: [ export LD_LIBRARY_PATH=/home/locnx/anaconda3/lib/ ] for compatibility tensorboard writer

################################################################

# Optimizers specified in the torch.optim package
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(name, model, training_loader, epoch_index, tb_writer, optimizer, device, report_freq):
    running_loss_details = [0., 0., 0., 0., 0.]
    last_loss_details = [0., 0. , 0., 0., 0.]

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
        loss0, loss1, loss2, loss3, loss4 = loss_function(vis_image, ir_image, outputs, device)

        loss0.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss_details[0] += loss0
        running_loss_details[1] += loss1
        running_loss_details[2] += loss2
        running_loss_details[3] += loss3
        running_loss_details[4] += loss4

        
        if i % report_freq == (report_freq-1):
            last_loss_details[0] = running_loss_details[0] / report_freq # loss per batch
            last_loss_details[1] = running_loss_details[1] / report_freq 
            last_loss_details[2] = running_loss_details[2] / report_freq 
            last_loss_details[3] = running_loss_details[3] / report_freq 
            last_loss_details[4] = running_loss_details[4] / report_freq 

            print(name + ' Epoch {} Batch {} train_loss: {}'.format(epoch_index + 1, i + 1, last_loss_details[0]))
            
            tb_x = epoch_index * len(training_loader) + i + 1 
            tb_writer.add_scalar('Loss/Train', last_loss_details[0], tb_x)
            tb_writer.add_scalars('Loss/Detail',{'Loss0' : last_loss_details[0], 'Loss1' : last_loss_details[1], 'Loss2' : last_loss_details[2], 'Loss3' : last_loss_details[3], 'Loss4': last_loss_details[4]}, tb_x)
            
            running_loss_details[0] = 0.
            running_loss_details[1] = 0.
            running_loss_details[2] = 0.
            running_loss_details[3] = 0.
            running_loss_details[4] = 0.

        # print("success training")

    return last_loss_details[0]

def validation_function(name, model, validation_loader, device, epoch_number, report_freq):
    running_vloss = 0.
    
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vis_image, ir_image, vis_name, ir_name = vdata
            vis_image = vis_image.to(device)
            ir_image = ir_image.to(device)

            voutputs = model(vis_image, ir_image)
            vloss, loss1, loss2, loss3, loss4 = loss_function(vis_image, ir_image, voutputs, device)            
            
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

        model_tpath2 = os.path.join(train_path,'model_train_{}_{}'.format(timestamp, epoch_number))
        torch.save(model.state_dict(), model_tpath2)

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

    