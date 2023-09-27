import os
import torch
from utils import mkdir
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from loss_fn2 import CustomLoss

################################################################
#IMPORTANT

#Note: [ export LD_LIBRARY_PATH=/home/locnx/anaconda3/lib/ ] for compatibility tensorboard writer

################################################################

# Optimizers specified in the torch.optim package
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(name, model, training_loader, epoch_index, tb_writer, optimizer, loss_function, device, report_freq):
    running_loss_details = [0., 0., 0., 0.]
    last_loss_details = [0., 0., 0., 0.]
    
    training_loss = 0.
    training_loss_avg = 0.

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
        loss0, loss1, loss2, loss3 = loss_function(vis_image, ir_image, outputs, device)

        loss0.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        training_loss += loss0.item()
        running_loss_details[0] += loss0.item()
        running_loss_details[1] += loss1.item()
        running_loss_details[2] += loss2.item()
        running_loss_details[3] += loss3.item()

        
        if i % report_freq == (report_freq-1):
            last_loss_details[0] = running_loss_details[0] / report_freq # loss per batch
            last_loss_details[1] = running_loss_details[1] / report_freq 
            last_loss_details[2] = running_loss_details[2] / report_freq 
            last_loss_details[3] = running_loss_details[3] / report_freq 

            print(name + ' Epoch {} Batch {} train_loss: {}'.format(epoch_index + 1, i + 1, last_loss_details[0]))
            
            tb_x = epoch_index * len(training_loader) + i + 1 
            tb_writer.add_scalar('Loss/Train', last_loss_details[0], tb_x)
            tb_writer.add_scalars('Loss/Detail_Train',{'Loss0' : last_loss_details[0], 'Loss1' : last_loss_details[1], 'Loss2' : last_loss_details[2], 'Loss3' : last_loss_details[3]}, tb_x)
            tb_writer.flush()

            running_loss_details[0] = 0.
            running_loss_details[1] = 0.
            running_loss_details[2] = 0.
            running_loss_details[3] = 0.

        #report last average traing loss validation

        # scheduler.step()
    
    training_loss_avg = training_loss/(i+1)
    training_loss = 0.

    # print("success training")

    return training_loss_avg, last_loss_details[0]

def validation_function(name, model, validation_loader, loss_function, device, epoch_number, tb_writer, report_freq):
    running_vloss_details = [0., 0., 0., 0.]
    last_vloss_details = [0., 0., 0., 0.]

    validation_loss = 0.
        
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vis_image, ir_image, vis_name, ir_name = vdata
            vis_image = vis_image.to(device)
            ir_image = ir_image.to(device)

            voutputs = model(vis_image, ir_image)
            vloss0, vloss1, vloss2, vloss3 = loss_function(vis_image, ir_image, voutputs, device)            
            
            # Gather data and report
            validation_loss += vloss0.item()
            running_vloss_details[0] += vloss0.item()
            running_vloss_details[1] += vloss1.item()
            running_vloss_details[2] += vloss2.item()
            running_vloss_details[3] += vloss3.item()
            
            if i % report_freq == (report_freq-1):
                last_vloss_details[0] = running_vloss_details[0] / report_freq # loss per batch
                last_vloss_details[1] = running_vloss_details[1] / report_freq 
                last_vloss_details[2] = running_vloss_details[2] / report_freq 
                last_vloss_details[3] = running_vloss_details[3] / report_freq 

                print(name+" Epoch {} validation Batch {}".format(epoch_number+1,i+1))

                tb_x = epoch_number * len(validation_loader) + i + 1 
                tb_writer.add_scalar('Loss/Validation', last_vloss_details[0], tb_x)
                tb_writer.add_scalars('Loss/Detail_Validation',{'Loss0' : last_vloss_details[0], 'Loss1' : last_vloss_details[1], 'Loss2' : last_vloss_details[2], 'Loss3' : last_vloss_details[3]}, tb_x)
                tb_writer.flush()

                running_vloss_details[0] = 0.
                running_vloss_details[1] = 0.
                running_vloss_details[2] = 0.
                running_vloss_details[3] = 0.

        avg_vloss = validation_loss / (i + 1)
        validation_loss = 0.

    return avg_vloss

def train(name, model, train_path, writer_path, training_loader, validation_loader, optimizer, loss_function, epoch, device, report_freq, epoch_number=0, best_loss = 1000000. , best_vloss = 1000000.):

    best_train_path = os.path.join(train_path,'best_train/')
    best_val_path = os.path.join(train_path,'best_val/')
    all_train_path = os.path.join(train_path,'all_train/')
    
    mkdir(best_train_path)
    mkdir(best_val_path)
    mkdir(all_train_path)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(writer_path+'fusion_trainer_{}_{}'.format(name,timestamp))
    
    epoch_number = epoch_number

    EPOCHS = epoch

    best_loss = best_loss
    best_vloss = best_vloss

    for epoch in range(EPOCHS):
        print(name + ' EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        
        model.train(True)
        avg_loss, last_loss = train_one_epoch(name, model, training_loader, epoch_number, writer, optimizer, loss_function, device, report_freq)
        
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
    
        model.eval()
        avg_vloss = validation_function(name, model, validation_loader, loss_function, device, epoch_number, writer, report_freq)

        print(name + ' LOSS train {} validation {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Avg_Training' : avg_loss,'Last_Training': last_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        # writer.add_scalars('Training Validation Loss gap',
        #                 { 'Training minus Validation' : avg_loss - avg_vloss},
        #                 epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss            
            model_vpath = os.path.join(train_path,'best_val/','model_best_val_{}_{}'.format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_vpath)
            print("model saved: "+model_vpath)

        if avg_loss < best_loss:
            best_loss = avg_loss            
            model_tpath = os.path.join(train_path,'best_train/','model_best_train_{}_{}'.format(timestamp, epoch_number))
            torch.save(model.state_dict(), model_tpath)
            print("model saved: "+model_tpath)

        # save all epoch for able to be resume
        model_tpath2 = os.path.join(train_path,'all_train/','model_train_{}_{}'.format(timestamp, epoch_number))
        torch.save({
            'epoch': epoch_number,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_tloss': best_loss,
            'best_vloss': best_vloss,
            }, model_tpath2)

        epoch_number += 1

        
