# Training DenseFuse network
# auto-encoder

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
# from ssim import msssim
from utils import mkdir, gradient, hist_similar, CustomVisionDataset
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim as msssim

from model import Fusionmodel
from args_fusion import args

def main():
	# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	train()


def train():
	
	root = '/storage/locnx/'
	sub1 = 'train2014'
	sub2 = 'train2014'
	dataset = CustomVisionDataset(root, sub1, sub2)
	train_dataloader = DataLoader(dataset, shuffle=False)
	
	# root = '/storage/locnx/VisDrone/train'
	# sub1 = 'trainimg'
	# sub2 = 'trainimgr'
	# dataset = CustomVisionDataset(root, sub1, sub2)
	# train_dataloader = DataLoader(dataset, shuffle=False)
	
	device = 'cuda'
	fusion_model = Fusionmodel().to(device)
	batches = 16

	print(fusion_model)
	optimizer = Adam(fusion_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = msssim

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[2])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[2])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	Loss_pixel = []
	Loss_ssim = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		imgs_T = iter(train_dataloader)

		count = 0
		for batch in range(batches):
			# get fusion image

			img_vis_ir = next(imgs_T)
			img_vis = img_vis_ir[0].to(device)
			img_ir = img_vis_ir[1].to(device)
			
			optimizer.zero_grad()
			
			outputs = fusion_model(img_vis, img_ir)

			# resolution loss
			x = Variable(img_vis.data.clone(), requires_grad=False)
			y = Variable(img_ir.data.clone(), requires_grad=False)
			

			ssim_loss_value1 = 0.
			pixel_loss_value1 = 0.
			ssim_loss_value2 = 0.
			pixel_loss_value2 = 0.
			for output in outputs:
				output = output.unsqueeze(0).float()
				pixel_loss_temp1 = mse_loss(output, x)
				ssim_loss_temp1 = ssim_loss(output, x)
				ssim_loss_value1 += (1-ssim_loss_temp1)
				pixel_loss_value1 += pixel_loss_temp1

				pixel_loss_temp2 = mse_loss(output, y)
				ssim_loss_temp2 = ssim_loss(output, y)
				ssim_loss_value2 += (1-ssim_loss_temp2)
				pixel_loss_value2 += pixel_loss_temp2

			ssim_loss_value =  (ssim_loss_value1 + ssim_loss_value2)/2
			pixel_loss_value = (pixel_loss_value1 + pixel_loss_value2)/2

			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)

			# total loss
			total_loss = pixel_loss_value + args.ssim_weight[2] * ssim_loss_value
			total_loss.backward()
			optimizer.step()

			all_ssim_loss += ssim_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
								  (args.ssim_weight[2] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[2] * all_ssim_loss + all_pixel_loss) / args.log_interval)

				all_ssim_loss = 0.
				all_pixel_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				fusion_model.eval()
				fusion_model.cpu()
				save_model_filename = args.ssim_path[2] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  2] + ".pkl"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(fusion_model.state_dict(), save_model_path)
				
				# save loss data
				# pixel loss
				loss_data_pixel = np.array(Loss_pixel)
				loss_filename_path = args.ssim_path[2] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[2] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
				
				# SSIM loss
				loss_data_ssim = np.array(Loss_ssim)
				loss_filename_path = args.ssim_path[2] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[2] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
				
				# all loss
				loss_data_total = np.array(Loss_all)
				loss_filename_path = args.ssim_path[2] + '/' + "loss_total_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[2] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_total': loss_data_total})

				fusion_model.train()
				fusion_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[2] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[2] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})

	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[2] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[2] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})

	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[2] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[2] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	
	# save model
	fusion_model.eval()
	fusion_model.cpu()
	save_model_filename = args.ssim_path[2] + '/' "Final_epoch_" + str(args.epochs) + "_" + "_" + args.ssim_path[2] + ".pkl"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(fusion_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
