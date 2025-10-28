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
import torch.nn.functional as F
import utils
#from net import DenseFuse_net
from vgg import VGG
from args_fusion import args
import pytorch_msssim
import vgg
import MILoss
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def main():
	original_imgs_path = utils.list_images(args.dataset_ir)
	train_num = 40000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	# for i in range(5):
	i = 1
	train(i, original_imgs_path)


def train(i, original_imgs_path):

	batch_size = args.batch_size

	in_c = 1 # 1 - gray; 3 - RGB
	if in_c == 1:
		img_model = 'L'
	else:
		img_model = 'RGB'
	input_nc = in_c
	output_nc = in_c

	transfusion_model = vgg.vgg19(pretrained=False)

	state_dict = torch.load(r".\models\stage1.model")#继续训练

	transfusion_model.load_state_dict(state_dict, strict=False)
	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		transfusion_model.load_state_dict(torch.load(args.resume))
	print(transfusion_model)



	#冻结参数
	params = []
	train_layer = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5',
				   'upsample', 'upsample1_1', 'upsample2_1', 'upsample2_2', 'upsample3_1', 'upsample3_2', 'upsample3_3', 'upsample4_1', 'upsample4_2', 'upsample4_3', 'upsample4_4',
				   'conv1_1_1', 'conv1_1_2', 'conv2_1_1', 'conv2_1_2', 'conv2_2_1', 'conv2_2_2', 'conv3_1_1', 'conv3_1_2', 'conv3_2_1', 'conv3_2_2', 'conv3_3_1', 'conv3_3_2', 'conv4_1_1', 'conv4_1_2', 'conv4_2_1', 'conv4_2_2', 'conv4_3_1', 'conv4_3_2', 'conv4_4_1', 'conv4_4_2', 'conv_out1', 'conv_out2', 'relu']
	for name, param in transfusion_model.named_parameters():
		if any(name.startswith(prefix) for prefix in train_layer):
			param.requires_grad = False
		else:
			print(name)
			params.append(param)
	optimizer = Adam(params, args.lr)

	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim
	mi_loss = MILoss.MI()

	if args.cuda:
		transfusion_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	Loss_pixel = []
	Loss_ssim = []
	Loss_mi = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	all_mi_loss = 0.

	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		transfusion_model.train()
		count = 0
		for batch in range(batches):

			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, mode=img_model)

			image_paths_vi = [x.replace('IR', 'VI_RGB') for x in image_paths_ir]
			img_vis = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, mode=img_model)

			count += 1
			optimizer.zero_grad()
			img_ir = Variable(img_ir, requires_grad=False)
			img_vis = Variable(img_vis, requires_grad=False)

			if args.cuda:
				img_ir = img_ir.cuda()
				img_vis = img_vis.cuda()

			input_img_ir = img_ir.expand(-1, 3, -1, -1)
			input_img_vis = img_vis.expand(-1, 3, -1, -1)

			# encoder
			ir_layer1_feature, ir_layer2_feature, ir_layer3_feature, ir_layer4_feature, ir_layer5_feature = transfusion_model.encode(input_img_ir)
			vis_layer1_feature, vis_layer2_feature, vis_layer3_feature, vis_layer4_feature, vis_layer5_feature = transfusion_model.encode(input_img_vis)
			#feature_selection
			layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature = transfusion_model.feature_selection(ir_layer1_feature, ir_layer2_feature, ir_layer3_feature, ir_layer4_feature, ir_layer5_feature, vis_layer1_feature, vis_layer2_feature, vis_layer3_feature, vis_layer4_feature, vis_layer5_feature)
			# decoder
			outputs = transfusion_model.decode(layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature)


			# resolution loss
			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			x_vi = Variable(img_vis.data.clone(), requires_grad=False)

			ssim_loss_value = 0.
			pixel_loss_value = 0.
			mi_loss_value = 0.
			for output in outputs:
				pixel_loss_temp = mse_loss(output, torch.max(x_ir, x_vi))
				ssim_loss_temp = (1 - ssim_loss(output, x_ir, normalize=True)) + (1 - ssim_loss(output, x_vi, normalize=True))


				# MI loss
				mi_loss_temp = mi_loss(layer1_feature, ir_layer1_feature, vis_layer1_feature)
				mi_loss_temp += mi_loss(layer5_feature, ir_layer5_feature, vis_layer5_feature)

				mi_loss_value += 100.0 / (-1.0 * mi_loss_temp)
				ssim_loss_value += ssim_loss_temp / 2
				pixel_loss_value += pixel_loss_temp * 100

			mi_loss_value /= len(outputs)
			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)

			# total loss
			total_loss = pixel_loss_value +  ssim_loss_value  + mi_loss_value  #0.01
			total_loss.backward()
			optimizer.step()

			all_mi_loss += mi_loss_value.item()
			all_ssim_loss += ssim_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t mi loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
								  all_mi_loss / args.log_interval,
								  ( all_ssim_loss  + all_pixel_loss + all_mi_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

				all_ssim_loss = 0.
				all_pixel_loss = 0.
				all_mi_loss = 0.

			if (batch + 1) % (100 * args.log_interval) == 0:
				# save model
				transfusion_model.eval()
				transfusion_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(transfusion_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_pixel = np.array(Loss_pixel)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = np.array(Loss_ssim)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data_total = np.array(Loss_all)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_total_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_total': loss_data_total})

				transfusion_model.train()
				transfusion_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	# save model
	transfusion_model.eval()
	transfusion_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(transfusion_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)

def gradient(input):
    filter1 = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    filter2 = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    filter1 = filter1.view(1, 1, 3, 3)
    filter2 = filter2.view(1, 1, 3, 3)
    filter1 = filter1.to(input.device)
    filter2 = filter2.to(input.device)
    Gradient1 = F.conv2d(input, filter1, stride=1, padding=1)
    Gradient2 = F.conv2d(input, filter2, stride=1, padding=1)
    Gradient = torch.abs(Gradient1) + torch.abs(Gradient2)
    return Gradient

if __name__ == "__main__":
	main()
