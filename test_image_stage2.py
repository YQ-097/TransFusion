# test phase
import torch
from torch.autograd import Variable
from net import DenseFuse_net
import utils
from args_fusion import args
import numpy as np
import time
import cv2
import os
import vgg
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def load_model(path):

	transfusion_model = vgg.vgg19(pretrained=False)
	transfusion_model.load_state_dict(torch.load(path))

	print(parameter_count_table(transfusion_model))
	para = sum([np.prod(list(p.size())) for p in transfusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(transfusion_model._get_name(), para * type_size / 1000 / 1000))

	transfusion_model.eval()
	transfusion_model.cuda()

	return transfusion_model


def run_demo(model, infrared_path, visible_path, output_path_root, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode):
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)
	ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)

	vis_img = vis_img.expand(-1, 3, -1, -1)
	ir_img = ir_img.expand(-1, 3, -1, -1)

	_, _, h_old, w_old = vis_img.size()
	xx = 16
	if h_old % xx != 0 or w_old % xx != 0:
		h_new = int(torch.ceil(torch.Tensor([h_old]) / (xx))) * (xx)
		w_new = int(torch.ceil(torch.Tensor([w_old]) / (xx))) * (xx)

		padding_h = h_new - h_old
		padding_w = w_new - w_old

		vis_img = torch.nn.functional.pad(vis_img, (0, padding_w, 0, padding_h))
		ir_img = torch.nn.functional.pad(ir_img, (0, padding_w, 0, padding_h))

	if args.cuda:
		vis_img = vis_img.cuda()
		ir_img = ir_img.cuda()
	vis_img = Variable(vis_img, requires_grad=False)
	ir_img = Variable(ir_img, requires_grad=False)

	output = model(ir_img, vis_img)[0]  * 255.0


	if h_old % xx != 0 or w_old % xx != 0:
		top = 0
		bottom = output.shape[2] - padding_h
		left = 0
		right = output.shape[3] - padding_w
		output = output[:, :, top:bottom, left:right]


	file_name = 'fusion_' + fusion_type + '_' + str(index) +  '_network_' + network_type + '_' + strategy_type + '_' + ssim_weight_str + '.png'
	output_path = output_path_root + file_name

	if args.cuda:
		img = output.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = output.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	utils.save_images(output_path, img)



def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)


def main():
	# run demo
	dataset = 'MSRS'
	if dataset is 'TNO':
		test_path = "images/TNO/"
	elif dataset is 'MSRS':
		test_path = "images/MSRS/"
	network_type = 'transfusion'
	fusion_type = 'auto'  # auto, fusion_layer, fusion_all
	strategy_type_list = ['addition', 'attention_weight']

	output_path = './output2/'
	strategy_type = strategy_type_list[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	# in_c = 3 for RGB images; in_c = 1 for gray images
	in_c = 3
	if in_c == 1:
		out_c = in_c
		mode = 'L'
		model_path = args.model_path_gray
	else:
		out_c = in_c
		mode = 'RGB'
		model_path = r".\models\stage2.model"
	start = time.time()
	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[2])
		ssim_weight_str = args.ssim_path[2]
		model = load_model(model_path)
		model.eval()
		model.cuda()
		for i in range(40):
			index = i + 1
			if dataset is 'MSRS':
				infrared_path = test_path + 'IR/'+ 'IR1 (' + str(index) + ').png'
				visible_path = test_path + 'VI_RGB/' + 'VIS1 (' + str(index) + ').png'
			elif dataset is 'TNO':
				infrared_path = test_path + 'IR/'+  str(index) + '.bmp'
				visible_path = test_path + 'VI_RGB/' +  str(index) + '.bmp'
			run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
	print('Done......')
	end = time.time()
	print(end - start, 's')

if __name__ == '__main__':
	main()
