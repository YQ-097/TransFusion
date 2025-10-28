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


def load_model(path):

	transfusion_model = vgg.vgg19(pretrained=False)
	transfusion_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in transfusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(transfusion_model._get_name(), para * type_size / 1000 / 1000))

	transfusion_model.eval()
	transfusion_model.cuda()

	return transfusion_model


def run_demo(model, visible_path, output_path_root, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode):
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)
	vis_img = vis_img.expand(-1, 3, -1, -1)
	print(vis_img)
	if args.cuda:
		vis_img = vis_img.cuda()
	vis_img = Variable(vis_img, requires_grad=False)

	layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature = model.encode(vis_img)
	output = model.decode(layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer5_feature)[0]  * 255.0

	print(output.shape)
	print(torch.min(output), torch.max(output))

	file_name = 'fusion_' + fusion_type + '_' + str(index) +  '_network_' + network_type + '_' + strategy_type + '_' + ssim_weight_str + '.png'
	output_path = output_path_root + file_name

	if args.cuda:
		img = output.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = output.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	utils.save_images(output_path, img)

	print(output_path)


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
	test_path = "images/test-RGB/"
	#test_path = "images/IV_images/"
	network_type = 'transfusion'
	fusion_type = 'auto'  # auto, fusion_layer, fusion_all
	strategy_type_list = ['addition', 'attention_weight']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

	output_path = './outputs/'
	strategy_type = strategy_type_list[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	in_c = 3
	if in_c == 1:
		out_c = in_c
		mode = 'L'
		model_path = args.model_path_gray
	else:
		out_c = in_c
		mode = 'RGB'
		model_path = r".\models\stage1.model"
	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[2])
		ssim_weight_str = args.ssim_path[2]
		model = load_model(model_path)
		model.eval()
		model.cuda()
		for i in range(40):
			index = i + 1
			visible_path = test_path + 'VIS1 (' + str(index) + ').png'
			run_demo(model, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
	print('Done......')

if __name__ == '__main__':
	main()
