'''
	Pytorch implementation of MobileNet_v2_deeplab semantic segmantation  

	Train code 

Author: Zhengwei Li
Data: July 1 2018
'''

import argparse
import timeit
from datetime import datetime														
import os
import glob
from collections import OrderedDict

# PyTorch includes
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# data
from data import dataset

from data.loaders import create_loaders
# model
# from model import deeplab_v3_plus, deeplab_xception, enet
from model import deeplab_v3_plus
from PIL import  Image
from thop import profile
from thop import clever_format
import cv2
# dataloder
from data import dataset
# train helper
from utils import *
import pdb
from data.default_args import *
from f1_score import *
from helpers import *
from data.datasets import CentralCrop
import shutil
# paramers
parser = argparse.ArgumentParser()

parser.add_argument('--dataDir', default='./data/', help='dataset directory')
parser.add_argument('--saveDir', default='./result', help='save result')
parser.add_argument('--trainData', default='SBD', help='train dataset name')
parser.add_argument('--load', default= 'deeplab_v3_plus', help='save model')

parser.add_argument('--finetuning', action='store_true', default=False, help='finetuning the training')
parser.add_argument('--load_pre_train', action='store_true', default=False, help='load pre_train model')
parser.add_argument('--without_gpu', action='store_true', default=False, help='finetuning the training')

parser.add_argument('--nThreads', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--train_batch', type=int, default=4, help='input batch size for train')
parser.add_argument('--test_batch', type=int, default=8, help='input batch size for test')
parser.add_argument('--gpus', type=list, default=[0], help='GPUs ID')


parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--lrDecay', type=int, default=100)
parser.add_argument('--decayType', default='step')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--save_epoch', type=int, default=1, help='number of epochs to save model')

parser.add_argument("--dataset_type", type=str, default='helen',# 'celebA',#, #'celebA-face',#'EG1800',#'celebA-binary',
                    help="dataset type to be trained or valued.")
parser.add_argument("--meta-train-prct", type=int, default=META_TRAIN_PRCT,
                    help="Percentage of examples for meta-training set.")
parser.add_argument("--shorter-side", type=int, nargs='+', default=SHORTER_SIDE,
                    help="Shorter side transformation.")
parser.add_argument("--crop-size", type=int, nargs='+', default=CROP_SIZE,
                    help="Crop size for training,")
parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS,
                    help="Normalisation parameters [scale, mean, std],")
# parser.add_argument("--batch-size", type=int, nargs='+', default=BATCH_SIZE,
#                     help="Batch size to train the segmenter model.")
parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                    help="Number of workers for pytorch's dataloader.")
# parser.add_argument("--num-classes", type=int, nargs='+', default=NUM_CLASSES,
#                     help="Number of output classes for each task.")
parser.add_argument("--low-scale", type=float, default=LOW_SCALE,
                    help="Lower bound for random scale")
parser.add_argument("--high-scale", type=float, default=HIGH_SCALE,
                    help="Upper bound for random scale")
# parser.add_argument("--n-task0", type=int, default=N_TASK0,
#                     help="Number of images per task0 (trainval)")
parser.add_argument("--val-shorter-side", type=int, default=VAL_SHORTER_SIDE,
                    help="Shorter side transformation during validation.")
parser.add_argument("--val-crop-size", type=int, default=VAL_CROP_SIZE,
                    help="Crop size for validation.")
args = parser.parse_args()

color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

# Multi-GPUs
if args.without_gpu:
	print("use CPU !")
	device = torch.device('cpu')
else:
	if torch.cuda.is_available():
		n_gpu = torch.cuda.device_count()
		print("----------------------------------------------------------")
		print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
		print("----------------------------------------------------------")

		device = torch.device('cuda')
#-----------------------------------------------------
# Network 
#---------------

net = deeplab_v3_plus.DeepLabv_v3_plus_mv2_os_32(nInputChannels=3, n_classes=11)
# net = deeplab_v3_plus.DeepLabv_v3_plus_mv2_os_8(nInputChannels=3, n_classes=21)
# net = deeplab_v3_plus.DeepLabv_v3_plus_mv2_os_8(nInputChannels=3, n_classes=11)

# net = enet.ENet(5)
# net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=15) 

# if n_gpu > 1:
# 	net = nn.DataParallel(net)

if args.load_pre_train:
	net = load_pretrain_pam(net)

net.to(device)

#-----------------------------------------------------
# Loss 
#---------------
criterion = nn.CrossEntropyLoss(weight=None, size_average=False, ignore_index=-1).to(device)
# criterion = nn.BCELoss()

#-----------------------------------------------------
# Data
#---------------

# train_data = dataset.SBD(base_dir=os.path.join(args.dataDir, 'benchmark_RELEASE'), split=['train', 'val'])
# train_data = dataset.VOC(base_dir=os.path.join(args.dataDir, 'benchmark_RELEASE'), split=['train', 'val'])
# test_data  = dataset.VOC(base_dir=os.path.join(args.dataDir, 'VOCdevkit/VOC2012'), split='val')
# train_data  = dataset.VOC(base_dir=os.path.join(args.dataDir, 'VOCdevkit/VOC2012'), split='train')
# train_data = getattr(dataset, args.trainData)(base_dir = args.dataDir)

# data load
# trainloader = DataLoader(train_data, batch_size=args.train_batch,
# 				drop_last=True, shuffle=True, num_workers=args.nThreads, pin_memory=False)
train_loader, val_loader, do_search = create_loaders(args)
#-----------------------------------------------------
# Train loop
#---------------

save = saveData(args)
# finetuning
# if args.finetuning:
net = save.load_model(net,"result/deeplab_v3_plus/model/model_lastest.pt")
# segm_crit = nn.NLLLoss2d(ignore_index=255).cuda()
# lr_ = set_lr(args, epoch)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.99, weight_decay=5e-4)
print("Start test ! ... ...")

net.eval()
TEST_NUM = 6  # 4
fig, axes = plt.subplots(3, TEST_NUM, figsize=(12, 12))
# axes.set_xticks
ax = axes.ravel()
color_array = np.array(color_list)
data_file = dataset_dirs[args.dataset_type]['VAL_LIST']
data_dir = dataset_dirs[args.dataset_type]['VAL_DIR']

with open(data_file, 'rb') as f:
	datalist = f.readlines()
try:
	datalist = [
		(k, v) for k, v, _ in \
		map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist)]
except ValueError:  # Adhoc for test.
	datalist = [
		(k, k) for k in map(lambda x: x.decode('utf-8').strip('\n'), datalist)]

# random_array = random.sample(range(0, len(datalist)), TEST_NUM)
# imgs = [os.path.join(data_dir,datalist[i][0]) for i in random_array]
# msks = [os.path.join(data_dir,datalist[i][1]) for i in random_array]

imgs_all = [os.path.join(data_dir, datalist[i][0]) for i in range(0, len(datalist))]
msks_all = [os.path.join(data_dir, datalist[i][1]) for i in range(0, len(datalist))]

imgs = [
	# '../data/datasets/EG1800/Images/02323.png', # EG1800
	# '../data/datasets/EG1800/Images/01232.png',
	# '../data/datasets/EG1800/Images/02178.png',
	# '../data/datasets/EG1800/Images/02033.png',
	# '../data/datasets/EG1800/Images/02235.png',
	# '../data/datasets/EG1800/Images/00105.png',
	# '../data/datasets/EG1800/00009_224.png',
	'./data/helen/test/141794264_1_image.jpg',  # HELEN
	'./data/helen/test/107635070_1_image.jpg',
	'./data/helen/test/1030333538_1_image.jpg',
	'./data/helen/test/122276700_1_image.jpg',
	# './data/helen/test/1344304961_1_image.jpg',
	# './data/helen/test/1240746154_1_image.jpg',
]
msks = [
	# '../data/datasets/EG1800/Labels/02323.png',  # EG1800
	# '../data/datasets/EG1800/Labels/01232.png',
	# '../data/datasets/EG1800/Labels/02178.png',
	# '../data/datasets/EG1800/Labels/02033.png',
	# '../data/datasets/EG1800/Labels/02235.png',
	# '../data/datasets/EG1800/Labels/00105.png',
	# '../data/datasets/EG1800/00009_224_mask.png',
	'./data/helen/test/141794264_1_label.png',  # HELEN
	'./data/helen/test/107635070_1_label.png',
	'./data/helen/test/1030333538_1_label.png',
	'./data/helen/test/122276700_1_label.png',
	# './data/helen/test/1344304961_1_label.png',
	# './data/helen/test/1240746154_1_label.png'
]

class NewCrop(object):
    """Crop centrally the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_mul_val):
        assert isinstance(crop_mul_val, int)
        self.crop_mul_val = crop_mul_val
        self.crop_size=[0,0]
        # if self.crop_size % 2 != 0:
        #     self.crop_size -= 1

    def __call__(self, image,mask):
        # image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        # self.crop_size[0] = h-(h % 4)
        # self.crop_size[1] = w-(w % 4)

        self.crop_size[0] = 224 #352
        self.crop_size[1] = 224 #352
        h_margin = (h - self.crop_size[0]) // 2
        w_margin = (w - self.crop_size[1]) // 2

        image = image[h_margin : h_margin + self.crop_size[0],
                      w_margin : w_margin + self.crop_size[1]]
        mask = mask[h_margin : h_margin + self.crop_size[0],
                    w_margin : w_margin + self.crop_size[1]]
        return  image,  mask


show_raw_portrait_seg = 1
for i, img_path in enumerate(imgs):
	# logger.info("Testing image:{}".format(img_path))
	img = np.array(Image.open(img_path))
	msk = np.array(Image.open(msks[i]))
	img,msk = NewCrop(8)(img,msk)
	orig_size = img.shape[:2][::-1]
	ax[i].imshow(img, aspect='auto')
	plt.axis('off')
	if args.dataset_type == 'EG1800' and show_raw_portrait_seg:
		img_msk = img.copy()
		img_msk[msk == 0] = (0, 0, 255)
		ax[TEST_NUM + i].imshow(img_msk, aspect='auto')
	elif args.dataset_type == 'helen' or args.dataset_type == 'helen_nohair':
		ax[TEST_NUM + i].imshow(img, aspect='auto')
		msk = color_array[msk]
		ax[TEST_NUM + i].imshow(msk, aspect='auto', alpha=0.7)
	else:
		# ax[3*i+1].imshow(img,aspect='auto')
		msk = color_array[msk]
		ax[TEST_NUM + i].imshow(msk, aspect='auto', )

	plt.axis('off')

	# if img.shape[0] % 4 != 0:
	# 	gap=img.shape[0] // 4
	# 	crop_size=img_path.size - gap
	img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float().to(device)
	# img_inp = torch.tensor(prepare_img(img)).float().to(device)
	segm = net(img_inp)[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))  # 47*63*21
	# cal params and flops
	# input = torch.randn(1,3,224,224)
	input = torch.randn(1, 3, 512, 512).cuda()
	# flops, params = profile(net, inputs = (input,), )
	flops, params = profile(net, inputs=(input,), )
	flops, params = clever_format([flops, params], "%.3f")
	print(flops)
	print(params)
	segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)  # 375*500*21  #EG1800 need
	segm = segm.argmax(axis=2).astype(np.uint8)
	if args.dataset_type == 'EG1800' and show_raw_portrait_seg:
		img_segm = img.copy()
		img_segm[segm == 0] = (0, 0, 255)
		ax[2 * TEST_NUM + i].imshow(img_segm, aspect='auto')
	elif args.dataset_type == 'helen' or args.dataset_type == 'helen_nohair':
		segm = color_array[segm]  # 375*500*3  #wath this usage ,very very important
		ax[2 * TEST_NUM + i].imshow(img, aspect='auto')
		ax[2 * TEST_NUM + i].imshow(segm, aspect='auto', alpha=0.7)
	# print(segm.shape)
	else:
		segm = color_array[segm]  # 375*500*3  #wath this usage ,very very important
		# ax[3 * i + 2].imshow(img, aspect='auto')
		ax[2 * TEST_NUM + i].imshow(segm, aspect='auto', )
		ax[2 * TEST_NUM + i].set_xticks([])
		ax[2 * TEST_NUM + i].set_yticks([])
	plt.axis('off')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()
# fig.savefig('./eg1800.jpg')

# if args.dataset_type == 'helen' or args.dataset_type == 'helen_nohair' or args.dataset_type == 'celebA':
# 	validate_output_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_output')
# 	validate_gt_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_gt')
# 	validate_color_dir = os.path.join(dataset_dirs[args.dataset_type]['VAL_DIR'], 'validate_output_color')
#
# 	if not os.path.exists(validate_output_dir):
# 		os.makedirs(validate_output_dir)
# 	else:
# 		shutil.rmtree(validate_output_dir)
# 		os.makedirs(validate_output_dir)
#
# 	if not os.path.exists(validate_gt_dir):
# 		os.makedirs(validate_gt_dir)
# 	else:
# 		shutil.rmtree(validate_gt_dir)
# 		os.makedirs(validate_gt_dir)
#
# 	if not os.path.exists(validate_color_dir):
# 		os.makedirs(validate_color_dir)
# 	else:
# 		shutil.rmtree(validate_color_dir)
# 		os.makedirs(validate_color_dir)
#
# 	# save_color = 0
# 	for i, img_path in enumerate(imgs_all):
# 		# logger.info("Testing image:{}".format(img_path))
# 		img = np.array(Image.open(img_path))
# 		msk = np.array(Image.open(msks_all[i]))
# 		orig_size = img.shape[:2][::-1]
#
# 		img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float().to(device)
# 		segm = segmenter(img_inp)[0].squeeze().data.cpu().numpy().transpose((1, 2, 0))  # 47*63*21
# 		if args.dataset_type == 'celebA':
# 			# msk = cv2.resize(msk,segm.shape[0:2],interpolation=cv2.INTER_NEAREST)
# 			segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)  # 375*500*21
# 		else:
# 			segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)  # 375*500*21
# 		segm = segm.argmax(axis=2).astype(np.uint8)
#
# 		image_name = img_path.split('/')[-1].split('.')[0]
# 		# image_name = val_loader.dataset.datalist[i][0].split('/')[1].split('.')[0]
# 		cv2.imwrite(os.path.join(validate_color_dir, "{}.png".format(image_name)), color_array[segm])
# 		# cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(image_name)), color_array[msk])
# 		cv2.imwrite(os.path.join(validate_output_dir, "{}.png".format(image_name)), segm)
# 		cv2.imwrite(os.path.join(validate_gt_dir, "{}.png".format(image_name)), msk)
#
# 	if args.dataset_type == 'celebA':
# 		cal_f1_score_celebA(validate_gt_dir, validate_output_dir)  # temp comment
# 	# pass
# 	else:
# 		cal_f1_score(validate_gt_dir, validate_output_dir)
#
# 	# plt.show(
#
#
#
#
