from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess
from models import *
import cv2
import glob

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015', help='KITTI version')
parser.add_argument(
    '--loadmodel', default='./trained/pretrained_model_KITTI2015.tar', help='loading model')
parser.add_argument('--leftimgdir', default=None, help='left Image directory')
parser.add_argument('--rightimgdir', default=None,
                    help='right Image directory')
parser.add_argument('--isgray', default=False, help='Is Gray Image')
parser.add_argument('--model', default='stackhourglass', help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--use-cuda', default=True,
                    help='enables CUDA training/Inference')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 1)')

parser.add_argument('--savedir', default='./', help='save directory')
args = parser.parse_args()

args.cuda = args.use_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

if args.leftimgdir is not None and os.path.exists(args.leftimgdir):
    leftimages_path = sorted(glob.glob(os.path.join(args.leftimgdir, "*.png")))
    leftimages = list(map(lambda x: x.split('/')[-1], leftimages_path))


if args.rightimgdir is not None and os.path.exists(args.rightimgdir):
    rightimages_path = sorted(glob.glob(os.path.join(args.rightimgdir, "*.png")))
    rightimages = list(map(lambda x: x.split('/')[-1], rightimages_path))

assert leftimages == rightimages, "Corresponding Stereo Images not found!!"


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda()

    imgL, imgR = Variable(imgL), Variable(imgR)

    with torch.no_grad():
        disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


def main():
    processed = preprocess.get_transform(augment=False)
    is_result_dir = False

    for leftimg, rightimg in zip(leftimages_path, rightimages_path):
        print("Left Image : {}, Right Image : {}".format(leftimg,rightimg))
        if args.isgray:
            imgL_o = cv2.cvtColor(cv2.imread(leftimg, 0), cv2.COLOR_GRAY2RGB)
            imgR_o = cv2.cvtColor(cv2.imread(rightimg, 0), cv2.COLOR_GRAY2RGB)
        else:
            imgL_o = (skimage.io.imread(leftimg).astype('float32'))
            imgR_o = (skimage.io.imread(rightimg).astype('float32'))

        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])


        # pad to width and hight to 16 times
        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
        else:
            top_pad = 0
        if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            left_pad = (times+1)*16-imgL.shape[3]
        else:
            left_pad = 0     
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        # Generating the depth map here 
        pred_disp = test(imgL,imgR)

        # Removing the padded pixels 
        if top_pad !=0 or left_pad != 0:
            img = pred_disp[top_pad:,:-left_pad]
        else:
            img = pred_disp

        img = (img*256).astype('uint16')

        imgname = leftimg.split('/')[-1].split('.')[0] + "_disparity.png"

        if not is_result_dir:
            os.makedirs(args.savedir,exist_ok=True) 

        skimage.io.imsave(os.path.join(args.savedir,imgname),img) 

if __name__ == "__main__":
    main()

