# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
# from PIL import Image
from torch.utils.data import DataLoader
import os
import shutil
from collections import OrderedDict
import onnx
import onnxoptimizer
from onnxsim import simplify
from torch.onnx import OperatorExportTypes
import io
import resnet
from resnet import resnet18
from ISDA import EstimatorCV, ISDALoss, Full_layer


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


N_CLASSES = 13

net = resnet18(num_classes = 13) #13
# net = Full_layer(feature_num=512, class_num=13)
net = torch.nn.DataParallel(net).cuda()
state_dictBA = torch.load('./resnet18/model_state_best.pt')
net.load_state_dict(state_dictBA)
net.eval()
inputs = torch.randn(1, 3, 224, 224).cuda()
# inputs = torch.randn(512).cuda()


torch.onnx._export(net.module, inputs, "carcolor_RESNET18.onnx", verbose=True, opset_version=11)

print('ok')
