from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda
import torch.onnx as onnx
import torchvision.models as models
from models.resnet import *
import sys
import os
from os.path import exists
from ISDA import EstimatorCV, ISDALoss
import glob
import os

import cv2
import numpy as np


# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


def load_model(model_name, pretrain=False):
    # print('==> Building model..')
    if model_name == 'test':
        return resnet50(pretrained=pretrain)

    if model_name == 'resnet18':
        return resnet18(pretrained=pretrain)

    if model_name == 'resnet50':
        return resnet50(pretrained=pretrain)

    if model_name == 'resnet101':
        return resnet101(pretrained=pretrain)

    if model_name == 'resnet152':
        return resnet152(pretrained=pretrain)


def test(epoch, batch_size, store_name, resume=False, model_path=None, backbone="resnet18"):
    input_size = 224
    exp_dir = backbone
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    store_name = backbone
    try:
        os.stat(store_name)
    except:
        os.makedirs(store_name)


    to_tensor = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # img = Image.open('./1.jpg')
    # inputs = to_tensor(img)
    # inputs = np.expand_dims(inputs, 0)
    # inputs = inputs.reshape([1, inputs.shape[0], inputs.shape[1], inputs.shape[2]])
    #
    # print(inputs.size())
    # inputs = inputs.cuda()

    transform_test = transforms.Compose([transforms.Resize((input_size, input_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/color_classification_ISDA/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=4)


    model = load_model(model_name=backbone, pretrain=False)
    fc = Full_layer(feature_num=512, class_num=13)
    model = torch.nn.DataParallel(model).cuda()
    fc = torch.nn.DataParallel(fc).cuda()
    if exists('./' + store_name + '/model_state_best.pt'):
        model.load_state_dict(torch.load('./' + store_name + '/model_state_best.pt'))
    else:
        print('building new model.')
    if exists('./' + store_name + '/fc_state_best.pt'):
        fc.load_state_dict(torch.load('./' + store_name + '/fc_state_best.pt'))
    else:
        print('building new fc.')
    # GPU

    # Initialize the loss function

    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            inputs = inputs.cuda()
            inputs = Variable(inputs)

            features = model(inputs)
            print(features.cpu().numpy())
            outputs = fc(features)

            print(outputs.cpu().numpy())


test(epoch=200,  # number of epoch
     batch_size=256,  # batch size
     store_name='',  # folder for output
     resume=False,  # resume training from checkpoint
     model_path='',
     backbone='resnet18')
