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
import resnet
from resnet import resnet18
from resnet import resnet50
from resnet import resnet101
from resnet import resnet152
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
    test_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/color_classification_ISDA/test',
                                                 transform=transform_test)
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

# import torch
# import os
# import numpy as np
# import torch.onnx
# import models
# from torch.autograd import Variable
# import torch.onnx as torch_onnx
# from PIL import Image
# import torchvision.transforms as transforms
# import torch.nn as nn
# from collections import OrderedDict
# import torch.nn.functional as F
# import time
# import onnx
# import caffe2.python.onnx.backend as backend
#
# arch = 'shufflenetv2'
# model_path = './pth/' + arch + '.pth'
# # model_path = './pth/shufflenetv2.pth'
# num_classes = 2
# print('==> Arch: ', arch)
# model = None
# if arch == 'mobilenetv2':
#     model = models.mobilenetv2.mobilenetv2_10(num_classes=num_classes,
#                                               input_size=112)
# elif arch == 'shufflenetv2':
#     use_gray = False
# if use_gray:
#     c = 1
# else:
#     c = 3
# model = models.shufflenet_v2.ShuffleNetV2(scale=0.5, in_channels=c, c_tag=0.5, num_classes=num_classes,
#                                           activation=nn.ReLU, SE=False,
#                                           residual=False) elif arch == 'resnet_new': model = models.resnet_new.ResNet18(
#     num_classes=num_classes)
# transform_s1 = transforms.Compose(
#     [transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
# src_image = '1.jpg'
# input_size = 112
# mean_vals = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
# std_vals = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
# infer_device = torch.device('cpu')
# net = modelbest_model = torch.load(model_path, map_location=lambda storage, loc: storage)
# state_dict = best_model["state_dict"]
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():    name = k[
#                                           7:]  # remove `module.`    new_state_dict[name] = vnet.load_state_dict(new_state_dict)net.to(infer_device)net.eval()def predict(img):    # crop = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    crop = transform_s1(img).unsqueeze(0).cpu()    t1 = time.time()    with torch.no_grad():        out = net(crop)    print(F.softmax(out, dim=1))    _, predicted = torch.max(out, 1)    classIndex_ = predicted[0]    print('result:', classIndex_)    t2 = time.time()    print('predict time:',t2-t1)pil_im = Image.open(src_image).convert('RGB')predict(pil_im)#----------export---------#input_names = ["input0"]output_names = ["output0"]x = transform_s1(pil_im).unsqueeze(0).cpu()output_onnx = "torch_model.onnx"#添加 keep_initializers_as_inputs 可以解决 key error的问题torch_out = torch.onnx._export(net,                               x,                               output_onnx,                               export_params=True,                               verbose=False,                               input_names=input_names,                               output_names=output_names,                               keep_initializers_as_inputs=True)print("Export of torch_model.onnx complete!")print('\n\n')is_compare_caffe_torch = Trueif is_compare_caffe_torch:    print('='*80)    print("==> Loading and checking exported model from '{}'".format(output_onnx))    onnx_model = onnx.load(output_onnx)    onnx.checker.check_model(onnx_model)  # assuming throw on error    print("==> Passed")    print("==> Loading onnx model into Caffe2 backend and comparing forward pass.".format(output_onnx))    caffe2_backend = backend.prepare(onnx_model)    B = {"input0": x.data.numpy()}    c2_out = caffe2_backend.run(B)["output0"]    print("==> compare torch output and caffe2 output")    np.testing.assert_almost_equal(torch_out.data.numpy(), c2_out, decimal=5)    print("==> Passed")print('\n\n')print('='*80)print("==> Loading with mxnet")# verify mxnet#这里似乎是mxnet的bug暂时无法解决，onnx_runtime可以运行verify_mxnet = Falseif verify_mxnet:    import mxnet as mx    from mxnet.contrib import onnx as onnx_mxnet    sym, arg, aux = onnx_mxnet.import_model("torch_model.onnx")    print("Loaded torch_model.onnx!")    print(sym.get_internals())
