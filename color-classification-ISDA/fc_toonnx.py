# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import numpy as np
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


def fliplr(img):
    """Flip horizontal"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


N_CLASSES = 13

# net = resnet18(num_classes = 13) #13
net = Full_layer(feature_num=512, class_num=13)
# net = torch.nn.DataParallel(net).cuda()
state_dictBA = torch.load('./resnet18/fc_state_best.pt')
# state_dictBA = torch.load('/home/chenzhaoyang/cls_isda/isda_model/resnet18/model_state_best.pt')
# state_dictBA = torch.load('/data/tumingfei/checkpoints/resnet18/2021-06-09T14:30:24.206629/resnet18-160-best.pth')
new_state_dictBA = OrderedDict()
for k, v in state_dictBA.items():
    name = k[7:]  # remove `module.`
    new_state_dictBA[name] = v
net.load_state_dict(new_state_dictBA)
net.cuda()
net.eval()
# inputs = torch.randn(1, 3, 224, 224).cuda()
inputs = torch.randn(512).cuda()


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    input_names = ['input']
    output_names = ['output']
    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            print(model)
            torch.onnx.export(
                model,
                inputs,
                f,
                # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                verbose=True,  # NOTE: uncomment this for debugging
                input_names=input_names,
                output_names=output_names,
                opset_version=10,
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    # Apply ONNX's Optimization

    all_passes = onnxoptimizer.get_available_passes()
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer", "fuse_bn_into_conv"]
    assert all(p in all_passes for p in passes)
    onnx_model = onnxoptimizer.optimize(onnx_model, passes)

    return onnx_model


onnx_model = export_onnx_model(net, inputs)

model_simp, check = simplify(onnx_model, dynamic_input_shape=True, input_shapes={'input': [512]})

model_simp = remove_initializer_from_input(model_simp)

assert check, "Simplified ONNX model could not be validated"

save_path = os.path.join('carcolor_FC.onnx')
onnx.save_model(model_simp, save_path)

print('ok')
