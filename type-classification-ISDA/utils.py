import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model2 import *
from resnet import *
from pytorchcv.model_provider import get_model as ptcv_get_model


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'test':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
        
    if model_name == 'resnet50':
        net = ptcv_get_model("resnet50", pretrained=pretrain)
        net.output = torch.nn.Linear(2048, 2048)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
    
    if model_name == 'resnet101':
        net = ptcv_get_model("resnet101", pretrained=pretrain)
        net.output = torch.nn.Linear(2048, 2048)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
    
    if model_name == 'resnet152':
        net = ptcv_get_model("resnet152", pretrained=pretrain)
        net.output = torch.nn.Linear(2048, 2048)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
    
    if model_name == 'efficientnet_b0':
        net = ptcv_get_model("efficientnet_b0", pretrained=pretrain)
        net.output.fc = torch.nn.Linear(1280, 2048)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
    
    if model_name == 'efficientnet_b1':
        net = ptcv_get_model("efficientnet_b1", pretrained=pretrain)
        net.output.fc = torch.nn.Linear(1280, 2048)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
    
    if model_name == 'efficientnet_b2c':
        net = ptcv_get_model("efficientnet_b2c", pretrained=pretrain)
       
        net.output = torch.nn.Linear(2048, 2048)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
        
    if model_name == 'efficientnet_b3b':
        net = ptcv_get_model("efficientnet_b3b", pretrained=pretrain)
        net.output = torch.nn.Linear(1536, 2048)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
        
    if model_name == 'efficientnet_b4b':
        net = ptcv_get_model("efficientnet_b4b", pretrained=pretrain)
        print(net)
        net.output = torch.nn.Linear(1792, 2048)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 810)
        
    return net


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws

def jigsaw_generator_2(images, n):
    
    image_size = 448
    bock_nums = 28
    pixels_per_block = 16
    
    # 1, row exchange
    block_spliter = np.linspace(1,bock_nums-1,bock_nums-1)
    random.shuffle(block_spliter)
    block_spliter = block_spliter[:n]
    block_spliter[-1] = bock_nums
    block_spliter = np.sort(block_spliter, axis=0)
    row_exchange = []
    for i in range(n):
        sp = block_spliter[i]
        if i == 0:
            st = 0
        else:
            st = block_spliter[i-1]
        row_exchange.append([st, sp])
    random.shuffle(row_exchange)
    new_image = torch.zeros(images.shape).cuda()
    pinting_len = 0
    for i,stsp in enumerate(row_exchange):
        stpix = int(stsp[0]) * pixels_per_block
        sppix = int(stsp[1]) * pixels_per_block
        
        st = pinting_len
        pinting_len += sppix-stpix
        new_image[:,:,st:sppix-stpix+st,:] = images[:,:,stpix:sppix,:]
    
    # 2, line cxchagne
    block_spliter = np.linspace(1,bock_nums-1,bock_nums-1)
    random.shuffle(block_spliter)
    block_spliter = block_spliter[:n]
    block_spliter[-1] = bock_nums
    block_spliter = np.sort(block_spliter, axis=0)
    line_exchange = []
    for i in range(n):
        sp = block_spliter[i]
        if i == 0:
            st = 0
        else:
            st = block_spliter[i-1]
        line_exchange.append([st, sp])
    random.shuffle(line_exchange)
    
    new_image2 = torch.zeros(images.shape).cuda()
    pinting_len = 0
    for i,stsp in enumerate(line_exchange):
        stpix = int(stsp[0]) * pixels_per_block
        sppix = int(stsp[1]) * pixels_per_block
        
        st = pinting_len
        pinting_len += sppix-stpix
        new_image2[:,:,:,st:sppix-stpix+st] = new_image[:,:,:,stpix:sppix]
    
    return new_image2
    


def test(net, criterion, batch_size, input_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Scale((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/data/FGVC8/val/val',
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs_com, _ = net(inputs)

        loss = criterion(outputs_com, targets)

        test_loss += loss.item()
        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct_com += predicted_com.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f | Combined Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct_com) / total, correct_com, total))

    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return 0, test_acc_en, test_loss


