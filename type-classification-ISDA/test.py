from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.onnx as onnx
import torchvision.models as models
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda
import resnet
from resnet import resnet50
from resnet import resnet101
from resnet import resnet152
import time
import sys
import os
from os.path import exists


def evaluate(model, criterion, batch_size, input_size, mode):
    model.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    if use_cuda:
        device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Scale((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/dataset/test/' + mode,
                                                 transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f | Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    test_acc_en = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss


def load_model(model_name, pretrain=False):
    print('==> Building model..')
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


def test(batch_size, model_path=None, backbone="resnet50", mode='hard'):
    input_size = 224

    exp_dir = backbone
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    # transform_train = transforms.Compose([
    #     transforms.Resize((input_size, input_size)),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # test_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/dataset/test/' + mode,
    #                                              transform=transform_train)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    #
    # # model = resnet50(pretrained=False).cuda()

    # model = load_model(model_name=backbone, pretrain=False)
    if exists('/home/chenzhaoyang/color_classification/' + backbone + '/model.pth'):
        print('using existed model.')
        model = torch.load('/home/chenzhaoyang/color_classification/' + backbone + '/model.pth')
    else:
        print('building new model.')
        model = load_model(model_name=backbone, pretrain=False)
    netp = model

    # GPU
    device = torch.device("cuda:0")
    model.to(device)
    # model.eval()
    learning_rate = 1e-2
    batch_size = 64
    epochs = 5

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # max_val_acc = 0
    # for t in range(epoch):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     # def train_loop(train_loader, model, loss_fn, optimizer):
    model.eval()
    # if epoch < 5 or epoch >= 80 or True:
    val_acc, val_loss = evaluate(model, loss_fn, 3, input_size, mode)
    # with open(exp_dir + '/test.txt', 'a') as file:
    #     file.write('test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
    #         val_acc, val_acc_com, val_loss))
    # print("Max Acc: %.4f" % (max_val_acc))
    print('%s %s: test_acc = %.5f, test_loss = %.6f\n' % (backbone, mode, val_acc, val_loss))


test(batch_size=64,  # batch size
     model_path='',
     backbone=sys.argv[1],
     mode=sys.argv[2])
