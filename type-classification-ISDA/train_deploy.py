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
import resnet_isda
from resnet_isda import resnet18
from resnet_isda import resnet50
from resnet_isda import resnet101
from resnet_isda import resnet152
import time
import sys
import os
from os.path import exists
from ISDA import EstimatorCV, ISDALoss, Full_layer
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE



def test(model, fc, criterion, batch_size, input_size):
    model.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    if use_cuda:
        device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/data/cls/tests/head', transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)

            features = model(inputs)
            outputs = fc(features)
            # outputs = model(inputs)

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


def train(epoch, batch_size, store_name, model_path=None, learning_rate=0.01):
    time_start = time.time()
    input_size = 224
    exp_dir = 'resnet18'
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    store_name = './model/resnet18'
    transform_train = transforms.Compose([transforms.Resize((input_size, input_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/data/cls/combine_new', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)



    model = load_model(model_name='resnet18', pretrain=False)
    fc = Full_layer(feature_num=512, class_num=15)
    model = torch.nn.DataParallel(model).cuda()
    fc = torch.nn.DataParallel(fc).cuda()

    if exists(store_name + '/model_state_best.pt'):
        print('loading model_state_best.pt.')
        model.load_state_dict(torch.load(store_name + '/model_state_best.pt'))
        # model =torch.load(store_name + '/model_state.pth')
    else:
        print('building new model.')
    if exists(store_name + '/fc_state_best.pt'):
        print('loading fc_state_best.pt.')
        fc.load_state_dict(torch.load(store_name + '/fc_state_best.pt'))
    else:
        print('building new fc.')
    # GPU

    batch_size = 2048
    epochs = 5

    # Initialize the loss function

    isda_criterion = ISDALoss(feature_num=512, class_num=15).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    max_val_acc = 81.2500
    for t in range(epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        # def train_loop(train_loader, model, loss_fn, optimizer):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        batch_total = len(train_loader)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            idx = batch_idx
            # Compute prediction and loss
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # Backpropagation
            optimizer.zero_grad()

            loss, outputs = isda_criterion(model, fc, inputs, targets, ratio=0.1)
            # outputs = model(inputs)
            # loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 10 == 0:
                print('Epoch %d Step: %d/%d | Loss1: %.3f |  Acc: %.3f%% (%d/%d)' % (
                    t + 1, batch_idx, batch_total, train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
            # if batch % 100 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)

        val_acc, val_loss = test(model,fc, ce_criterion, 3, input_size)
        torch.save(model.state_dict(), store_name + '/model_state.pt')
        torch.save(fc.state_dict(), store_name + '/fc_state.pt')
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), store_name + '/model_state_best.pt')
            torch.save(fc.state_dict(), store_name + '/fc_state_best.pt')
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc, val_loss))
        print("Acc: %.4f" % (val_acc))
        print("Max Acc: %.4f" % (max_val_acc))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')


train(epoch=200,  # number of epoch
      batch_size=2048,  # batch size
      store_name='',  # folder for output
      model_path='',
      learning_rate=float(sys.argv[1])
      )
