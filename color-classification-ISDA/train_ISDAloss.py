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
import time
import sys
import os
from os.path import exists
from ISDA import EstimatorCV, ISDALoss, Full_layer
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
        x = nn.Softmax(x)
        return x

def test(model, fc, criterion, batch_size, input_size):
    model.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0

    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/dataset/test/hard', transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)

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


def train(epoch, batch_size, store_name, resume=False, model_path=None, backbone="resnet18", learning_rate=1e-3):
    time_start = time.time()
    input_size = 224
    exp_dir = backbone
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    store_name = backbone
    transform_train = transforms.Compose([transforms.Resize((input_size, input_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/dataset/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)



    model = load_model(model_name=backbone, pretrain=False)
    fc = Full_layer(feature_num=512, class_num=13)
    # GPU
    model = torch.nn.DataParallel(model).cuda()
    fc = torch.nn.DataParallel(fc).cuda()
    if exists('./' + store_name + '/model_state_best.pt'):
        print('using existed model.')
        model.load_state_dict(torch.load('./' + store_name + '/model_state_best.pt'))
    else:
        print('building new model.')
    if exists('./' + store_name + '/fc_state_best.pt'):
        print('using existed fc.')
        fc.load_state_dict(torch.load('./' + store_name + '/fc_state_best.pt'))
    else:
        print('building new fc.')

    batch_size = 64
    epochs = 5

    # Initialize the loss function

    isda_criterion = ISDALoss(feature_num=512, class_num=13).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    max_val_acc = 0
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

            loss, outputs = isda_criterion(model, fc, inputs, targets, ratio=0.5)
            # outputs = model(inputs)
            # loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 10 == 0:
                print('%s Step: %d/%d | Loss1: %.3f |  Acc: %.3f%% (%d/%d)' % (
                    backbone, batch_idx, batch_total, train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
            # if batch % 100 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)

        # if epoch < 5 or epoch >= 80 or True:
        val_acc, val_loss = test(model,fc, ce_criterion, 3, input_size)
        torch.save(model.state_dict(), './' + store_name + '/model_state.pt')
        torch.save(fc.state_dict(), './' + store_name + '/fc_state.pt')
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), './' + store_name + '/model_state_best.pt')
            torch.save(fc.state_dict(), './' + store_name + '/fc_state_best.pt')
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc, val_loss))
        print("Max Acc: %.4f" % (max_val_acc))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')


train(epoch=200,  # number of epoch
      batch_size=1024,  # batch size
      store_name='',  # folder for output
      resume=False,  # resume training from checkpoint
      model_path='',
      backbone=sys.argv[1],
      learning_rate=float(sys.argv[2]))
