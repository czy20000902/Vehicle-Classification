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
import sys
import os
from os.path import exists
from ISDA import EstimatorCV, ISDALoss


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


def test(epoch, batch_size, store_name, resume=False, model_path=None, backbone="resnet18", test_path='head'):
    input_size = 224
    exp_dir = backbone
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    store_name = './model/' + backbone

    model = load_model(model_name=backbone, pretrain=False)
    fc = Full_layer(feature_num=512, class_num=15)
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

    batch_size = 512
    epochs = 5

    # Initialize the loss function

    ce_criterion = nn.CrossEntropyLoss().cuda()

    max_val_acc = 0
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0

    transform_test = transforms.Compose([transforms.Resize((input_size, input_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_data = torchvision.datasets.ImageFolder(root='/home/chenzhaoyang/data/cls_14/tests/' + test_path,
                                                 transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=4)
    size_total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        size_total += len(inputs)
        idx = batch_idx
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)

            features = model(inputs)
            outputs = fc(features)

            loss = ce_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # if batch_idx % 10 == 0:
        # print('Step: %d | Loss: %.3f | Combined Acc: %.3f%% (%d/%d)' % (
        #     batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)

    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
            epoch, test_acc, test_loss))
    # print(test_path,": Average Acc: %.4f" % (test_acc))
    print("%.4f" % (test_acc), size_total)



test(epoch=200,  # number of epoch
     batch_size=256,  # batch size
     store_name='',  # folder for output
     resume=False,  # resume training from checkpoint
     model_path='',
     backbone=sys.argv[1],
     test_path=sys.argv[2])
