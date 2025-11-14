import os
import csv
from pyexpat import model
import random
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.utils.data
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision import models

from models import *
from tools import progress_bar
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

def main():
    parser = argparse.ArgumentParser(description="semanticnn with PyTorch")
    parser.add_argument('--models', default="ResNet50", type=str, help='models[ResNet50,MobileNetv2,semanticnn]')
    parser.add_argument('--testonly', default=False, type=bool, help='test-only')
    parser.add_argument('--basemodel', default="ResNetonImageNet", type=str, help='[ResNetonImageNet, ResNetoncifar100, MobilenetonImageNet, Mobilenetoncifar100]')
    parser.add_argument('--stage', default=1, type=int, help='stage[1,2]')
    parser.add_argument('--useXAI', default=False, type=bool, help='use-XAI')
    parser.add_argument('--cr', default=0, type=float, help='compression-ratio [0,1]')
    parser.add_argument('--deepcodkd', default=False, type=bool, help='deepcod-kd')
    parser.add_argument('--datasets', default="cifar10", type=str, help='datasets[cifar10,cifar100,imagenet,satellite]')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=30, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=32, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=128, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--gpu', default="1", type=str, help='which gpu')
    args = parser.parse_args()
    solver = Solver(args)
    solver.run()

class Solver(object):
    def __init__(self, config):
        self.model_name = config.models
        self.base_model_name = config.basemodel
        self.test_only = config.testonly
        self.stage = config.stage
        self.use_XAI = config.useXAI
        self.cr = config.cr
        self.datasets = config.datasets
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.cuda = config.cuda
        self.gpu = config.gpu
        self.deepcod_kd = config.deepcodkd
        self.num_classes = None
        self.num_centers = 8 # number of quantization

        self.train_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device_ids = [0, 1]
        self.lamda = 1  # control how much the muInfo will affect deepSC model
        self.alpha = 1
        self.beta = 0

    def load_data(self):
        if self.datasets == "cifar10":
            self.num_classes = 10
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616])
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                     transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)
        elif self.datasets == 'cifar100':
            self.num_classes = 100
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
            test_set = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                     transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)
        elif self.datasets == 'imagenet':
            self.num_classes = 200
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_set = datasets.ImageFolder('./data/tiny-imagenet-200/train', transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
            test_set = datasets.ImageFolder('./data/tiny-imagenet-200/val', transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

        elif self.datasets == 'satellite':
            self.num_classes = 4
            desired_size = (64, 64)
            transform = transforms.Compose([
                transforms.Resize(desired_size),
                transforms.ToTensor()
            ])
            dataset = ImageFolder('./data/Satellite_Image_Classification/data', transform=transform)

            torch.manual_seed(30)
            test_size = 1000
            train_size = len(dataset) - test_size

            train_set, test_set = random_split(dataset, [train_size, test_size])
            self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)
        print("Loaded {} dataset".format(self.datasets))
    
    def load_model(self):

        if self.model_name == "ResNet50":
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features 
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs,self.num_classes), nn.LogSoftmax(dim=1))
            self.model.cuda()
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(self.epochs*0.5), int(self.epochs*0.85)], gamma=0.2) # lr decay
            self.criterion = nn.CrossEntropyLoss().cuda()
            if self.test_only:
                if self.datasets == "imagenet":
                    self.model.load_state_dict(torch.load('test_models/base_model/ResNet50_imagenetbest_79.67.pt'))
                elif self.datasets == 'cifar100':
                    self.model.load_state_dict(torch.load('test_models/base_model/ResNet50_cifar100best_67.91.pt'))
                print("Loaded ResNet50 model on {} dataset".format(self.datasets))

        elif self.model_name == "MobileNetv2":
            self.model = agilenn.remote_MobileNetV2(self.num_classes,types=2).cuda()
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(self.epochs*0.5), int(self.epochs*0.85)], gamma=0.2) # lr decay
            self.criterion = nn.CrossEntropyLoss().cuda()
            if self.test_only:
                if self.datasets == "imagenet":
                    self.model.load_state_dict(torch.load('test_models/base_model/MobileNetv2_imagenetbest_63.67.pt'))
                elif self.datasets == 'cifar100':
                    self.model.load_state_dict(torch.load('test_models/base_model/MobileNetv2_cifar100best_58.26.pt'))
                print("Loaded MobileNetv2 model on {} dataset".format(self.datasets))
        
        elif self.model_name == "semanticnn":
            if self.deepcod_kd:
                if self.base_model_name.startswith('ResNet'):
                    self.model = DeepSC.deepcod().cuda()
                else:
                    self.model = DeepSC.deepcod_mobilenet().cuda()
            else:
                if self.base_model_name.startswith('ResNet'):
                    self.model = DeepSC.SemanticNN().cuda()
                else:
                    self.model = DeepSC.Semanticnn_mobilenet().cuda()
            # you can add various ADJSCC versions in DeepSC.py
            # elif xxx:

            if self.test_only:
                model_f = './save/semanticnn/semanticnn_Mobilenetoncifar100_cifar100_best_xx.pt' # change model path here
                self.model.load_state_dict(torch.load(model_f))           
                        
            if self.base_model_name == 'ResNetonImageNet':
                self.resnet_50 = ResNet.resnet50(pretrained=False)
                num_ftrs = self.resnet_50.fc.in_features 
                self.resnet_50.fc = nn.Sequential(nn.Linear(num_ftrs,self.num_classes), nn.LogSoftmax(dim=1))
                self.resnet_50.cuda()
                self.resnet_50.load_state_dict(torch.load('test_models/base_model/ResNet50_imagenetbest_79.67.pt'))
                if self.stage == 2:
                    model_f = './save/semanticnn/semanticnn_imagenet_best_xx.pt' # change stage 1 model path here
                    self.model.load_state_dict(torch.load(model_f))
            elif self.base_model_name == 'ResNetoncifar100':
                self.resnet_50 = ResNet.resnet50(pretrained=False)
                num_ftrs = self.resnet_50.fc.in_features 
                self.resnet_50.fc = nn.Sequential(nn.Linear(num_ftrs,self.num_classes), nn.LogSoftmax(dim=1))
                self.resnet_50.cuda()
                self.resnet_50.load_state_dict(torch.load('test_models/base_model/ResNet50_cifar100best_67.91.pt'))
                if self.stage == 2:
                    model_f = './save/semanticnn/semanticnn_cifar100_best_xx.pt' # change stage 1 model path here
                    self.model.load_state_dict(torch.load(model_f))
            elif self.base_model_name == 'MobilenetonImageNet': 
                self.resnet_50 = agilenn.MobileNetV2(self.num_classes,types=2).cuda()
                self.resnet_50.load_state_dict(torch.load('test_models/base_model/MobileNetv2_imagenetbest_63.67.pt'))
                if self.stage == 2:
                    model_f = './save/semanticnn/semanticnn_Mobilenetoncifar100_cifar100_best_xx.pt' # change stage 1 model path here
                    self.model.load_state_dict(torch.load(model_f))  
            elif self.base_model_name == 'Mobilenetoncifar100':
                self.resnet_50 = agilenn.MobileNetV2(self.num_classes,types=2).cuda()
                self.resnet_50.load_state_dict(torch.load('test_models/base_model/MobileNetv2_cifar100best_58.26.pt'))
                if self.stage == 2:
                    model_f = './save/semanticnn/semanticnn_Mobilenetoncifar100_cifar100_best_xx.pt' # change stage 1 model path here
                    self.model.load_state_dict(torch.load(model_f))  

            # quantization centers is very important! -> generate centers
            minval, maxval = map(int, [-1,1])
            centers = torch.rand(self.num_centers, dtype=torch.float32).cuda() * (maxval - minval) - maxval
            self.centers = nn.Parameter(centers)

            self.criterion = nn.CrossEntropyLoss().cuda()
            self.l1_loss = nn.MSELoss(reduction="mean").cuda()
            self.optimizer = optim.Adam([{'params': self.model.parameters()}, {'params': self.centers}], lr=0.0001)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(self.epochs*0.5), int(self.epochs*0.85)], gamma=0.2) # lr decay

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        train_correct_t5 = 0
        total = 0
            
        for batch_num, (data, target) in enumerate(self.train_loader):
            data_var = data.cuda()
            target_var = target.cuda()
            output = self.model(data_var)
            loss = self.criterion(output, target_var)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),True)
            self.optimizer.step()
            
            del data_var
            del target_var
            del loss

            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            # top5
            if self.num_classes > 5:
                _, pred = output.topk(5, 1, True, True)
                pred = pred.t().cpu()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                train_correct_t5 += correct[:5].contiguous().view(-1).float().sum(0)


            total += target.size(0)

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc1: %.3f%% | Acc5: %.3f%%'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, 100. * train_correct_t5 / total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        test_correct_t5 = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data_var = data.cuda()
                target_var = target.cuda()

                # self.model.zero_grad()
                # ig = IntegratedGradients(self.model)
                # attributions = ig.attribute(data_var, data_var * 0, target=target_var, n_steps= 30 ,return_convergence_delta=False)
                # print('IG Attributions:', attributions)

                output = self.model(data_var)
                loss = self.criterion(output, target_var)
                test_loss += loss.item()
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                # train_correct incremented by one if predicted right
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                
                # top5
                if self.num_classes > 5:
                    _, pred = output.topk(5, 1, True, True)
                    pred = pred.t().cpu()
                    correct = pred.eq(target.view(1, -1).expand_as(pred))
                    test_correct_t5 += correct[:5].contiguous().view(-1).float().sum(0)
                total += target.size(0)
                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc1: %.3f%% | Acc5: %.3f%%'
                         % (test_loss / (batch_num + 1), 100. * test_correct / total, 100. * test_correct_t5 / total))

        return test_loss, test_correct/ total

    def knowledge_distill(self):
        print("deepcod train:")
        self.model.train()
        self.resnet_50.eval()
        train_loss = 0
        train_correct = 0
        train_correct_t5 = 0
        total = 0
        snr = random.randint(1,12)
        for param in self.resnet_50.parameters():
            param.requires_grad = False

        for batch_num, (data, target) in enumerate(self.train_loader):
            data_var = data.cuda()
            target_var = target.cuda()
            
            ## read network intermediate layer features
            features = []

            def hook(module, input, output):
                # module: model.conv2
                # input :in forward function  [#2]
                # output:is  [#3 self.conv2(out)]
                features.append(output.detach())
            if self.base_model_name.startswith('ResNet'):
                hook1 = self.resnet_50.layer1.register_forward_hook(hook)
                hook2 = self.resnet_50.layer2.register_forward_hook(hook)
                hook3 = self.resnet_50.layer3.register_forward_hook(hook)
                hook4 = self.resnet_50.layer4.register_forward_hook(hook)
            else:
                hook1 = self.resnet_50.features[0].register_forward_hook(hook)
                hook2 = self.resnet_50.features[1].register_forward_hook(hook)
                hook3 = self.resnet_50.features[2].register_forward_hook(hook)
                hook4 = self.resnet_50.features[3].register_forward_hook(hook)

            [c_inputs, c_noise, output, c_send, c_received, output_, count, feature_q]  = self.resnet_50(data_var, self.model, snr, self.centers, self.base_model_name, self.cr)

            hook1.remove()
            hook2.remove()
            hook3.remove()
            hook4.remove()

            
            loss = 0
            # add regularization term for orthogonal regularization of convolution kernels
            loss += 0.0001 * conv_regulation.loss_regulation(self.model)

            for i in range(4):
                loss += self.l1_loss(features[i], features[i + 4])


            ## loss function backpropagation
            loss += self.l1_loss(output, output_) + self.criterion(output, target_var)
            train_loss += loss.item()
        

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),True)
            self.optimizer.step()
        
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t().cpu()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            train_correct_t5 += correct[:5].contiguous().view(-1).float().sum(0)


            total += target.size(0)

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc1: %.3f%% | Acc5: %.3f%%'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, 100. * train_correct_t5 / total))

        return train_loss, train_correct / total

    def train_channel(self):
        print("train SemanticNN:")
        self.model.train()
        self.resnet_50.eval()
        train_loss = 0
        train_correct = 0
        train_correct_t5 = 0
        total = 0
            
        for batch_num, (data, target) in enumerate(self.train_loader):
            snr = random.randint(-6,7)
            data_var = data.cuda()
            target_var = target.cuda()
            [c_inputs, c_noise, output, c_send, c_received, output_, count, feature_q] = self.resnet_50(data_var, self.model, snr, self.centers, self.base_model_name, self.cr)
            if self.stage == 1:
                for i in range(len(c_received)):
                    if i == 0:
                        SC_loss = self.l1_loss(c_received[i], c_send[i])
                    else:
                        SC_loss += self.l1_loss(c_received[i], c_send[i])
                loss =  SC_loss
            elif self.stage == 2:
                ce_loss = self.criterion(output, target_var)
                loss = ce_loss 

                if self.use_XAI:
                    if self.base_model_name == 'ResNetonImageNet' or self.base_model_name == 'ResNetoncifar100':
                        self.toyModel = ResNet.toymodel(pretrained=False)
                        num_ftrs = self.toyModel.fc.in_features 
                        self.toyModel.fc = nn.Sequential(nn.Linear(num_ftrs,self.num_classes), nn.LogSoftmax(dim=1))
                        self.toyModel.cuda()
                        if self.base_model_name == 'ResNetonImageNet':
                            self.toyModel.load_state_dict(torch.load('test_models/base_model/ResNet50_imagenetbest_79.67.pt'))
                        else:
                            self.toyModel.load_state_dict(torch.load('test_models/base_model/ResNet50_cifar100best_67.91.pt'))
                        self.toyModel.eval()

                        f = feature_q.clone()
                        f.requires_grad_(True)
                        y_pro = self.toyModel(f, self.model, snr, self.centers, self.base_model_name, self.cr, 1)
                        c_loss_pro = self.criterion(y_pro, target_var)
                        f.retain_grad()
                        c_loss_pro.backward(retain_graph=True)
                        grads = f.grad

                        f0 = torch.zeros_like(f)
                        f0.requires_grad_(True)
                        y_pro = self.toyModel(f0, self.model, snr, self.centers, self.base_model_name, self.cr, 1)
                        c_loss_pro = self.criterion(y_pro, target_var)
                        f0.retain_grad()
                        c_loss_pro.backward(retain_graph=True)
                        grads0 = f0.grad

                        attributions = (0.5 * (grads0 + grads) * (feature_q - f0)).view(self.train_batch_size, -1)
                        loss += 1 * (attributions.numel()/attributions[attributions>0].numel() + 1.0 / attributions[attributions>0].sum())

                    elif self.base_model_name == 'mobilenetonImageNet' or self.base_model_name == 'mobilenetoncifar100':
                        self.mobilenetv2 = agilenn.remote_MobileNetV2(self.num_classes,types=3).cuda()
                        pre_model = agilenn.remote_MobileNetV2(self.num_classes,types=2).cuda()
                        pre_model.load_state_dict(torch.load('test_models/base_model//MobileNetv2_imagenetbest_63.69.pt'))
                        for name, parms in pre_model.named_parameters():
                            if name in self.mobilenetv2.state_dict():
                                self.mobilenetv2.state_dict()[name].copy_(pre_model.state_dict()[name]) 
                        self.mobilenetv2.eval()

                        f = feature_q.clone()
                        f.requires_grad_(True)
                        y_pro = self.mobilenetv2(f)
                        c_loss_pro = self.criterion(y_pro, target_var)
                        f.retain_grad()
                        c_loss_pro.backward(retain_graph=True)
                        grads = f.grad

                        f0 = torch.zeros_like(f)
                        f0.requires_grad_(True)
                        y_pro = self.mobilenetv2(f0)
                        c_loss_pro = self.criterion(y_pro, target_var)
                        f0.retain_grad()
                        c_loss_pro.backward(retain_graph=True)
                        grads0 = f0.grad

                        attributions = torch.abs(0.5 * (grads0 + grads) * (feature_q - f0)).view(self.train_batch_size, -1)
                        loss += 0.5 * (attributions.numel()/attributions[attributions>0].numel()-1 + 1.0 / attributions[attributions>0].sum())
                    
                # add regularization term for orthogonal regularization of convolution kernels
                # loss += 0.0001 * conv_regulation.loss_regulation(self.model)

                # quantization distribution
                freq = torch.ones((8))*(1.0/8)
                loss += 2 * self.l1_loss(count, freq.cuda())
            train_loss += loss.item()
        
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),True)
            self.optimizer.step()
        
            prediction = torch.max(output, 1) 
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t().cpu()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            train_correct_t5 += correct[:5].contiguous().view(-1).float().sum(0)
            total += target.size(0)

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc1: %.3f%% | Acc5: %.3f%%'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, 100. * train_correct_t5 / total))

        return train_loss, train_correct / total

    def test_channel(self):
        print("test SemanticNN:")
        self.model.eval()
        self.resnet_50.eval()
        test_loss = 0
        test_correct = 0
        test_correct_t5 = 0
        total = 0
        counts = torch.zeros((8)).cuda()
        importants_num_rate = 0
        importants_sum = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                snr = random.randint(-6,7)
                data_var = data.cuda()
                target_var = target.cuda()
                [c_inputs, c_noise, output, _ , _, _, count, feature_q] = self.resnet_50(data_var, self.model, snr, self.centers, self.base_model_name, self.cr)
                counts += count*(1.0/10000)

                ce_loss = self.criterion(output, target_var)
                loss = ce_loss
                    
                test_loss += loss.item()
                total += target.size(0)
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                    
                _, pred = output.topk(5, 1, True, True)
                pred = pred.t().cpu()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                test_correct_t5 += correct[:5].contiguous().view(-1).float().sum(0)
                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc1: %.3f%% | Acc5: %.3f%%'
                            % (test_loss / (batch_num + 1), 100. * test_correct / total, 100. * test_correct_t5 / total))
            
        
        return test_loss, test_correct/ total

    def save(self,name):
        folder_path = "./save/"+str(self.model_name)+"/"
        model_out_path = "./save/"+str(self.model_name)+"/"+str(self.model_name)+"_"+str(self.base_model_name)+"_"+str(self.datasets)+name+".pt"
        if not os.path.exists(folder_path):
            # create
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' have been created")
        torch.save(self.model.state_dict(), model_out_path)
        # model_out_path = "./save/"+str(self.model_name)+"/"+str(self.model_name)+"_"+str(self.datasets)+"mobilenetv2_v3.pt"
        # torch.save(self.mobilenetv2.state_dict(), model_out_path)
        # torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        if self.model_name == 'semanticnn':
            with open(self.centers_path, 'w') as f:
                f.write(str(self.centers.tolist()))
                f.close()

    def run(self):
        self.load_data()
        accuracy = 0
        self.centers_path = './save/centers/'+str(self.model_name)+"_"+str(self.base_model_name)+"_"+str(self.datasets)+".txt"

        if os.path.exists(self.centers_path):
            with open(self.centers_path, 'r') as f1:
                self.centers = torch.Tensor(eval(f1.read()))
            f1.close()
        self.load_model()

        if self.model_name == 'semanticnn':
            # train and test
            if not(self.test_only):  
                for epoch in range(1, self.epochs + 1):
                    print("\n===> epoch: %d/%s" % (epoch, self.epochs))

                    if self.deepcod_kd:
                        train_result = self.knowledge_distill()
                    else:
                        train_result = self.train_channel()
                    self.scheduler.step()
                    test_result = self.test_channel()
                    if test_result[1] > accuracy:
                        accuracy = test_result[1]
                        print("===> BEST ACC PERFORMANCE: %.3f%%" % (accuracy))
                        self.save("_best_"+str(round(accuracy,3)))
                    if (epoch) % 100 == 0:
                        print("===> BEST ACC PERFORMANCE: %.3f%%" % (accuracy))
                        self.save("_epoch_"+str(epoch+1))
                
            # only test
            else:
                print("Repeat tests for 5 times!")
                for epoch in range(1, 6): # repeat 5 times
                    print("\n===> epoch: %d/%s" % (epoch, 5))
                    test_result = self.test_channel()
                    if test_result[1] > accuracy:
                        accuracy = test_result[1]
                        print("===> BEST ACC PERFORMANCE: %.3f%%" % (accuracy))

        else:
            # base model train 
            if not(self.test_only):
                for epoch in range(1, self.epochs + 1):
                    print("\n===> epoch: %d/%s" % (epoch, self.epochs))
                    train_result = self.train()
                    self.scheduler.step()
                    print(train_result)
                    test_result = self.test()
                    if test_result[1] > accuracy:
                        accuracy = test_result[1]
                        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                        self.save("best_"+str(accuracy * 100))
                    if (epoch+1) % 100 == 0:
                        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                        self.save("epoch_"+str(epoch+1))   
            # base model only test
            else:
                test_result = self.test()
                if test_result[1] > accuracy:
                    accuracy = test_result[1]
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))


if __name__ == '__main__':
    main()