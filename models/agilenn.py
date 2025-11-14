import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .ResNet import *
from .channel import *
from .STE_optimizer import *
from .quantization_part import *
from .Huffman_encode_and_decode import *

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, types = 1, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.type = types
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        # layers = [conv_3x3_bn(3, input_channel, 2)]
        if types == 1:
            input_channel = 19
            layers = []
        elif types == 2:
            # input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
            input_channel = 24
            layers = [conv_3x3_bn(3, input_channel, 2)]
            self.start = nn.Sequential(*layers)
        else:
            input_channel = 24
            layers = []
        layers = []
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self._initialize_weights()

    def forward(self, x, models, snr, centers, model_name, cr): # , models, snr, centers
        if self.type == 2:
            x = self.start(x)
            x_ = x.clone()

            # ========= SemanticCommunication ==========
            output_shape = x.shape
            codeSend = x.flatten(2).permute(0, 2, 1) # 256*1024*64
            [codeSent, codeWithNoise, codeReceived, count, x_q] = models(codeSend, snr, centers, model_name, cr)
            x = codeReceived.permute(0, 2, 1).reshape(output_shape) # 256*64*32*32
            # ========= communnication ==========

            
            x_out = self.features(x_)
            x_out = self.conv(x_out)
            x_out = self.avgpool(x_out)
            x_out = x_out.view(x_out.size(0), -1)
            x_out = self.classifier(x_out)
            

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.type == 2:
            return codeSent, codeWithNoise, x, codeSend, codeReceived, x_out, count, x_q
        else:
            return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




class remote_MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, types = 1, width_mult=1.):
        super(remote_MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.type = types
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        # layers = [conv_3x3_bn(3, input_channel, 2)]
        if types == 1:
            input_channel = 19
            layers = []
        elif types == 2:
            # input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
            input_channel = 24
            layers = [conv_3x3_bn(3, input_channel, 2)]
            self.start = nn.Sequential(*layers)
        else:
            input_channel = 24
            layers = []
        layers = []
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x, models=0, snr=0, centers=0, model_name=0, cr=0, use_model = 0): # , models=0, snr=0, centers=0, use_model = 0
        if self.type == 2:
            x = self.start(x)
        elif self.type == 3:
            if model_name == 'MobilenetonImageNet':
                output_shape = [-1, 24, 112, 112]
            else:
                output_shape = [-1, 24, 32, 32]
            if models:
                [codeSent, codeWithNoise, codeReceived, count, x_q] = models(x, snr, centers, model_name, cr, 0)
                x = codeReceived.permute(0, 2, 1).reshape(output_shape) 
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class feature_extractor(nn.Module):
    def __init__(self, C_in, C_out):
        super(feature_extractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=3, stride=3, padding=1, groups=C_in, bias=False),
            nn.Conv2d(C_in, 4, kernel_size=3, stride=1, padding=1, bias=False),  # separable conv
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, groups=4, bias=False),
            nn.Conv2d(4, C_out, kernel_size=3, stride=1, padding=1, bias=False), # separable conv
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        # use Xavier parameter initialization method to initialize the weight parameters
        # nn.init.xavier_uniform_(self.conv1[0].weight)
        # nn.init.xavier_uniform_(self.conv1[1].weight)
        # nn.init.xavier_uniform_(self.conv2[0].weight)
        # nn.init.xavier_uniform_(self.conv2[1].weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        return x


class local_predictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(local_predictor, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size)
        )

    def forward(self, x):
        x = x.mean(dim=-1)
        x = x.mean(dim=-1) #全局平均池化
        x = self.dense(x)
        return x


class agilenn(nn.Module):
    def __init__(self, C_in, C_out, output_size, split_ratio=0.2): # cin=3 cout=24
        super(agilenn, self).__init__()
        self.sigma = 1
        self.K_top = int(np.round(split_ratio * C_out))

        self.feature_extractor_1 = feature_extractor(C_in, C_out)
        self.local_predictor_1 = local_predictor(self.K_top, output_size)
        self.remote_predictor_1 = remote_MobileNetV2(output_size)
        self.full_predictor = remote_MobileNetV2(output_size,2)
        self.quantizer = quanti_STE_2.apply
        
    def feature_splitter(self, features): # (32, 24, 75, 75) split_ratio = 20%
        top_features = features[:, :self.K_top, :, :]
        bottom_features = features[:, self.K_top:, :, :]
        return top_features, bottom_features

    def forward(self, x, centers, SNR):
        features = self.feature_extractor_1(x)
        top_features, bottom_features = self.feature_splitter(features)
        local_logits = self.local_predictor_1(top_features) #(32, 5, 75, 75)
        
        q_bottom_features = self.quantizer(bottom_features, centers.cuda(), self.sigma, 'NCHW')
        

        # # simulate binary encoding
        q_bottom_features = ascii_encode_v2(q_bottom_features, centers.cuda(), snr=SNR)
        
        
        remote_logits = self.remote_predictor_1(q_bottom_features) #(32, 19, 75, 75)
        reweighted_final_outs = local_logits + remote_logits
        

        return local_logits, remote_logits, reweighted_final_outs, features


class agilenn_resnet(nn.Module):
    def __init__(self, C_in, C_out, output_size, split_ratio): # cin=3 cout=24
        super(agilenn_resnet, self).__init__()
        self.sigma = 1
        self.K_top = int(np.round(split_ratio * C_out))

        self.feature_extractor_1 = feature_extractor(C_in, C_out)
        self.local_predictor_1 = local_predictor(self.K_top, output_size)
        # self.remote_predictor_1 = remote_MobileNetV2(output_size)
        self.remote_predictor_1 = agilenn_ResNet(output_size)
        self.full_predictor = remote_MobileNetV2(output_size,2)
        self.quantizer = quanti_STE_2.apply
        
    def feature_splitter(self, features): # (32, 24, 75, 75) split_ratio = 20%
        top_features = features[:, :self.K_top, :, :]
        bottom_features = features[:, self.K_top:, :, :]
        return top_features, bottom_features

    def forward(self, x, centers, SNR): #, SNR
        features = self.feature_extractor_1(x)
        top_features, bottom_features = self.feature_splitter(features)
        local_logits = self.local_predictor_1(top_features) #(32, 5, 75, 75)
        
        #, symbols_hard
        q_bottom_features = self.quantizer(bottom_features, centers.cuda(), self.sigma, 'NCHW')
        
        # # huffman编解码
        # x_shape = q_bottom_features.shape
        # x = q_bottom_features
        # huffman_str, codes, centers = huffman_encode(x)
        # codeWithNoise = AWGN_channel_bitstream(huffman_str, snr=SNR)
        # decoded_x = huffman_decode(codeWithNoise, centers, codes) #huffman_str
        # length = x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]
        # if len(decoded_x) >= length:
        #     decoded_x = decoded_x[:length]
        # else:
        #     appends_decoded = torch.ones(length-len(decoded_x))*centers[0]
        #     decoded_x = torch.cat((decoded_x, appends_decoded))
        # decoded_x = decoded_x.view(x.shape).cuda()
        # q_bottom_features = decoded_x

        # # 模拟二进制编码
        # q_bottom_features = ascii_encode_v2(q_bottom_features, centers.cuda(), snr=SNR)
        
        
        remote_logits = self.remote_predictor_1(q_bottom_features) #(32, 19, 75, 75)
        reweighted_final_outs = local_logits + remote_logits
        

        return local_logits, remote_logits, reweighted_final_outs, features
