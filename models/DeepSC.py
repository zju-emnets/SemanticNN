import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from .channel import *
from .gdn import GDN
from .STE_optimizer import *
from .quantization_part import *
from .Huffman_encode_and_decode import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def embedding(input_size, output_size):  # embedding layer, the former is the size of dic and
    # the latter is the dimension of the embedding vector
    return nn.Embedding(input_size, output_size)

def dense(input_size, output_size):  # dense layer is a full connection layer and used to gather information
    return torch.nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.ReLU()
    )
def depconv(C_in, C_out, kernel_size=3, stride=1):
    return torch.nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size, stride, kernel_size//2, groups=C_in, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
        )
def depconv_trans(C_in, C_out, kernel_size=3, stride=1):
    return torch.nn.Sequential(
        nn.ConvTranspose2d(C_in, C_out, kernel_size, stride, kernel_size//2, groups=C_in, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
        )
def hw_flatten(x):
    x_shape = x.shape
    return x.view(-1, x_shape[1], x_shape[2] * x_shape[3])


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, channel_split, pool_size, pool_stride):
        super(Self_Attn, self).__init__()
        # self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // channel_split, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // channel_split, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.gamma = torch.autograd.Variable(torch.ones(1))
        self.after_attention_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mpool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        self.softmax = nn.Softmax(dim=-1)  #
        # torch.nn.init.kaiming_normal_(self.query_conv.weight)
        # torch.nn.init.kaiming_normal_(self.key_conv.weight)
        # torch.nn.init.kaiming_normal_(self.value_conv.weight)
        # torch.nn.init.kaiming_normal_(self.after_attention_conv.weight)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B, C, W, H)
            returns :
                out : self attention value + input feature
                attention: (B, N, N) (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = hw_flatten(self.mpool(self.query_conv(x))).permute(0, 2, 1)  # (B, N2, C2)
        proj_key = hw_flatten(self.key_conv(x))  # (B, C2, N)
        # matrix multiplication
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # (B, N2, N)

        proj_value = hw_flatten(self.mpool(self.value_conv(x)))  # (B, C2, N2)
        out = torch.bmm(proj_value, attention)  # (B, C2, N)
        out = out.view(m_batchsize, -1, width, height)
        out = self.after_attention_conv(out)
        out = self.gamma * out + x
        return out

class Residual_Transposed_Conv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Residual_Transposed_Conv, self).__init__()
        self.chanel_in = in_dim
        self.out_channels = out_dim
        self.conv_t1 = nn.ConvTranspose2d(self.chanel_in, self.out_channels, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv_t2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv_t3 = nn.ConvTranspose2d(self.chanel_in, self.out_channels, 3, stride=2, padding=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.LeakyReLU()
        # torch.nn.init.kaiming_normal_(self.conv_t1.weight)
        # torch.nn.init.kaiming_normal_(self.conv_t2.weight)
        # torch.nn.init.kaiming_normal_(self.conv_t3.weight)

    def forward(self, x):
        out_on = self.relu(self.bn1(self.conv_t1(x)))
        out_on = self.relu(self.bn2(self.conv_t2(out_on)))
        out_down = self.conv_t3(x)
        out = out_on + out_down
        return out

class AF(nn.Module):
    def __init__(self,shape):
        super(AF,self).__init__()
        self.factor_prediction = nn.Sequential(
            nn.Conv1d(shape+1,shape,1),
            nn.ReLU(),
            nn.Conv1d(shape,shape,1),
            nn.Sigmoid()
        )

    def forward(self,x,SNR):
        temp = F.adaptive_avg_pool2d(x,(1,1))
        snr_ones = torch.ones(x.shape[0],1,1,1).to(device)
        temp = torch.concat([temp,SNR*snr_ones],dim=1)
        temp = self.factor_prediction(torch.squeeze(temp,dim=3))
        temp = torch.unsqueeze(temp,dim=2)
        x *= temp
        return x
class FL1(nn.Module):
    def __init__(self,F,D_in,D_out,S,P):
        super(FL1,self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=D_in,
                      out_channels=D_out,
                      kernel_size=F,
                      stride=S,
                      padding=P,
                      groups=1)
        )
        self.gdn = GDN(D_out,device)
        self.last = nn.PReLU()

    def forward(self,x):
        x = self.module(x)
        x = self.gdn.forward(x)
        x = self.last(x)
        return x
class FL2(nn.Module):
    def __init__(self,F,D_in,D_out,S,P):
        super(FL2,self).__init__()
        self.module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=D_in,
                      out_channels=D_out,
                      kernel_size=F,
                      stride=S,
                      padding=P)
        )
        self.gdn = GDN(D_out,device ,inverse=True)
        self.last = nn.PReLU()

    def forward(self,x):
        x = self.module(x)
        x = self.gdn.forward(x)
        x = self.last(x)
        return x
# 4+4
class ADJSCC(nn.Module):
    def __init__(self):
        super(ADJSCC, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(3,64,96,1,0)
        self.af1 = AF(96)
        self.fl2 = FL1(3,96,96,2,1)
        self.af2 = AF(96)
        self.fl3 = FL1(3,96,64,2,1)
        self.af3 = AF(64)
        self.fl4 = FL1(3,64,32,2,1)
        self.quantizer = quanti_STE_2.apply

        self.fl_1 = FL2(3,32,64,2,1)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,2,0)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,96,2,0)
        self.af_3 = AF(96)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(96,64,2,1,0)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr):
        if model_name == 'ResNetonImageNet':
            shape = [-1, 64, 56, 56] # resnet on imagenets
        elif model_name == 'ResNetoncifar100':
            shape = [-1, 64, 8, 8] # resnet on cifar100
        x = inputs.permute(0, 2, 1).reshape(shape)
        x = self.fl1.forward(x)
        x = self.af1.forward(x,SNR)
        x = self.fl2.forward(x)
        x = self.af2.forward(x,SNR)
        x = self.fl3.forward(x)
        x = self.af3.forward(x,SNR)
        x = self.fl4.forward(x)
        temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
        self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
        x = x * (1 / self.norm) * math.sqrt(64*4*4)
        x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
        count = torch.zeros((8)).cuda()
        for i in range(8):
            count[i] = (torch.sum(x == centers[i]).float()/x.numel())
    
        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR) 
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================

        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, count

# 3+3
class ADJSCC2(nn.Module):
    def __init__(self):
        super(ADJSCC2, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(2,64,128,1,2)
        self.af1 = AF(128)
        self.fl2 = FL1(3,128,64,2,0)
        self.af2 = AF(64)
        self.fl3 = FL1(3,64,32,2,0)
        self.quantizer = quanti_STE_2.apply

        self.fl_2 = FL2(3,32,64,2,0)
        self.af_2 = AF(64)
        self.fl_3 = FL2(3,64,128,2,0)
        self.af_3 = AF(128)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(128,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr):
        if model_name == 'ResNetonImageNet':
            shape = [-1, 64, 56, 56] # resnet on imagenets
        elif model_name == 'ResNetoncifar100':
            shape = [-1, 64, 8, 8] # resnet on cifar100
        x = inputs.permute(0, 2, 1).reshape(shape)
        x = self.fl1.forward(x)
        x = self.af1.forward(x,SNR)
        x = self.fl2.forward(x)
        x = self.af2.forward(x,SNR)
        x = self.fl3.forward(x)
        temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
        self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
        x = x * (1 / self.norm) * math.sqrt(64*4*4)

        x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')

        count = torch.zeros((8)).cuda()
        for i in range(8):
            count[i] = (torch.sum(x == centers[i]).float()/x.numel())
        x_q = x.clone()

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR) 
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================

        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)
        return codeSent, codeWithNoise, codeReceived, count, x_q
# 1+4
class ADJSCC3(nn.Module):
    def __init__(self):
        super(ADJSCC3, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(4,64,32,4,0)
        # self.af1 = AF(32)
        self.quantizer = quanti_STE_2.apply
        self.fl_1 = FL2(3,32,64,1,1)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,2,0)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,128,2,0)
        self.af_3 = AF(128)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(128,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr):
        if model_name == 'ResNetonImageNet':
            shape = [-1, 64, 56, 56] # resnet on imagenets
        elif model_name == 'ResNetoncifar100':
            shape = [-1, 64, 8, 8] # resnet on cifar100
        x = inputs.permute(0, 2, 1).reshape(shape)
        x = self.fl1.forward(x)
        temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
        self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
        x = x * (1 / self.norm) * math.sqrt(64*4*4)
        x_shape = x.shape
        x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
        count = torch.zeros((8)).cuda()
        for i in range(8):
            count[i] = (torch.sum(x == centers[i]).float()/x.numel())

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR) 
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================

        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)
        return codeSent, codeWithNoise, codeReceived, count

# 1+5
class ADJSCC4_more(nn.Module):
    def __init__(self):
        super(ADJSCC4, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(4,64,32,2,0)
        # self.af1 = AF(32)
        self.quantizer = quanti_STE_2.apply
        self.fl_1 = FL2(3,32,64,1,1)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,1,0)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,128,2,1)
        self.af_3 = AF(128)
        self.fl_4 = FL2(3,128,96,2,2)
        self.af_4 = AF(96)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(96,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr, flag = 1):
        x = inputs
        count = torch.zeros((8)).cuda()
        if flag:
            if model_name == 'ResNetonImageNet':
                shape = [-1, 64, 56, 56] # resnet on imagenets
            elif model_name == 'ResNetoncifar100':
                shape = [-1, 64, 8, 8] # resnet on cifar100
            x = inputs.permute(0, 2, 1).reshape(shape)
            x = self.fl1.forward(x)
            temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
            self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
            x = x * (1 / self.norm) * math.sqrt(64*4*4)
            x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
         
            # ========== compress version
            cr = 0  #compress ratio
            x_l = x.shape[2]
            x_size = x.shape[2]*x.shape[3]
            x_length = torch.sqrt(x_size * torch.tensor(1-cr))
            padding_length = int(torch.round(0.5*(x_l - x_length)))
            padding_end = padding_length+int(x_length)
            x = x[:, :, padding_length:padding_end, padding_length:padding_end]

            for i in range(8):
                count[i] = (torch.sum(x == centers[i]).float()/x.numel())
        x_q = x.clone()
        if flag:
            # x_shape = x.shape
            value, _ = torch.max(count,dim=0)
            padding_value = value.tolist()
            # padding_value = centers.tolist()[0]
            pll = x_l-padding_end
            # pad = nn.ConstantPad2d(padding=(padding_length,pll,padding_length,pll), value = padding_value)
            # x_q = pad(x_q)
            x_q = torch._C._nn.reflection_pad2d(x_q, (padding_length,pll,padding_length,pll))

        
        # =================== Add noise to the bit stream ======================
        output_shape = x.shape
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        if flag:
            codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        else:
            codeWithNoise = x
        x = codeWithNoise#.permute(0, 2, 1).reshape(output_shape) # 256*64*32*32
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================
        
        if flag:
            padding_value = centers.tolist()[0]
            pll = x_l-padding_end
            # pad = nn.ConstantPad2d(padding=(padding_length,pll,padding_length,pll), value = padding_value)
            # x = pad(x)
            x = torch._C._nn.reflection_pad2d(x, (padding_length,pll,padding_length,pll))

        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.fl_4.forward(x)
        x = self.af_4.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)


        return codeSent, codeWithNoise, codeReceived, count, x_q

# 1+1
class ADJSCC5(nn.Module):
    def __init__(self):
        super(ADJSCC5, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(4,64,32,4,0)
        # self.af1 = AF(32)
        self.quantizer = quanti_STE_2.apply

        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(32,64,4,4,0)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr):
        if model_name == 'ResNetonImageNet':
            shape = [-1, 64, 56, 56] # resnet on imagenets
        elif model_name == 'ResNetoncifar100':
            shape = [-1, 64, 8, 8] # resnet on cifar100
        shape = [-1, 64, 56, 56]
        x = inputs.permute(0, 2, 1).reshape(shape)
        x = self.fl1.forward(x)
        temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
        self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
        x = x * (1 / self.norm) * math.sqrt(64*4*4)
        x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')

        count = torch.zeros((8)).cuda()
        for i in range(8):
            count[i] = (torch.sum(x == centers[i]).float()/x.numel())
        x_q = x.clone()
        
        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR) 
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================
        
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)
        return codeSent, codeWithNoise, codeReceived, count, x_q
# 1+3
class ADJSCC6(nn.Module):
    def __init__(self):
        super(ADJSCC6, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(4,64,32,4,0)
        self.quantizer = quanti_STE_2.apply
        self.fl_2 = FL2(3,32,64,2,0)
        self.af_2 = AF(64)
        self.fl_3 = FL2(3,64,128,2,0)
        self.af_3 = AF(128)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(128,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr):
        if model_name == 'ResNetonImageNet':
            shape = [-1, 64, 56, 56] # resnet on imagenets
        elif model_name == 'ResNetoncifar100':
            shape = [-1, 64, 8, 8] # resnet on cifar100
        x = inputs.permute(0, 2, 1).reshape(shape)
        x = self.fl1.forward(x)
        temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
        self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
        x = x * (1 / self.norm) * math.sqrt(64*4*4)
        x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
        count = torch.zeros((8)).cuda()
        for i in range(8):
            count[i] = (torch.sum(x == centers[i]).float()/x.numel())
        x_q = x.clone()

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR) 
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================

        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, count, x_q
# 5+5
class ADJSCC7(nn.Module):
    def __init__(self):
        super(ADJSCC7, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(2,64,96,1,2)
        self.af1 = AF(96)
        self.fl2 = FL1(3,96,128,2,2)
        self.af2 = AF(128)
        self.fl3 = FL1(3,128,96,2,1)
        self.af3 = AF(96)
        self.fl4 = FL1(3,96,64,1,0)
        self.af4 = AF(64)
        self.fl5 = FL1(3,64,32,1,1)
        self.quantizer = quanti_STE_2.apply
        self.fl_1 = FL2(3,32,64,1,1)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,1,0)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,128,2,1)
        self.af_3 = AF(128)
        self.fl_4 = FL2(3,128,96,2,2)
        self.af_4 = AF(96)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(96,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr):
        if model_name == 'ResNetonImageNet':
            shape = [-1, 64, 56, 56] # resnet on imagenets
        elif model_name == 'ResNetoncifar100':
            shape = [-1, 64, 8, 8] # resnet on cifar100
        x = inputs.permute(0, 2, 1).reshape(shape)
        x = self.fl1.forward(x)
        x = self.af1.forward(x,SNR)
        x = self.fl2.forward(x)
        x = self.af2.forward(x,SNR)
        x = self.fl3.forward(x)
        x = self.af3.forward(x,SNR)
        x = self.fl4.forward(x)
        x = self.af4.forward(x,SNR)
        x = self.fl5.forward(x)        
        temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
        self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
        x = x * (1 / self.norm) * math.sqrt(64*4*4)
        x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
        x_q = x.clone()

        count = torch.zeros((8)).cuda()
        for i in range(8):
            count[i] = (torch.sum(x == centers[i]).float()/x.numel())

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR) 
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================
        
        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.fl_4.forward(x)
        x = self.af_4.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)
        return codeSent, codeWithNoise, codeReceived, count, x_q
# a-adjscc
class ADJSCC8(nn.Module):
    def __init__(self):
        super(ADJSCC8, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(4,64,32,4,0)
        # self.af1 = AF(32)
        self.quantizer = quanti_STE_2.apply
        self.fl_1 = FL2(3,32,64,1,1)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,1,0)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,128,2,1)
        self.af_3 = AF(128)
        self.fl_4 = FL2(3,128,96,2,2)
        self.af_4 = AF(96)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(96,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr, flag = 1):
        x = inputs
        count = torch.zeros((8)).cuda()
        if flag:
            if model_name == 'ResNetonImageNet':
                shape = [-1, 64, 56, 56] # resnet on imagenets
            elif model_name == 'ResNetoncifar100':
                shape = [-1, 64, 8, 8] # resnet on cifar100
            x = inputs.permute(0, 2, 1).reshape(shape)
            x = self.fl1.forward(x)
            temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
            self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
            x = x * (1 / self.norm) * math.sqrt(64*4*4)
            for i in range(8):
                count[i] = (torch.sum(x == centers[i]).float()/x.numel())
        
        x_q = x.clone()

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        if flag:
            codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        else:
            codeWithNoise = x
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================

        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.fl_4.forward(x)
        x = self.af_4.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, count, x_q

class ADJSCC_c100(nn.Module):
    def __init__(self):
        super(ADJSCC_c100, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(2,64,32,2,0)
        # self.af1 = AF(32)
        self.quantizer = quanti_STE_2.apply
        self.fl_1 = FL2(3,32,64,1,1)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,1,0)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,128,2,3)
        self.af_3 = AF(128)
        self.fl_4 = FL2(3,128,96,2,2)
        self.af_4 = AF(96)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(96,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, inputs, SNR, centers,model_name, cr, flag = 1):
        x = inputs
        count = torch.zeros((8)).cuda()
        if flag:
            if model_name == 'ResNetonImageNet':
                shape = [-1, 64, 56, 56] # resnet on imagenets
            elif model_name == 'ResNetoncifar100':
                shape = [-1, 64, 8, 8] # resnet on cifar100
            x = inputs.permute(0, 2, 1).reshape(shape)
            x = self.fl1.forward(x)
            temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
            self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
            x = x * (1 / self.norm) * math.sqrt(64*4*4)
            x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
            for i in range(8):
                count[i] = (torch.sum(x == centers[i]).float()/x.numel())

        x_q = x.clone()

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        if flag:
            codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        else:
            codeWithNoise = x
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================
        
        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.fl_4.forward(x)
        x = self.af_4.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)
        return codeSent, codeWithNoise, codeReceived, count, x_q

# more
class ADJSCC_c100_more(nn.Module):
    def __init__(self):
        super(ADJSCC_c100_more, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(1,64,32,1,0)
        # self.af1 = AF(32)
        self.quantizer = quanti_STE_2.apply
        self.fl_1 = FL2(2,32,64,1,0)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,1,0)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,128,1,1)
        self.af_3 = AF(128)
        self.fl_4 = FL2(3,128,96,1,1)
        self.af_4 = AF(96)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(96,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, inputs, SNR, centers, model_name, cr, flag = 1):
        x = inputs
        count = torch.zeros((8)).cuda()
        if flag:
            if model_name == 'ResNetonImageNet':
                shape = [-1, 64, 56, 56] # resnet on imagenets
            elif model_name == 'ResNetoncifar100':
                shape = [-1, 64, 8, 8] # resnet on cifar100
            x = inputs.permute(0, 2, 1).reshape(shape)
            x = self.fl1.forward(x)
            temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
            self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
            x = x * (1 / self.norm) * math.sqrt(64*4*4)
            x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
            for i in range(8):
                count[i] = (torch.sum(x == centers[i]).float()/x.numel())
        x_q = x.clone()

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # simulate binary encoding
        if flag:
            codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        else:
            codeWithNoise = x
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================
        
        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.fl_4.forward(x)
        x = self.af_4.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, count, x_q


# a-adjscc satellite
class A_ADJSCC(nn.Module):
    def __init__(self):
        super(A_ADJSCC, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(2,64,32,2,0)
        self.quantizer = quanti_STE_2.apply
        self.fl_1 = FL2(3,32,64,1,1)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,1,1)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,128,2,3)
        self.af_3 = AF(128)
        self.fl_4 = FL2(3,128,96,2,2)
        self.af_4 = AF(96)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(96,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss().cuda()

        # self.sigma = 1
        # # self.fl1 = FL1(4,24,12,4,0)
        # self.fl1 = FL1(2,24,12,2,0)
        # # self.af1 = AF(32)
        # self.quantizer = quanti_STE_2.apply
        # self.fl_1 = FL2(2,12,24,1,1)
        # self.af_1 = AF(24)
        # self.fl_2 = FL2(2,24,36,1,2)
        # self.af_2 = AF(36)
        # self.fl_3 = FL2(2,36,48,2,2)
        # self.af_3 = AF(48)
        # self.fl_4 = FL2(3,48,36,2,3)
        # self.af_4 = AF(36)
        # self.trans_conv4 = nn.Sequential(
        #     nn.ConvTranspose2d(36,24,2,1,2)
        # )
        # self.igdn4 = GDN(24,device,inverse=True)
        # self.decoder_last = nn.Sigmoid()

        # self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, inputs, SNR, centers, model_name, cr, flag = 1):
        x = inputs
        count = torch.zeros((8)).cuda()
        if flag:
            shape = [-1, 64, 16, 16] # resnet_satellite
            # shape = [-1, 24, 32, 32] # mobilenet_satellite
            x = inputs.permute(0, 2, 1).reshape(shape)
            x = self.fl1.forward(x)
            # x = self.af1.forward(x,SNR)
            # add energy Pnorm module
            temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
            self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
            x = x * (1 / self.norm) * math.sqrt(64*4*4)
            x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')

            # compress ratio
            cr = 0
            x_l = x.shape[2]
            x_size = x.shape[2]*x.shape[3]
            x_length = torch.sqrt(x_size * torch.tensor(1-cr))
            padding_length = int(torch.round(0.5*(x_l - x_length)))
            padding_end = padding_length+int(x_length)
            x = x[:, :, padding_length:padding_end, padding_length:padding_end]

            for i in range(8):
                count[i] = (torch.sum(x == centers[i]).float()/x.numel())

        x_q = x.clone()
        if flag:
            value, _ = torch.max(count,dim=0)
            padding_value = value.tolist()
            pll = x_l-padding_end
            x_q = torch._C._nn.reflection_pad2d(x_q, (padding_length,pll,padding_length,pll))
        
        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
    
        # Simulated binary encoding
        if flag:
            codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        else:
            codeWithNoise = x
        x = codeWithNoise

        # =================== Add noise to the bit stream ======================
        if flag:
            padding_value = centers.tolist()[0]
            pll = x_l-padding_end
            x = torch._C._nn.reflection_pad2d(x, (padding_length,pll,padding_length,pll))


        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.fl_4.forward(x)
        x = self.af_4.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, count, x_q


# Encoder 1 layers + decoder 5 layers
class SemanticNN(nn.Module):
    def __init__(self):
        super(SemanticNN, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(4,64,32,4,0)
        self.quantizer = quanti_STE_2.apply

        self.fl_1 = FL2(3,32,64,1,1)
        self.af_1 = AF(64)
        self.fl_2 = FL2(3,64,96,1,0)
        self.af_2 = AF(96)
        self.fl_3 = FL2(3,96,128,2,1)
        self.af_3 = AF(128)
        self.fl_4 = FL2(3,128,96,2,2)
        self.af_4 = AF(96)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(96,64,2,1,2)
        )
        self.igdn4 = GDN(64,device,inverse=True)
        self.decoder_last = nn.Sigmoid()

    def forward(self, inputs, SNR, centers, model_name, cr, flag = 1):
        x = inputs
        count = torch.zeros((8)).cuda()
        if flag:
            if model_name == 'ResNetonImageNet':
                shape = [-1, 64, 56, 56] # resnet on imagenets
            elif model_name == 'ResNetoncifar100':
                shape = [-1, 64, 8, 8] # resnet on cifar100
            x = inputs.permute(0, 2, 1).reshape(shape)
            x = self.fl1.forward(x)
            #add energy Pnorm module
            temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
            self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
            x = x * (1 / self.norm) * math.sqrt(64*4*4)

            # quantization
            x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
         
            # ========== compress version ==========
            if cr != 1: #compress ratio
                x_l = x.shape[2]
                x_size = x.shape[2]*x.shape[3]
                x_length = torch.sqrt(x_size * torch.tensor(1-cr))
                padding_length = int(torch.round(0.5*(x_l - x_length)))
                padding_end = padding_length+int(x_length)
                x = x[:, :, padding_length:padding_end, padding_length:padding_end]

            for i in range(8):
                count[i] = (torch.sum(x == centers[i]).float()/x.numel())


        x_q = x.clone()
        if flag:
            value, _ = torch.max(count,dim=0)
            padding_value = value.tolist()
            pll = x_l-padding_end
            x_q = torch._C._nn.reflection_pad2d(x_q, (padding_length,pll,padding_length,pll))



        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        
        # Simulated binary encoding
        if flag:
            codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        else:
            codeWithNoise = x
        x = codeWithNoise
        # =================== Add noise to the bit stream ======================
        
        if flag:
            padding_value = centers.tolist()[0]
            pll = x_l-padding_end
            x = torch._C._nn.reflection_pad2d(x, (padding_length,pll,padding_length,pll))
        

        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.fl_4.forward(x)
        x = self.af_4.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, count, x_q

class Semanticnn_mobilenet(nn.Module):
    def __init__(self):
        super(Semanticnn_mobilenet, self).__init__()
        self.sigma = 1
        self.fl1 = FL1(4,24,12,4,0)
        # self.af1 = AF(32)
        self.quantizer = quanti_STE_2.apply

        self.fl_1 = FL2(3,12,24,1,1)
        self.af_1 = AF(24)
        self.fl_2 = FL2(3,24,36,1,0)
        self.af_2 = AF(36)
        self.fl_3 = FL2(3,36,48,2,1)
        self.af_3 = AF(48)
        self.fl_4 = FL2(3,48,36,2,2)
        self.af_4 = AF(36)
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose2d(36,24,2,1,2)
        )
        self.igdn4 = GDN(24,device,inverse=True)
        self.decoder_last = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, inputs, SNR, centers, model_name, cr, flag = 1):
        x = inputs
        count = torch.zeros((8)).cuda()
        if flag:
            if model_name == 'MobilenetonImageNet':
                shape = [-1, 24, 112, 112] # mobilenet on imagenets
            elif model_name == 'Mobilenetoncifar100':
                shape = [-1, 24, 16, 16] # mobilenet on cifar100
            x = inputs.permute(0, 2, 1).reshape(shape)
            x = self.fl1.forward(x)
            temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
            self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
            x = x * (1 / self.norm) * math.sqrt(64*4*4)
            x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')

            # compress ratio
            if cr != 1:
                x_l = x.shape[2]
                x_size = x.shape[2]*x.shape[3]
                x_length = torch.sqrt(x_size * torch.tensor(1-cr))
                padding_length = int(torch.round(0.5*(x_l - x_length)))
                padding_end = padding_length+int(x_length)
                x = x[:, :, padding_length:padding_end, padding_length:padding_end]

            for i in range(8):
                count[i] = (torch.sum(x == centers[i]).float()/x.numel())

        x_q = x.clone()
        if flag:
            value, _ = torch.max(count,dim=0)
            padding_value = value.tolist()
            # padding_value = centers.tolist()[0]
            pll = x_l-padding_end
            # pad = nn.ConstantPad2d(padding=(padding_length,pll,padding_length,pll), value = padding_value)
            # x_q = pad(x_q)
            x_q = torch._C._nn.reflection_pad2d(x_q, (padding_length,pll,padding_length,pll))


        
        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # Simulated binary encoding
        if flag:
            codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        else:
            codeWithNoise = x
        x = codeWithNoise
        # =================== Add noise to the bit stream ======================
        if flag:
            padding_value = centers.tolist()[0]
            pll = x_l-padding_end
            # pad = nn.ConstantPad2d(padding=(padding_length,pll,padding_length,pll), value = padding_value)
            # x = pad(x)
            x = torch._C._nn.reflection_pad2d(x, (padding_length,pll,padding_length,pll))

        x = self.fl_1.forward(x)
        x = self.af_1.forward(x,SNR)
        x = self.fl_2.forward(x)
        x = self.af_2.forward(x,SNR)
        x = self.fl_3.forward(x)
        x = self.af_3.forward(x,SNR)
        x = self.fl_4.forward(x)
        x = self.af_4.forward(x,SNR)
        x = self.trans_conv4(x)
        x = self.igdn4.forward(x)
        x = self.decoder_last(x)
        codeReceived = x.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, count, x_q


class deepcod(nn.Module):
    def __init__(self,in_channels=64, out_channels=32, in_dim=32, out_dim=3, kernel_size=4):
        super(deepcod,self).__init__()
        self.sigma = 1
        self.num_centers = 8
        self.centers_initial_range = [-1,1]
        # self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, 4)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 2, 2)
        self.quantizer = quanti_STE_2.apply
        torch.nn.init.kaiming_normal_(self.conv_layer.weight)

        self.att1 = Self_Attn(in_dim, in_dim // out_dim, 4, 4)
        self.res_transposed1 = Residual_Transposed_Conv(in_dim, in_dim*2)
        self.att2 = Self_Attn(in_dim*2, 8, 8, 8)
        self.res_transposed2 = Residual_Transposed_Conv(in_dim*2, in_dim*4)
        # self.conv_out = nn.Conv2d(in_channels=in_dim*4, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels=in_dim*4, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(in_dim*4)
        self.relu = nn.LeakyReLU()
        self.tan = nn.Tanh()

    def forward(self, inputs, SNR, centers, model_name, cr):
        if model_name == 'ResNetonImageNet':
            shape = [-1, 64, 56, 56]
        elif model_name == 'ResNetoncifar100':
            shape = [-1, 64, 8, 8] # cifar100_resnet
        # shape = [-1, 64, 16, 16] # satellite_resnet
        # shape = [-1, 24, 32, 32] # satellite_mobilenet
        x = inputs.permute(0, 2, 1).reshape(shape)
        x = self.conv_layer(x)

        # quantization encoding , symbols_hard
        x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
        x_q = x.clone()

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # simulate binary encoding
        codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================

        out = self.att1(x)
        out = self.res_transposed1(out)
        out = self.att2(out)
        out = self.relu(self.bn(self.res_transposed2(out)))
        out = self.tan(self.conv_out(out))
        codeReceived = out.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, x, x_q

class deepcod_mobilenet(nn.Module):
    def __init__(self,in_channels=24, out_channels=12, in_dim=12, out_dim=3, kernel_size=4):
        super(deepcod_mobilenet,self).__init__()
        self.sigma = 1
        self.num_centers = 8
        self.centers_initial_range = [-1,1]
        # self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, 4)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 2, 2)
        self.quantizer = quanti_STE_2.apply
        torch.nn.init.kaiming_normal_(self.conv_layer.weight)

        self.att1 = Self_Attn(in_dim, in_dim // out_dim, 4, 4)
        self.res_transposed1 = Residual_Transposed_Conv(in_dim, in_dim*2)
        self.att2 = Self_Attn(in_dim*2, 8, 8, 8)
        self.res_transposed2 = Residual_Transposed_Conv(in_dim*2, in_dim*4)
        # self.conv_out = nn.Conv2d(in_channels=in_dim*4, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels=in_dim*4, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(in_dim*4)
        self.relu = nn.LeakyReLU()
        self.tan = nn.Tanh()

    def forward(self, inputs, SNR, centers, model_name, cr):
        if model_name == 'MobilenetonImageNet':
            shape = [-1, 24, 112, 112]
        elif model_name == 'Mobilenetoncifar100':
            shape = [-1, 24, 16, 16] 
        x = inputs.permute(0, 2, 1).reshape(shape)
        x = self.conv_layer(x)

        # quantization encoding , symbols_hard
        x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
        x_q = x.clone()

        # =================== Add noise to the bit stream ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        # simulate binary encoding
        codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
        x = codeWithNoise
        # # huffman encoding and decoding
        # =================== Add noise to the bit stream ======================

        out = self.att1(x)
        out = self.res_transposed1(out)
        out = self.att2(out)
        out = self.relu(self.bn(self.res_transposed2(out)))
        out = self.tan(self.conv_out(out))
        codeReceived = out.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived, x, x_q




class SemanticCommunicationSystem(nn.Module):  # pure DeepSC
    def __init__(self):
        super(SemanticCommunicationSystem, self).__init__()
        self.embedding = embedding(14432, 128)  # which means the corpus has 35632 kinds of words and
        # each word will be coded with a 128 dimensions vector
        self.frontEncoder = nn.TransformerEncoderLayer(d_model=128, nhead=8)  # according to the paper
        self.encoder = nn.TransformerEncoder(self.frontEncoder, num_layers=3)
        self.denseEncoder1 = dense(128, 256)
        self.denseEncoder2 = dense(256, 16)

        self.denseDecoder1 = dense(16, 256)
        self.denseDecoder2 = dense(256, 128)
        self.frontDecoder = nn.TransformerDecoderLayer(d_model=128, nhead=8)
        self.decoder = nn.TransformerDecoder(self.frontDecoder, num_layers=3)

        self.prediction = nn.Linear(128, 14432)
        self.softmax = nn.Softmax(dim=2)  # dim=2 means that it calculates softmax in the feature dimension

    def forward(self, inputs): # , h_I, h_Q
        embeddingVector = self.embedding(inputs)
        code = self.encoder(embeddingVector)
        codeSent = self.denseEncoder1(code)
        codeSent = self.denseEncoder2(codeSent)
        codeWithNoise = AWGN_channel(codeSent, 12)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db
        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        codeReceived = self.decoder(codeReceived, code)
        infoPredicted = self.prediction(codeReceived)
        infoPredicted = self.softmax(infoPredicted)
        return infoPredicted, codeSent, codeWithNoise

class ViSemanticCommunication(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViSemanticCommunication, self).__init__()
        self.frontEncoder = nn.TransformerEncoderLayer(d_model=64, nhead=8)  # according to the paper
        self.encoder = nn.TransformerEncoder(self.frontEncoder, num_layers=3)
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了8倍的传输量
        self.denseEncoder2 = dense(128, 8)
        # self.denseEncoder3 = dense(32, 1)

        # self.denseDecoder1 = dense(1, 32)
        self.denseDecoder1 = dense(8, 128)
        self.denseDecoder2 = dense(128, 64)
        self.frontDecoder = nn.TransformerDecoderLayer(d_model=64, nhead=8)
        self.decoder = nn.TransformerDecoder(self.frontDecoder, num_layers=3)

    def forward(self, inputs): # , h_I, h_Q
        code = self.encoder(inputs)
        codeSent = self.denseEncoder1(code)
        codeSent = self.denseEncoder2(codeSent)
        # codeSent = self.denseEncoder3(codeSent)

        # codeWithNoise = codeSent
        # codeSent = torch.as_tensor(codeSent, dtype=torch.float16)
        codeWithNoise = AWGN_channel(codeSent, 3)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db
        # codeWithNoise = torch.as_tensor(codeWithNoise, dtype=torch.float32)

        codeReceived = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(codeReceived)
        # codeReceived = self.denseDecoder3(codeReceived)
        # codeReceived = self.decoder(codeReceived, code)
        codeReceived = self.decoder(codeReceived, codeReceived)

        return codeSent, codeWithNoise, codeReceived

class ViChannelEncoder_2(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_2, self).__init__()
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了21倍的传输量
        self.denseEncoder2 = dense(128, 8)

        self.denseDecoder1 = dense(8, 128)
        self.denseDecoder2 = dense(128, 64)

    def forward(self, inputs): # , h_I, h_Q
        code = self.denseEncoder1(inputs)
        codeSent = self.denseEncoder2(code)

        codeWithNoise = codeSent
        # codeWithNoise = AWGN_channel(codeSent, 1)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder1(codeWithNoise)
        codeReceived = self.denseDecoder2(code)

        return codeSent, codeWithNoise, codeReceived

class ViChannelEncoder_3(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_3, self).__init__()
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了8倍的传输量
        self.denseEncoder2 = dense(128, 8)
        self.denseEncoder3 = dense(8, 3)

        self.denseDecoder1 = dense(3, 8)
        self.denseDecoder2 = dense(8, 128)
        self.denseDecoder3 = dense(128, 64)

    def forward(self, inputs): # , h_I, h_Q
        code = self.denseEncoder1(inputs)
        code = self.denseEncoder2(code)
        codeSent = self.denseEncoder3(code)

        # codeWithNoise = codeSent
        codeWithNoise = AWGN_channel(codeSent, 12)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder1(codeWithNoise)
        code = self.denseDecoder2(code)
        codeReceived = self.denseDecoder3(code)

        return codeSent, codeWithNoise, codeReceived

class ViChannelEncoder_5(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_5, self).__init__()
        self.denseEncoder1 = dense(64, 128)  
        self.denseEncoder2 = dense(128, 256)
        self.denseEncoder3 = dense(256, 128)
        self.denseEncoder4 = dense(128, 32)
        self.denseEncoder5 = dense(32, 8)

        self.denseDecoder1 = dense(8, 32)
        self.denseDecoder2 = dense(32, 128)
        self.denseDecoder3 = dense(128, 256)
        self.denseDecoder4 = dense(256, 128)
        self.denseDecoder5 = dense(128, 64)

    def forward(self, inputs): # , h_I, h_Q
        code = self.denseEncoder1(inputs)
        code = self.denseEncoder2(code)
        code = self.denseEncoder3(code)
        code = self.denseEncoder4(code)
        codeSent = self.denseEncoder5(code)

        codeWithNoise = codeSent
        # codeWithNoise = AWGN_channel(codeSent, 1)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder1(codeWithNoise)
        code = self.denseDecoder2(code)
        code = self.denseDecoder3(code)
        code = self.denseDecoder4(code)
        codeReceived = self.denseDecoder5(code)

        return codeSent, codeWithNoise, codeReceived

class ViChannelEncoder_7(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_7, self).__init__()
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了8倍的传输量
        self.denseEncoder2 = dense(128, 256)
        self.denseEncoder3 = dense(256, 128)
        self.denseEncoder4 = dense(128, 64)
        self.denseEncoder5 = dense(64, 32)
        self.denseEncoder6 = dense(32, 8)
        self.denseEncoder7 = dense(8, 3)


        self.denseDecoder1 = dense(3, 8)
        self.denseDecoder2 = dense(8, 32)
        self.denseDecoder3 = dense(32, 64)
        self.denseDecoder4 = dense(64, 128)
        self.denseDecoder5 = dense(128, 256)
        self.denseDecoder6 = dense(256, 128)
        self.denseDecoder7 = dense(128, 64)


    def forward(self, inputs): # , h_I, h_Q
        code = self.denseEncoder1(inputs)
        code = self.denseEncoder2(code)
        code = self.denseEncoder3(code)
        code = self.denseEncoder4(code)
        code = self.denseEncoder5(code)
        code = self.denseEncoder6(code)
        codeSent = self.denseEncoder7(code)

        codeWithNoise = codeSent
        # codeWithNoise = AWGN_channel(codeSent, 1)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder1(codeWithNoise)
        code = self.denseDecoder2(code)
        code = self.denseDecoder3(code)
        code = self.denseDecoder4(code)
        code = self.denseDecoder5(code)
        code = self.denseDecoder6(code)
        codeReceived = self.denseDecoder7(code)

        return codeSent, codeWithNoise, codeReceived

class ViChannelEncoder_8(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_8, self).__init__()
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了8倍的传输量
        self.denseEncoder2 = dense(128, 256)
        self.denseEncoder3 = dense(256, 512)
        self.denseEncoder4 = dense(512, 256)
        self.denseEncoder5 = dense(256, 128)
        self.denseEncoder6 = dense(128, 64)
        self.denseEncoder7 = dense(64, 32)
        self.denseEncoder8 = dense(32, 8)


        self.denseDecoder2 = dense(8, 32)
        self.denseDecoder3 = dense(32, 64)
        self.denseDecoder4 = dense(64, 128)
        self.denseDecoder5 = dense(128, 256)
        self.denseDecoder6 = dense(256, 512)
        self.denseDecoder7 = dense(512, 256)
        self.denseDecoder8 = dense(256, 128)
        self.denseDecoder9 = dense(128, 64)


    def forward(self, inputs): # , h_I, h_Q
        code = self.denseEncoder1(inputs)
        code = self.denseEncoder2(code)
        code = self.denseEncoder3(code)
        code = self.denseEncoder4(code)
        code = self.denseEncoder5(code)
        code = self.denseEncoder6(code)
        code = self.denseEncoder7(code)
        codeSent = self.denseEncoder8(code)

        codeWithNoise = codeSent
        # codeWithNoise = AWGN_channel(codeSent, 1)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder2(codeWithNoise)
        code = self.denseDecoder3(code)
        code = self.denseDecoder4(code)
        code = self.denseDecoder5(code)
        code = self.denseDecoder6(code)
        code = self.denseDecoder7(code)
        code = self.denseDecoder8(code)
        codeReceived = self.denseDecoder9(code)

        return codeSent, codeWithNoise, codeReceived


class ViChannelEncoder_9(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_9, self).__init__()
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了8倍的传输量
        self.denseEncoder2 = dense(128, 256)
        self.denseEncoder3 = dense(256, 512)
        self.denseEncoder4 = dense(512, 256)
        self.denseEncoder5 = dense(256, 128)
        self.denseEncoder6 = dense(128, 64)
        self.denseEncoder7 = dense(64, 32)
        self.denseEncoder8 = dense(32, 8)
        self.denseEncoder9 = dense(8, 3)


        self.denseDecoder1 = dense(3, 8)
        self.denseDecoder2 = dense(8, 32)
        self.denseDecoder3 = dense(32, 64)
        self.denseDecoder4 = dense(64, 128)
        self.denseDecoder5 = dense(128, 256)
        self.denseDecoder6 = dense(256, 512)
        self.denseDecoder7 = dense(512, 256)
        self.denseDecoder8 = dense(256, 128)
        self.denseDecoder9 = dense(128, 64)


    def forward(self, inputs): # , h_I, h_Q
        code = self.denseEncoder1(inputs)
        code = self.denseEncoder2(code)
        code = self.denseEncoder3(code)
        code = self.denseEncoder4(code)
        code = self.denseEncoder5(code)
        code = self.denseEncoder6(code)
        code = self.denseEncoder7(code)
        code = self.denseEncoder8(code)
        codeSent = self.denseEncoder9(code)

        codeWithNoise = codeSent
        # codeWithNoise = AWGN_channel(codeSent, 1)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder1(codeWithNoise)
        code = self.denseDecoder2(code)
        code = self.denseDecoder3(code)
        code = self.denseDecoder4(code)
        code = self.denseDecoder5(code)
        code = self.denseDecoder6(code)
        code = self.denseDecoder7(code)
        code = self.denseDecoder8(code)
        codeReceived = self.denseDecoder9(code)

        return codeSent, codeWithNoise, codeReceived

class ViChannelEncoder_conv5(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_conv5, self).__init__()
        self.conv = depconv(64,64)
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了8倍的传输量
        self.denseEncoder2 = dense(128, 256)
        self.denseEncoder3 = dense(256, 128)
        self.denseEncoder4 = dense(128, 32)
        self.denseEncoder5 = dense(32, 8)


        self.denseDecoder1 = dense(8, 32)
        self.denseDecoder2 = dense(32, 128)
        self.denseDecoder3 = dense(128, 256)
        self.denseDecoder4 = dense(256, 128)
        self.denseDecoder5 = dense(128, 64)


    def forward(self, inputs, snr = 12, flag = 0): # , h_I, h_Q
        shape = [-1, 64, 56, 56]
        if flag == 0:
            inputs = inputs.permute(0, 2, 1).reshape(shape)
            code = self.conv(inputs)
            code = code.flatten(2).permute(0, 2, 1)

            code = self.denseEncoder1(code)
            code = self.denseEncoder2(code)
            code = self.denseEncoder3(code)
            code = self.denseEncoder4(code)
            codeSent = self.denseEncoder5(code)
        else:
            codeSent = inputs

        # codeWithNoise = codeSent
        codeWithNoise = AWGN_channel(codeSent, snr)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder1(codeWithNoise)
        code = self.denseDecoder2(code)
        code = self.denseDecoder3(code)
        code = self.denseDecoder4(code)
        codeReceived = self.denseDecoder5(code)

        return codeSent, codeWithNoise, codeReceived


class ViChannelEncoder_conv_5(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_conv_5, self).__init__()
        self.conv1 = depconv(64,64)
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了8倍的传输量
        self.denseEncoder2 = dense(128, 256)
        self.denseEncoder3 = dense(256, 128)
        self.denseEncoder4 = dense(128, 32)
        self.denseEncoder5 = dense(32, 8)


        self.denseDecoder1 = dense(8, 32)
        self.denseDecoder2 = dense(32, 128)
        self.denseDecoder3 = dense(128, 256)
        self.denseDecoder4 = dense(256, 128)
        self.denseDecoder5 = dense(128, 64)
        self.conv2 = depconv_trans(64,64)


    def forward(self, inputs, flag = 0): # , h_I, h_Q
        shape = [-1, 64, 56, 56]
        if flag == 0:
            inputs = inputs.permute(0, 2, 1).reshape(shape)
            code = self.conv1(inputs)
            code = code.flatten(2).permute(0, 2, 1)

            code = self.denseEncoder1(code)
            code = self.denseEncoder2(code)
            code = self.denseEncoder3(code)
            code = self.denseEncoder4(code)
            codeSent = self.denseEncoder5(code)
        else:
            codeSent = inputs



        # codeWithNoise = codeSent
        codeWithNoise = AWGN_channel(codeSent, 3)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder1(codeWithNoise)
        code = self.denseDecoder2(code)
        code = self.denseDecoder3(code)
        code = self.denseDecoder4(code)
        code = self.denseDecoder5(code)
        code = code.permute(0, 2, 1).reshape(shape)
        code = self.conv2(code)
        codeReceived = code.flatten(2).permute(0, 2, 1)

        return codeSent, codeWithNoise, codeReceived


class ViChannelEncoder_11(nn.Module):  # visual DeepSC
    def __init__(self):
        super(ViChannelEncoder_11, self).__init__()
        self.denseEncoder1 = dense(64, 128)   # 从64压到了8，降低了8倍的传输量
        self.denseEncoder2 = dense(128, 256)
        self.denseEncoder3 = dense(256, 512)
        self.denseEncoder4 = dense(512, 1024)
        self.denseEncoder5 = dense(1024, 512)
        self.denseEncoder6 = dense(512, 256)
        self.denseEncoder7 = dense(256, 128)
        self.denseEncoder8 = dense(128, 64)
        self.denseEncoder9 = dense(64, 32)
        self.denseEncoder10 = dense(32, 8)
        self.denseEncoder11 = dense(8, 3)


        self.denseDecoder1 = dense(3, 8)
        self.denseDecoder2 = dense(8, 32)
        self.denseDecoder3 = dense(32, 64)
        self.denseDecoder4 = dense(64, 128)
        self.denseDecoder5 = dense(128, 256)
        self.denseDecoder6 = dense(256, 512)
        self.denseDecoder7 = dense(512, 1024)
        self.denseDecoder8 = dense(1024, 512)
        self.denseDecoder9 = dense(512, 256)
        self.denseDecoder10 = dense(256, 128)
        self.denseDecoder11 = dense(128, 64)


    def forward(self, inputs): # , h_I, h_Q
        code = self.denseEncoder1(inputs)
        code = self.denseEncoder2(code)
        code = self.denseEncoder3(code)
        code = self.denseEncoder4(code)
        code = self.denseEncoder5(code)
        code = self.denseEncoder6(code)
        code = self.denseEncoder7(code)
        code = self.denseEncoder8(code)
        code = self.denseEncoder9(code)
        code = self.denseEncoder10(code)
        codeSent = self.denseEncoder11(code)

        codeWithNoise = codeSent
        # codeWithNoise = AWGN_channel(codeSent, 1)  # assuming snr = 12db
        # codeWithNoise = fading_channel(codeSent, h_I, h_Q, 12)  # assuming snr = 12db

        code = self.denseDecoder1(codeWithNoise)
        code = self.denseDecoder2(code)
        code = self.denseDecoder3(code)
        code = self.denseDecoder4(code)
        code = self.denseDecoder5(code)
        code = self.denseDecoder6(code)
        code = self.denseDecoder7(code)
        code = self.denseDecoder8(code)
        code = self.denseDecoder9(code)
        code = self.denseDecoder10(code)
        codeReceived = self.denseDecoder11(code)

        return codeSent, codeWithNoise, codeReceived


class MutualInfoSystem(nn.Module):  # mutual information used to maximize channel capacity
    def __init__(self):
        super(MutualInfoSystem, self).__init__()
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, inputs):
        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        return output

def sample_batch(batch_size, sample_mode, x, y):  # used to sample data for mutual info system
    length = x.shape[0]
    if sample_mode == 'joint':
        index = np.random.choice(range(length), size=batch_size, replace=False)
        batch_x = x[index, :]
        batch_y = y[index, :]
    elif sample_mode == 'marginal':
        joint_index = np.random.choice(range(length), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(length), size=batch_size, replace=False)
        batch_x = x[joint_index, :]
        batch_y = y[marginal_index, :]
    batch = torch.cat((batch_x, batch_y), 1)

    return batch

class LossFn(nn.Module):  # Loss function
    def __init__(self):
        super(LossFn, self).__init__()

    def forward(self, output, label, length_sen, num_sample, batch_size):  # num_sample means the num of sentence
        # considering that num_sample may not the integer multiple of batch_size
        delta = 1e-7  # used to avoid vanishing gradient
        result = 0
        for i in range(num_sample):  # for every sentence in batch
            length = length_sen[i]  # get every length of sentence, attention that it's the length of sen without padding
            output_term = output[i, 0:length, :]  # get the sentence of corresponding vector
            label_term = label[i, 0:length, :]
            result -= torch.sum(label_term * torch.log(output_term + delta)) / length
        return result/batch_size

def calBLEU(n_gram, s_predicted, s, length):
    num_gram = length - n_gram + 1  # when n_gram = 1, num_gram = length, in which case the BLEU will calculate by one word
    # and the same, when n_gram = 2, num_gram = length - 1, in which case the BLEU will calculate by two words
    # so it's used to padding zero matrix
    s_predicted_gram = np.zeros((num_gram, n_gram))
    s_gram = np.zeros((num_gram, n_gram))  # used to create a matrix which stores word group to calculate matrix
    gram = np.zeros((2*num_gram, n_gram))
    count = 0
    for i in range(num_gram):
        s_predicted_gram[i, :] = s_predicted[i:i+n_gram]  # get data decoded by system
        s_gram[i, :] = s[i:i+n_gram]  # get origin data
        if s_predicted[i:i+n_gram] not in gram:
            gram[count, :] = s_predicted[i:i+n_gram]
            count += 1
        if s_gram[i:i+n_gram] not in gram:
            gram[count, :] = s[i:i+n_gram]
            count += 1

    gram2 = gram[0:count, :]

    min_zi = 0
    min_mu = 0
    for i in range(0, count):
        gram = gram2[i, :]
        s_predicted_count = 0
        s_count = 0
        for j in range(num_gram):
            if((gram == s_predicted_gram[j, :]).all()):
                s_predicted_count += 1
            if ((gram == s_gram[j, :]).all()):
                s_count += 1
        min_zi += min(s_predicted_count, s_count)
        min_mu += s_predicted_count
    return min_zi/min_mu
