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
    

class ADJSCC4(nn.Module):
    def __init__(self):
        super(ADJSCC4, self).__init__()
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

    def forward(self, inputs, SNR, centers, flag = 1):
        # print(SNR)
        x = inputs
        count = torch.zeros((8)).cuda()
        if flag:
            shape = [-1, 64, 56, 56] # resnet on imagenets

            # shape = [-1, 64, 8, 8]
            x = inputs.permute(0, 2, 1).reshape(shape)
            x = self.fl1.forward(x)
            # x = self.af1.forward(x,SNR)
            #加入能量Pnorm模块
            temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
            self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
            x = x * (1 / self.norm) * math.sqrt(64*4*4)

            # 量化2 # , symbols_hard
            x = self.quantizer(x, centers.cuda(), self.sigma, 'NCHW')
        

            for i in range(8):
                count[i] = (torch.sum(x == centers[i]).float()/x.numel())
        
            # compress ratio
            cr = 0.2
            x_size = x.shape[2]*x.shape[3]
            x_length = torch.sqrt(x_size * torch.tensor(1-cr))
            padding_length = int(torch.round(0.5*(x.shape[2] - x_length)))
            x = x[:, :, padding_length:, padding_length:]
            # print(x.shape)
        
        x_q = x.clone()
        if flag:
            # x_shape = x.shape
            value, _ = torch.max(count,dim=0)
            padding_value = value.tolist()
            # padding_value = centers.tolist()[0]
            pad = nn.ConstantPad2d(padding=(padding_length,0,padding_length,0), value = padding_value)
            x_q = pad(x_q)



        # =================== 特征值上加噪声 ======================
        # codeSent = x.flatten(2).permute(0, 2, 1)

        # codeWithNoise = AWGN_channel(codeSent, snr=6)
        # x = codeWithNoise.permute(0, 2, 1).reshape(x_shape)
        # =================== 特征值上加噪声 ======================
        

        # =================== 比特流上加噪声 ======================
        codeSent = x.flatten(2).permute(0, 2, 1)
        
        # 二进制编码
        # encoded_str = ascii_encode(symbols_hard)
        # # codeWithNoise = encoded_str
        # codeWithNoise = AWGN_channel_bitstream(encoded_str, snr=SNR)
        # decoded_x = ascii_decode(encoded_str, centers)
        # decoded_x = decoded_x.view(x.shape).cuda()
        # x = decoded_x
        # 模拟二进制编码
        if flag:
            codeWithNoise = ascii_encode_v2(x, centers.cuda(), snr=SNR)
            # codeWithNoise = x
        else:
            codeWithNoise = x
        x = codeWithNoise

        # # # huffman编码
        # huffman_str, codes = huffman_encode(x)
        # print(len(huffman_str)/8,'B')

        # codeWithNoise = AWGN_channel_bitstream(huffman_str, snr=6)

        # # huffman解码
        # temp = symbols_hard.view(-1).unsqueeze(0).tolist()[0]
        # res = []
        # for i in temp:
        #     if i not in res:
        #         res.append(i)
        # centers = centers[res]
        # decoded_x = huffman_decode(codeWithNoise, centers, codes) #huffman_str
        # length = x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]
        # if len(decoded_x) >= length:
        #     decoded_x = decoded_x[:length]
        # else:
        #     appends_decoded = torch.ones(length-len(decoded_x))*centers[0]
        #     decoded_x = torch.cat((decoded_x, appends_decoded))
        # decoded_x = decoded_x.view(x.shape).cuda()
        # print(decoded_x.cuda().equal(x))
        # x = decoded_x
        # =================== 比特流上加噪声 ======================
        
        if flag:
            padding_value = centers.tolist()[0]
            pad = nn.ConstantPad2d(padding=(padding_length,0,padding_length,0), value = padding_value)
            x = pad(x)
            # print(x.shape)
            # print(x)

        


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
        # print(x.shape)
        codeReceived = x.flatten(2).permute(0, 2, 1)


        return codeSent, codeWithNoise, codeReceived, count, x_q
