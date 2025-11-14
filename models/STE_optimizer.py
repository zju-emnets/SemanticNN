import torch
from .quantization_part import *
from .quantization_part import quantize1d

class quanti_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, centers, sigma, data_format):
        softout, hardout, symbols_hard = quantize1d(input, centers, sigma, data_format)
        ### 这里返回softout是为了让反向传播的时候能利用这个计算梯度
        out = torch.add(hardout,-softout).detach()+softout
        return out, symbols_hard

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output.shape)
        return grad_output.clamp_(-1, 1),None,None,None


class quanti_STE_2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, centers, sigma, data_format):
        softout, hardout, symbols_hard = quantize1d(input, centers, sigma, data_format)
        ### 这里返回softout是为了让反向传播的时候能利用这个计算梯度
        out = torch.add(hardout,-softout).detach()+softout
        return out#, symbols_hard

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output.shape)
        return grad_output.clamp_(-1, 1),None,None,None
