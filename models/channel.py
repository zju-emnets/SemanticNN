import math
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fading_channel(x, h_I, h_Q, snr):
    [batch_size, length, feature_length] = x.shape
    x = torch.reshape(x, (batch_size, -1, 2))
    x_com = torch.complex(x[:, :, 0], x[:, :, 1])
    x_fft = torch.fft.fft(x_com)
    h = torch.complex(torch.tensor(h_I), torch.tensor(h_Q))
    h_fft = torch.fft.fft(h, feature_length * length//2).to(device)
    y_fft = h_fft * x_fft
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(y_fft ** 2) / (length * feature_length * batch_size // 2)
    npower = xpower / snr
    n = torch.randn(batch_size, feature_length * length // 2, device=device) * npower
    y_add = y_fft + n
    y_add = y_add / h_fft
    y = torch.fft.ifft(y_add)
    y_tensor = torch.zeros((y.shape[0], y.shape[1], 2), device=device)
    y_tensor[:, :, 0] = y.real
    y_tensor[:, :, 1] = y.imag
    y_tensor = torch.reshape(y_tensor, (batch_size, length, feature_length))

    return y_tensor


def AWGN_channel(x, snr):  # used to simulate additive white gaussian noise channel
    [batch_size, length, len_feature] = x.shape
    x_power = torch.sum(torch.abs(x),dtype=torch.float32)/ (batch_size * length * len_feature)
    n_power = x_power / (10 ** (snr / 10.0))
    noise = torch.rand(batch_size, length, len_feature, device=device) * n_power
    x_noise = x + noise
    x_noise = torch.as_tensor(x_noise, dtype=torch.float32)
    return x_noise

def AWGN_channel_bitstream(x, snr):  # used to simulate additive white gaussian noise channel

    x_list = []
    length = len(x)
    for i in range(len(x)):
        if x[i] == '0':
            x_list.append(0)
        else:
            x_list.append(1)
    
    x_tensor = torch.tensor(x_list).cuda()

    ber = 0.5 * math.erfc(math.sqrt(10 ** (snr*0.1)))
    n = int(ber * length)
    # print(n)
    indices = torch.tensor(np.random.choice(length, n, replace=False))

    # x_power = torch.sum(torch.abs(x),dtype=torch.float32)/ (batch_size * length * len_feature)
    # x_power = torch.sum(torch.abs(x_tensor)) / length
    # n_power = x_power / (10 ** (snr / 10.0))
    # noise = torch.tensor(np.random.randn(length)).cuda() * torch.sqrt(n_power)
    # x_noise = x_tensor + noise
    # x_noise = torch.as_tensor(x_noise, dtype=torch.float32)
    x_str = ''
    for i in range(length):
        if i in indices:
            x_tensor[i] = torch.abs(x_tensor[i]-1)
        if x_tensor[i] == 0:
            x_str += '0'
        else:
            x_str += '1'

    return x_str


def AWGN_channel_signal(x, snr):  # used to simulate additive white gaussian noise channel
    [batch_size, length, len_feature] = x.shape
    fc = 4000
    fs = 20 * fc 
    ts = np.arange(0, (100 * size) / fs, 1 / fs) #size是码元数
    
    coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))
    
    
    
    bpsk = np.cos(np.dot(2 * pi * fc, ts) + pi * (m - 1) + pi / 4)


    x_power = torch.sum(torch.abs(x),dtype=torch.float32)/ (batch_size * length * len_feature)
    n_power = x_power / (10 ** (snr / 10.0))
    noise = torch.rand(batch_size, length, len_feature, device=device) * n_power
    x_noise = x + noise
    x_noise = torch.as_tensor(x_noise, dtype=torch.float32)
    return x_noise
