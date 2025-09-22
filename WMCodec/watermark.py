import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import random
import numpy as np
from resnet import ResNet221
from torch.nn import Linear, Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
try:
    from utils import init_weights, get_padding
except:
    from .utils import init_weights, get_padding
import pdb

LRELU_SLOPE = 0.1


def Random_watermark(batch_size): # 4-digit base-16
    sign = torch.randint(low=0, high=16, size=(batch_size, 4))
    return sign

class Watermark_Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Watermark_Encoder, self).__init__()
        self.h = h
        self.embeding = nn.Embedding(16, 16) 
        self.linear_layer1 = weight_norm(Linear(in_features=64, out_features=128))
        self.linear_layer2 = weight_norm(Linear(in_features=128, out_features=512)) 

    def forward(self, x): 
        x_e = self.embeding(x) # [32, 4, 16]
        x = x_e.reshape(x_e.shape[0], x_e.shape[1] * x_e.shape[2]) # [32, 64]
        x = self.linear_layer1(x)
        # x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.linear_layer2(x)
    
        return x

class Watermark_Decoder(torch.nn.Module):
    def __init__(self, h):
        super(Watermark_Decoder, self).__init__()
        self.h = h
        self.recover = ResNet221(feat_dim=80, embed_dim=512, pooling_func='MQMHASTP') 
        self.linear_layer1_1 = weight_norm(Linear(in_features=512, out_features=128))
        self.linear_layer1_2 = weight_norm(Linear(in_features=128, out_features=16))
        self.linear_layer2_1 = weight_norm(Linear(in_features=512, out_features=128))
        self.linear_layer2_2 = weight_norm(Linear(in_features=128, out_features=16)) 
        self.linear_layer3_1 = weight_norm(Linear(in_features=512, out_features=128))
        self.linear_layer3_2 = weight_norm(Linear(in_features=128, out_features=16))
        self.linear_layer4_1 = weight_norm(Linear(in_features=512, out_features=128))
        self.linear_layer4_2 = weight_norm(Linear(in_features=128, out_features=16))
    
    def forward(self, x): 
        x = x.transpose(1, 2) 
        x = self.recover(x)  
        x = x[-1] 
        sign_1_score = self.linear_layer1_2(self.linear_layer1_1(x)) # predict the probability for every digit
        sign_2_score = self.linear_layer2_2(self.linear_layer2_1(x)) 
        sign_3_score = self.linear_layer3_2(self.linear_layer3_1(x))
        sign_4_score = self.linear_layer4_2(self.linear_layer4_1(x))
        sign_score = (sign_1_score, sign_2_score, sign_3_score, sign_4_score) 
        indices1 = torch.argmax(sign_1_score, dim=1, keepdim=True) 
        indices2 = torch.argmax(sign_2_score, dim=1, keepdim=True)
        indices3 = torch.argmax(sign_3_score, dim=1, keepdim=True)
        indices4 = torch.argmax(sign_4_score, dim=1, keepdim=True)
        sign_g_hat = torch.cat([indices1, indices2, indices3, indices4], dim=1) 

        return sign_score, sign_g_hat

def sign_loss(sign_score, sign): # [32, 10], [32, 4] 
    target_tuple = [split.squeeze(dim=1) for split in torch.split(sign, 1, dim=1)] # ([32],[32],[32],[32])
    loss = 0
    for i in range(4): 
        loss += F.cross_entropy(sign_score[i], target_tuple[i])
    loss = loss / 4
    
    return loss


def attack(y_g_hat, order_list = None): 
    # attack is used for whole batch
    # order_list is set of operation [CLP, RSP-16 , Noise-W20,  Noise-P20, APS-05, APS-15 , HPF-18 , LPF-10]
    # order is tuple，[(CLP, 0.4), (RSP-16, 0.3), (Noise-W20, 0.3)]
    '''
    close loop: 完全无影响(CLP)
    re sampling: Uniformly resampled to 16kHz (RSP-16)
    lossy compression: MP3 64 kbps (MP3-64)
    random noise: Noise type is uniformly sampled from White, and Pink Noise 20dB (Noise-W20) (Noise-P20)
    Gain: Gain multiplies a random amplitude to reduce or increase the volume, 0.5 amplitude scaling(APS-05) 1.5 amplitude scaling(APS-15)
    Pass filter: (HPF-18)、(LPF-10)
    '''

    random_number = random.random()
    raw_random_number = random_number
    #print("raw_random_number: ", raw_random_number)
    for order in order_list:
        random_number = random_number - order[1]
        if random_number < 0:
            Opera = order[0]
            break
    #print("raw_random_number: ", raw_random_number, "  Opera:", Opera)

    if Opera == "CLP":
        y_g_hat_att = y_g_hat
        return y_g_hat_att, Opera
    
    if Opera == "RSP-90":
        resample1 = torchaudio.transforms.Resample(24000, 21600).to(y_g_hat.device)
        resample2 = torchaudio.transforms.Resample(21600, 24000).to(y_g_hat.device)
        y_g_hat_att = resample1(y_g_hat)
        y_g_hat_att = resample2(y_g_hat_att)
        return y_g_hat_att, Opera
    
    if Opera == "Noise-W35": 
        def generate_white_noise(X, N, snr):
            noise = torch.randn(N)
            noise = noise.to(X.device)
            snr = 10 ** (snr/10)
            power = torch.mean(torch.square(X))
            npower = power / snr
            noise = noise * torch.sqrt(npower)
            X = X + noise
            return X, noise

        y_g_hat_att, noise = generate_white_noise(y_g_hat, y_g_hat.shape[2], 35)
        return y_g_hat_att, Opera
    
    if Opera == "SS-01": 
        def generate_random_tensor(N, rate):
            num_zeros = int(N * rate)  
            num_ones = N - num_zeros
            tensor_data = np.concatenate((np.zeros(num_zeros), np.ones(num_ones)))
            np.random.shuffle(tensor_data)
            mask = torch.tensor(tensor_data).float()
            return mask
        
        mask = generate_random_tensor(y_g_hat.shape[2], 0.001)
        mask = mask.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * mask
        
        return y_g_hat_att, Opera
    
    if Opera == "AS-90": 
        def generate_rate_tensor(N, rate):
           tensor = torch.full((N,), rate)
           return tensor
        
        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.9)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para


        return y_g_hat_att, Opera
        
    if Opera == "TS-09": 
        speed_factor = 0.95
        resampler = torchaudio.transforms.Resample(
            orig_freq=24000,  
            new_freq=int(24000* speed_factor) 
        )
        
        y_g_hat_att = resampler(y_g_hat)
        


        return y_g_hat_att, Opera
    
    if Opera == "EA-0301": 
        def generate_rate_tensor(N, rate):
           tensor = torch.full((N,), rate)
           return tensor
        
        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.3)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_echo = y_g_hat * rate_para
        shift_amount = int(y_g_hat.size(2) * 0.15)  
        y_g_hat_truncated =y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:,:,:shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo),dim = 2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] +  y_g_hat

        return y_g_hat_att, Opera 
    
    if Opera == "LP5000": 
        y_g_hat_att = torchaudio.functional.lowpass_biquad(y_g_hat, 24000, cutoff_freq = 5000, Q = 0.707)

        return y_g_hat_att, Opera 
    
    if Opera == "MF-3":
        window_size = 3
        filtered_signal = torch.zeros_like(y_g_hat)
        for i in range(y_g_hat.size(2)):
            start = max(0, i - window_size // 2)
            end = min(y_g_hat.size(2), i + window_size // 2 + 1)
            window = y_g_hat[:,:,start:end]
            filtered_signal[:,:,start:end] = torch.median(window)
        
        y_g_hat_att = filtered_signal
        return y_g_hat_att, Opera


def clip(y_g_hat): # "The default slice attack only applies once."
    random_number = random.random()
    if random_number <= 0.5:
        clip_flag = "Y"
        length = y_g_hat.size(2)
        cut_length_1 = torch.randint(length // 4, length // 3, size=())
        cut_end_1 = torch.randint(length // 3, length - 1, size=())
        y_g_hat_clip_one = torch.cat([y_g_hat[:,:,:cut_end_1 - cut_length_1], y_g_hat[:,:,cut_end_1:]], dim=2)
        #cut_length_2 = torch.randint(length // 4, length // 3, size=())
        #cut_end_2 = torch.randint(length // 3, length - 1, size=())
        #y_g_hat_clip = torch.cat([y_g_hat_clip_one[:,:,:cut_end_2 - cut_length_2], y_g_hat_clip_one[:,:,cut_end_2:]], dim=2)
        y_g_hat_clip = y_g_hat_clip_one 
    else:
        clip_flag = "N"
        y_g_hat_clip = y_g_hat

    return y_g_hat_clip, clip_flag

