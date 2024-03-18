import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math 
from typing import Optional, Tuple, Union, List
import numpy as np
from Deblurring.model.ConditionalNAFNET import ConditionalNAFNet, ConditionalNAFNetLocal
from Deblurring.model.NAFNET import NAFNet,NAFNetLocal

class NAFDPM(nn.Module):
    def __init__(self, input_channels: int = 2, output_channels: int = 1, n_channels: int = 32,
                 middle_blk_num: int = 1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],mode=1):
        super(NAFDPM, self).__init__()
        #MODE TEST
        if mode == 0: 
            self.denoiser  = ConditionalNAFNetLocal(img_channel= input_channels, 
                                           width=n_channels, 
                                           middle_blk_num=middle_blk_num, 
                                           enc_blk_nums=enc_blk_nums, 
                                           dec_blk_nums=dec_blk_nums, 
                                           upscale=1)
        #MODE TEST
        else:
            self.denoiser  = ConditionalNAFNet(img_channel= input_channels, 
                                           width=n_channels, 
                                           middle_blk_num=middle_blk_num, 
                                           enc_blk_nums=enc_blk_nums, 
                                           dec_blk_nums=dec_blk_nums, 
                                           upscale=1)
            
        #MODE TRAIN
        if mode == 0:
            self.init_predictor = NAFNetLocal(      img_channel=input_channels//2, 
                                           width=n_channels, 
                                           middle_blk_num=middle_blk_num, 
                                           enc_blk_nums=enc_blk_nums, 
                                           dec_blk_nums=dec_blk_nums)
        #MODE TEST
        else: 
            self.init_predictor = NAFNet(      img_channel=input_channels//2, 
                                           width=n_channels, 
                                           middle_blk_num=middle_blk_num, 
                                           enc_blk_nums=enc_blk_nums, 
                                           dec_blk_nums=dec_blk_nums)
            
    def forward(self, x, condition, t, diffusion):
        x_ = self.init_predictor(condition)
        residual = x - x_
        noisy_image, noise_ref = diffusion.noisy_image(t, residual)
        x__ = self.denoiser(inp=noisy_image,cond= x_.clone().detach(), time=t)
        return x_, x__, noisy_image, noise_ref


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


if __name__ == '__main__':
    from src.config import load_config
    import argparse
    from schedule.diffusionSample import GaussianDiffusion
    from schedule.schedule import Schedule
    import torchsummary
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../conf.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    model = NAFDPM(input_channels=config.CHANNEL_X + config.CHANNEL_Y,
            output_channels = config.CHANNEL_Y,
            n_channels      = config.MODEL_CHANNELS,
            middle_blk_num  = config.MIDDLE_BLOCKS, 
            enc_blk_nums    = config.ENC_BLOCKS, 
            dec_blk_nums    = config.DEC_BLOCKS)
    
    schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
    diffusion = GaussianDiffusion(model, config.TIMESTEPS, schedule)
    model.eval()
    print(torchsummary.summary(model.cuda(), [(3, 128, 128)], batch_size=32))

