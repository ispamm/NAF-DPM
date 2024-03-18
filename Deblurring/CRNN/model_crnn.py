import torch
import torch.nn as nn
import torch.nn.functional as fn

class CRNN(nn.Module):

    def __init__(self, vocab_size, multi_gpu = True):
        super(CRNN, self).__init__()
        self.lstm = nn.LSTM(512, 256, 2, bidirectional=True)
        self.linear = nn.Linear(512, vocab_size)
        if multi_gpu:
            self.convo = nn.DataParallel(Convolutional())
        else:
            self.convo = Convolutional()

    def forward(self, x):
        x = self.convo(x)
        x = self.map_to_sequence(x)
        x, _  = self.lstm(x)
        x = fn.log_softmax(self.linear(x), 2)
        return x

    def map_to_sequence(self, map):
        batch, channel, height, width = map.size()
        sequence = map.permute(3, 0, 1, 2) #(batch, channel, height, width) -> (width , batch, channel, height)
        sequence.contiguous()
        sequence = sequence.view(width,batch,-1) #(width , batch, channel, height) -> (seq_len, batch, input_size)
        return sequence
    
    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

class Convolutional (nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)

    def forward(self, x):
        x = fn.max_pool2d(fn.relu(self.conv1(x)), (2,2))
        x = fn.max_pool2d(fn.relu(self.conv2(x)), (2,2))
        x = fn.relu(self.conv3(x))
        x = fn.max_pool2d(fn.relu(self.conv4(x)), (2,1))
        x = fn.relu(self.batchnorm1(self.conv5(x)))
        x = fn.relu(self.batchnorm2(self.conv6(x)))
        x = fn.max_pool2d(x, (2,1))
        x = self.conv7(x)
        return x
    
    
'''
@InProceedings{unknown-Box,
author="Randika, Ayantha
and Ray, Nilanjan
and Xiao, Xiao
and Latimer, Allegra",
editor="Llad{\'o}s, Josep
and Lopresti, Daniel
and Uchida, Seiichi",
title="Unknown-Box Approximation to Improve Optical Character Recognition Performance",
booktitle="Document Analysis and Recognition -- ICDAR 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="481--496",
isbn="978-3-030-86549-8"
}
'''