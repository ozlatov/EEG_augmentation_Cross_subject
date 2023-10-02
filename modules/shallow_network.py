from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import torch
from torch.nn import init
from torch import nn

class Ensure4d(torch.nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        return x
    
class Permute4d(torch.nn.Module):
    def forward(self, x):
        x = x.permute(0,3,2,1)
        return x

class Square(torch.nn.Module):
    def forward(self, x):
        return x*x

class SafeLog(torch.nn.Module):
    def forward (self, x, eps=1e-6):
        """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
        return torch.log(torch.clamp(x, min=eps))

class SqueezeFinalOutput(torch.nn.Module):
    def forward (self, x):
        """
        Removes empty dimension at end and potentially removes empty time
         dimension. It does  not just use squeeze as we never want to remove
         first dimension.
        Returns
        -------
        x: torch.Tensor
            squeezed tensor
        """
    
        assert x.size()[3] == 1
        x = x[:, :, :, 0]
        if x.size()[2] == 1:
            x = x[:, :, 0]
        return x

class ShallowNetwork(nn.Module):
    def __init__(
        self,
        in_chans = 19,
        n_classes = 2,
        n_filters_time=40,
        filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=75,
        pool_time_stride=15,
        final_conv_length=17, #WAS 30, but my trials are shorter
        pool_mode="mean",
        batch_norm=True,
        batch_norm_alpha=0.1,
        drop_prob=0.5,
        split_first_layer=True, #Here assumed always True
    ):
        super(ShallowNetwork, self).__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.split_first_layer = split_first_layer
        
        
        
        self.add_module("ensuredims", Ensure4d())                   # add needed dimension
        self.add_module("permute", Permute4d())                     # 32(batch_size) x 1 x 351(time_steps) x 19(channels)
        self.add_module("conv_time", nn.Conv2d(
                                        1,
                                        self.n_filters_time,
                                        (self.filter_time_length, 1),
                                        stride=1))                  # 32 x 40(time_filters) x 327 x 19
        self.add_module("conv_spat", nn.Conv2d(
                                self.n_filters_time,
                                self.n_filters_spat,
                                (1, self.in_chans),
                                stride=1,
                                bias=not self.batch_norm))          # 32 x 40 x 327 x 1
        n_filters_conv = self.n_filters_spat
        self.add_module("bnorm",
                        nn.BatchNorm2d(n_filters_conv, momentum = self.batch_norm_alpha, affine = True))
        self.add_module("square", Square())                         # 32 x 40 x 327 x 1
        self.add_module("pool", nn.AvgPool2d(
                            kernel_size=(self.pool_time_length, 1),
                            stride=(self.pool_time_stride, 1)))     # 32 x 40 x 17 x 1
        self.add_module("log", SafeLog())
        self.add_module("drop", nn.Dropout(p=self.drop_prob))       # 32 x 40 x 17 x 1
        self.eval() #TRY WITHOUT
        self.add_module("conv_classifier", nn.Conv2d(
                            n_filters_conv,
                            self.n_classes,
                            (self.final_conv_length, 1),
                            bias=True))                             # 32 x 2 x 1 x 1
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", SqueezeFinalOutput())            # 32 x 2
        
        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)
        
        
    def forward(self, x):
        a = self.ensuredims(x)
        b = self.permute(a)
        c = self.conv_time(b)
        d = self.conv_spat(c)
        e = self.bnorm(d)
        f = self.square(e)
        g = self.pool(f)
        h = self.log(g)
        k = self.drop(h)
        l = self.conv_classifier(k)
        m = self.softmax(l)
        n = self.squeeze(m)
        
        return n
