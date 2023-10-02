from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import torch
from torch.nn import init
from torch import nn
from torch.nn.functional import elu

class Ensure4d(torch.nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        return x
    
class Permute4d(torch.nn.Module):
    def forward(self, x):
        x = x.permute(0,3,2,1)
        return x

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



class DeepNetwork(nn.Module):
    def __init__(
        self,
        in_chans=19,
        n_classes=2,
        final_conv_length = 2, #was auto
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=3,      #WAS 10, had to reduce because of shorter trials
        first_nonlin=elu,
        first_pool_mode="max",
        #first_pool_nonlin=identity,
        later_nonlin=elu,
        later_pool_mode="max",
        #later_pool_nonlin=identity,
        drop_prob=0.5,
        double_time_convs=False,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
    ):
        super(DeepNetwork, self).__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.final_conv_length = final_conv_length
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_nonlin = first_nonlin
        self.first_pool_mode = first_pool_mode
        #self.first_pool_nonlin = first_pool_nonlin
        self.later_nonlin = later_nonlin
        self.later_pool_mode = later_pool_mode
        #self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.double_time_convs = double_time_convs
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        
        conv_stride = 1
        pool_stride = self.pool_time_stride
        
        self.add_module("ensuredims", Ensure4d())                   # add needed dimension
        self.add_module("permute", Permute4d())                     # 32(batch_size) x 1 x 351(time_steps) x 19(channels)
        #CONV BLOCK 1
        self.add_module("conv_time", nn.Conv2d(
                                1,
                                self.n_filters_time,
                                (self.filter_time_length, 1),
                                stride=1))                          # 32 x 25(time_filters) x 342 x 19
        self.add_module("conv_spat", nn.Conv2d(
                                self.n_filters_time,
                                self.n_filters_spat,
                                (1, self.in_chans),
                                stride=1,
                                bias=not self.batch_norm))          # 32 x 25 x 342 x 1
        n_filters_conv = self.n_filters_spat
        self.add_module("bnorm",
                        nn.BatchNorm2d(n_filters_conv, momentum = self.batch_norm_alpha, affine = True, eps=1e-5))
        self.add_module("elu1", nn.ELU())
        self.add_module("pool", nn.MaxPool2d(
                                kernel_size=(self.pool_time_length, 1),
                                stride=(pool_stride, 1)))           # 32 x 25 x 114 x 1
        #CONV BLOCK 2
        self.add_module("drop2", nn.Dropout(p=self.drop_prob))
        self.add_module("conv2", nn.Conv2d(
                                n_filters_conv,
                                self.n_filters_2,
                                (self.filter_length_2, 1),
                                stride=1,
                                bias=not self.batch_norm))          # 32 x 50 x 105 x 1
        self.add_module("bnorm2",
                        nn.BatchNorm2d(self.n_filters_2, momentum = self.batch_norm_alpha, affine = True, eps=1e-5))
        self.add_module("elu2", nn.ELU())
        self.add_module("pool2", nn.MaxPool2d(
                                kernel_size=(self.pool_time_length, 1),
                                stride=(pool_stride, 1)))           # 32 x 50 x 35 x 1
        
        #CONV BLOCK 3
        self.add_module("drop3", nn.Dropout(p=self.drop_prob))
        self.add_module("conv3", nn.Conv2d(
                                self.n_filters_2,
                                self.n_filters_3,
                                (self.filter_length_3, 1),
                                stride=1,
                                bias=not self.batch_norm))          
        self.add_module("bnorm3",
                        nn.BatchNorm2d(self.n_filters_3, momentum = self.batch_norm_alpha, affine = True, eps=1e-5))
        self.add_module("elu3", nn.ELU())
        self.add_module("pool3", nn.MaxPool2d(
                                kernel_size=(self.pool_time_length, 1),
                                stride=(pool_stride, 1)))           # 32 x 100 x 8 x 1
        
        #CONV BLOCK 4
        self.add_module("drop4", nn.Dropout(p=self.drop_prob))
        self.add_module("conv4", nn.Conv2d(
                                self.n_filters_3,
                                self.n_filters_4,
                                (self.filter_length_4, 1),
                                stride=1,
                                bias=not self.batch_norm))          
        self.add_module("bnorm4",
                        nn.BatchNorm2d(self.n_filters_4, momentum = self.batch_norm_alpha, affine = True, eps=1e-5))
        self.add_module("elu4", nn.ELU())
        self.add_module("pool4", nn.MaxPool2d(
                                kernel_size=(self.pool_time_length, 1),
                                stride=(pool_stride, 1)))           # 32 x 200 x 2 x 1
        
        self.eval()
        self.add_module("conv_classifier", nn.Conv2d(
                            self.n_filters_4,
                            self.n_classes,
                            (self.final_conv_length, 1),
                            bias=True))                             # 32 x 2 x 1 x 1
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", SqueezeFinalOutput())            # 32 x 2
        
        
        # Initialization, xavier is same as in our paper...
        # was default from lasagne
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
        param_dict = dict(list(self.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)

        # Start in eval mode
        self.eval()
        


    def forward(self, x):
        a = self.ensuredims(x)
        b = self.permute(a)
        c = self.conv_time(b)
        d = self.conv_spat(c)
        e = self.bnorm(d)
        f = self.elu1(e)
        g = self.pool(f)
        
        k = self.drop2(g)
        l = self.conv2(k)
        m = self.bnorm2(l)
        n = self.elu2(m)
        o = self.pool2(n)
        
        k2 = self.drop3(o)
        l2 = self.conv3(k2)
        m2 = self.bnorm3(l2)
        n2 = self.elu3(m2)
        o2 = self.pool3(n2)
        
        k3 = self.drop4(o2)
        l3 = self.conv4(k3)
        m3 = self.bnorm4(l3)
        n3 = self.elu4(m3)
        o3 = self.pool4(n3)
        
        p = self.conv_classifier(o3)
        r = self.softmax(p)
        s = self.squeeze(r)

        return s