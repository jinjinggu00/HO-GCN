import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer

from .init_func import bn_init, conv_branch_init, conv_init

EPS = 1e-4

class HOGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        LEAKY_ALPHA = 0.1
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        self.softmax = nn.Softmax(-1)
        self.init_weights()

    def forward(self, x, A=None, alpha=1 , beta=1):
        # Input: N, C, T, V
        N,C,T,V = x.size()
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        diff = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2)) 
        q = self.conv1(x)                   
        k = self.conv2(x)                    
        att = self.relu(torch.einsum('nctu,nctv->ncuv', q, k) / T)
        Q= self.conv4(diff) * alpha + self.conv4(att) * beta + (A[None, None] if A is not None else 0)   
                      
        out = torch.einsum('ncuv,nctu->nctv', Q, x3)        
        return out 

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


class unit_hogcn(nn.Module):
    def __init__(self, in_channels, out_channels, A):

        super(unit_hogcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(HOGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None
        
        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha, self.beta)
            #z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
