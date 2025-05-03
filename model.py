#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from mmengine.model import BaseModule
from .coordatt import CoordAtt
import math
class ChannelExchange(BaseModule):


    def __init__(self, p=2):
        super(ChannelExchange, self).__init__()
        self.p = p
    def forward(self, x1, x2):
        N, c, h, w = x1.shape   #读取矩阵长度

        exchange_map = torch.arange(c) % self.p == 0   #从0-c选取要交换的图
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))  #取出来并

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class SpatialExchange(BaseModule):


    def __init__(self, p=2):
        super(SpatialExchange, self).__init__()
        self.p = p
    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2





class BiMulCrossAttention(nn.Module):
    def __init__(self, in_d=None, out_d=None):
        super(BiMulCrossAttention, self).__init__()
        if in_d is None:
            in_d = [64, 128, 256, 512, 512]
        self.in_d = in_d
        if out_d is None:
            out_d = [64, 128, 256, 512, 512]
        self.out_d = out_d
        self.EX = ChannelExchange()
        self.SE = SpatialExchange()
        self.att1 = CoordAtt(self.out_d[1],self.out_d[1])
        self.att2 = CoordAtt(self.out_d[2], self.out_d[2])
        self.att3 = CoordAtt(self.out_d[3], self.out_d[3])
        # scale 2
        self.conv_scale2_t1 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.out_d[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[1]),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_t2 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.out_d[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[1]),
            nn.ReLU(inplace=True)
        )
        # scale 3
        self.conv_scale3_t1 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.out_d[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[2]),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_t2 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.out_d[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[2]),
            nn.ReLU(inplace=True)
        )
        # scale 4
        self.conv_scale4_t1 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.out_d[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_t2 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.out_d[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[3]),
            nn.ReLU(inplace=True)
        )
        # scale 5
        self.conv_scale5_t1 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.out_d[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d[4]),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, s2_1, s2_2, s3_1, s3_2, s4_1, s4_2, s5_1, s5_2):
        #print(s2_1.size(), s2_2.size(), s3_1.size(), s3_2.size(), s4_1.size(), s4_2.size(), s5_1.size(), s5_2.size())
        # scale 2
        c2_1 = s2_1
        s2_1 = self.conv_scale2_t1(s2_1)
        #print(s2_1.size(), s3_2.size())

        s3_2 = self.conv_scale2_t2(s3_2)
        s3_2 = F.interpolate(s3_2, scale_factor=(2, 2), mode='bilinear')

        a2_1, a3_2 = self.SE(s2_1, s3_2)
        a2_1 = self.att1(a2_1)

        b2_1, b3_2 = self.EX(s2_1, s3_2)
        b2_1 = self.att1(b2_1)
        s2_1 = a2_1 + b2_1
        s2 = self.relu(s2_1 + c2_1)
        # scale 3
        c3_1 = s3_1
        s3_1 = self.conv_scale3_t1(s3_1)

        s4_2 = self.conv_scale3_t2(s4_2)
        s4_2 = F.interpolate(s4_2, scale_factor=(2, 2), mode='bilinear')

        a3_1, a4_2 = self.SE(s3_1, s4_2)
        a3_1 = self.att2(a3_1)
        b3_1, b4_2 = self.EX(s3_1, s4_2)
        b3_1 = self.att2(b3_1)
        s3_1 = a3_1 + b3_1
        s3 = self.relu(s3_1 + c3_1)
        # scale 4
        c4_1 = s4_1
        s4_1 = self.conv_scale4_t1(s4_1)

        s5_2 = self.conv_scale4_t2(s5_2)
        s5_2 = F.interpolate(s5_2, scale_factor=(2, 2), mode='bilinear')

        a4_1, a5_2 = self.SE(s4_1, s5_2)
        a4_1 = self.att3(a4_1)
        b4_1, b5_2 = self.EX(s4_1, s5_2)
        b4_1 = self.att3(b4_1)
        s4_1 = a4_1 + b4_1
        s4 = self.relu(s4_1 + c4_1)
        # scale 5
        s5_1 = self.conv_scale5_t1(s5_1)


        return s2, s3, s4, s5_1





def createPDCFunc(PDC_type):  # 创建像素差卷积函数
    assert PDC_type in ['shun','ni', 'shun5', 'ni5', 'shun7', 'ni7'], 'unknown PDC type: %s' % str(PDC_type)

    if PDC_type == 'shun':  # CPDC,基于顺时针的像素差卷积
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            # assert padding == dilation, 'padding for ad_conv set wrong'
            # print('0', weights[0][0])
            shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] = buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] + buffer[:, :,
                                                                                              [1, 2, 5, 0, 8, 3, 6, 7]] - 2 * buffer[:, :,
                                                                                                         [4]]
            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]
            buffer[:, :, [4]] = 0
            weights = buffer.view(shape)
            # print(weights[0][0])
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif PDC_type == 'ni':  # CPDC,基于y=-x的像素差卷积

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            # assert padding == dilation, 'padding for ad_conv set wrong'
            # print('0', weights[0][0])
            shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] = buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] + buffer[:, :,
                                                                                              [3, 0, 1, 6, 2, 7, 8, 5]] - 2 * buffer[:, :,
                                                                                                         [4]]
            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]
            buffer[:, :, [4]] = 0
            weights = buffer.view(shape)
            # print(weights[0][0])
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif PDC_type == 'shun5':  # CPDC,基于顺时针的像素差卷积

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'

            assert weights.size(2) == 5 or weights.size(3) == 5, 'kernel size for ad_conv should be 5x5'

            # assert padding == dilation, 'padding for ad_conv set wrong'

            # print('0', weights[0][0])

            shape = weights.shape

            # if weights.is_cuda:

            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)

            # else:

            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)

            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引

            buffer = weights.clone()

            # print(buffer)

            # buffer = weights

            # 1 2 3

            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]

            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 4,

                          5, 6, 7, 8, 9, 10,

                          11, 13, 14, 15,

                          16, 17, 18, 19, 20,

                          21, 22, 23, 24]] = buffer[:, :, [0, 1, 2, 3, 4,

                                                           5, 6, 7, 8, 9, 10,

                                                           11, 13, 14, 15,

                                                           16, 17, 18, 19, 20,

                                                           21, 22, 23, 24]] + buffer[:, :, [5, 0, 1, 2, 3,
                                                                                            10, 7, 8, 13, 4,
                                                                                            15, 6, 18, 9,
                                                                                            20, 11, 16, 17, 14,
                                                                                            21, 22, 23, 24, 19]] - 2 * buffer[:, :,

                                                                                                       [12]]

            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]

            buffer[:, :, [12]] = 0

            weights = buffer.view(shape)

            # print(weights[0][0])

            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            return y

        return func


    elif PDC_type == 'ni5':  # CPDC,基于y=-x的像素差卷积

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'

            assert weights.size(2) == 5 or weights.size(3) == 5, 'kernel size for ad_conv should be 5x5'

            # assert padding == dilation, 'padding for ad_conv set wrong'

            # print('0', weights[0][0])

            shape = weights.shape

            # if weights.is_cuda:

            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)

            # else:

            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)

            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引

            buffer = weights.clone()

            # print(buffer)

            # buffer = weights

            # 1 2 3

            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]

            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 4,

                          5, 6, 7, 8, 9,

                          10, 11, 13, 14,

                          15, 16, 17, 18, 19,

                          20, 21, 22, 23, 24]] = buffer[:, :, [0, 1, 2, 3, 4,

                                                               5, 6, 7, 8, 9, 10,

                                                               11, 13, 14, 15,

                                                               16, 17, 18, 19, 20,

                                                               21, 22, 23, 24]] + buffer[:, :, [1, 2, 3, 4, 9,
                                                                                                0, 11, 6, 7, 14,
                                                                                                5, 16, 8, 19,
                                                                                                10, 18, 17, 13, 24,
                                                                                                15, 20, 21, 22, 23]] - 2 * buffer[:, :,

                                                                                                          [12]]

            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]

            buffer[:, :, [12]] = 0

            weights = buffer.view(shape)

            # print(weights[0][0])

            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            return y

        return func
    elif PDC_type == 'shun7':  # CPDC,基于顺时针的像素差卷积

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'

            assert weights.size(2) == 7 or weights.size(3) == 7, 'kernel size for ad_conv should be 7x7'

            # assert padding == dilation, 'padding for ad_conv set wrong'

            # print('0', weights[0][0])

            shape = weights.shape

            # if weights.is_cuda:

            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)

            # else:

            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)

            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引

            buffer = weights.clone()

            # print(buffer)

            # buffer = weights

            # 1 2 3

            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]

            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 4, 5, 6,
                          7, 8, 9, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20,
                          21, 22, 23, 25, 26, 27,
                          28, 29, 30, 31, 32, 33, 34,
                          35, 36, 37, 38, 39, 40, 41,
                          42, 43, 44, 45, 46, 47, 48]] = buffer[:, :, [0, 1, 2, 3, 4, 5, 6,
                                                                       7, 8, 9, 10, 11, 12, 13,
                                                                       14, 15, 16, 17, 18, 19, 20,
                                                                       21, 22, 23, 25, 26, 27,
                                                                       28, 29, 30, 31, 32, 33, 34,
                                                                       35, 36, 37, 38, 39, 40, 41,
                                                                       42, 43, 44, 45, 46, 47, 48]] + buffer[:, :, [1, 2, 3, 4, 5, 6, 13,
                                                                                                                    0, 15, 8, 9, 10, 11, 20,
                                                                                                                    7, 22, 17, 18, 25, 12, 27,
                                                                                                                    14, 29, 16, 32, 19, 34,
                                                                                                                    21, 36, 23, 30, 21, 26, 41,
                                                                                                                    28, 37, 38, 39, 40, 33, 48,
                                                                                                                    35, 42, 43, 44, 45, 46, 47]] - 2 * buffer[:, :,

                                                                                                       [24]]

            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]

            buffer[:, :, [24]] = 0

            weights = buffer.view(shape)

            # print(weights[0][0])

            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            return y

        return func


    elif PDC_type == 'ni7':  # CPDC,基于y=-x的像素差卷积

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'

            assert weights.size(2) == 7 or weights.size(3) == 7, 'kernel size for ad_conv should be 7x7'

            # assert padding == dilation, 'padding for ad_conv set wrong'

            # print('0', weights[0][0])

            shape = weights.shape

            # if weights.is_cuda:

            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)

            # else:

            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)

            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引

            buffer = weights.clone()

            # print(buffer)

            # buffer = weights

            # 1 2 3

            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]

            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 4, 5, 6,
                          7, 8, 9, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20,
                          21, 22, 23, 25, 26, 27,
                          28, 29, 30, 31, 32, 33, 34,
                          35, 36, 37, 38, 39, 40, 41,
                          42, 43, 44, 45, 46, 47, 48]] = buffer[:, :, [0, 1, 2, 3, 4, 5, 6,
                                                                       7, 8, 9, 10, 11, 12, 13,
                                                                       14, 15, 16, 17, 18, 19, 20,
                                                                       21, 22, 23, 25, 26, 27,
                                                                       28, 29, 30, 31, 32, 33, 34,
                                                                       35, 36, 37, 38, 39, 40, 41,
                                                                       42, 43, 44, 45, 46, 47, 48]] + buffer[:, :,
                                                                                                      [7, 0, 1, 2, 3, 4, 5,
                                                                                                       14, 9, 10, 11, 12, 19, 6,
                                                                                                       21, 8, 23, 16, 17, 26, 13,
                                                                                                       28, 15, 30, 18, 33, 20,
                                                                                                       35, 22, 31, 32, 25, 40, 27,
                                                                                                       42, 29, 36, 37, 38, 39, 34,
                                                                                                       43, 44, 45, 46, 47, 48, 41]] - 2 * buffer[
                                                                                                                  :, :,

                                                                                                                  [24]]

            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]

            buffer[:, :, [24]] = 0

            weights = buffer.view(shape)

            # print(weights[0][0])

            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

            return y

        return func


    else:
        print('unknown PDC type: %s' % str(PDC_type))  # 正常来说走不到这里
        return None




class Conv2d(nn.Module):  # 把之前创建的卷积函数包装成torch卷积api相同的格式
    def __init__(self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        """
        :param pdc_func: 卷积函数
        """
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:  # depth wise卷积要求通道要能被分组数整除
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation  # 用于控制空洞卷积，默认为1，卷积核尺寸不膨胀
        self.groups = groups
        # print(self.kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc_func = createPDCFunc(pdc_func)

    def reset_parameters(self):
        # 凯明初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        #               输入      卷积核权重     偏置                                卷积核膨胀（用于空洞卷积）
        return self.pdc_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)




class adap_conv1(nn.Module):
    def __init__(self, in_channels, out_channels,kz=3,pd=1):
        super(adap_conv1, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='shun', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x

class adap_conv2(nn.Module):
    def __init__(self, in_channels, out_channels,kz=3,pd=1):
        super(adap_conv2, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='ni', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x
class adap_conv3(nn.Module):
    def __init__(self, in_channels, out_channels,kz=5,pd=2):
        super(adap_conv3, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='shun5', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x

class adap_conv4(nn.Module):
    def __init__(self, in_channels, out_channels,kz=5,pd=2):
        super(adap_conv4, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='ni5', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x
class adap_conv5(nn.Module):
    def __init__(self, in_channels, out_channels,kz=7,pd=3):
        super(adap_conv5, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='shun7', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x

class adap_conv6(nn.Module):
    def __init__(self, in_channels, out_channels,kz=7,pd=3):
        super(adap_conv6, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='ni7', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x



class TemporalFeatureFusionModule1(nn.Module):
    def __init__(self, in_d=None, out_d=None):
        super(TemporalFeatureFusionModule1, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        self.conv_branch = nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        # branch 1
        self.conv_branch1_f = adap_conv1(self.in_d, self.out_d, kz=3, pd=1)
        # branch 2
        self.conv_branch2_f = adap_conv2(self.in_d, self.out_d, kz=3, pd=1)

        self.conv_ = nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)


    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        guiding_map = F.sigmoid(guiding_map0)
        x = guiding_map * x
        # branch 1
        x_branch1 = self.conv_branch1_f(x)
        # print(x_branch1.size(), self.conv_branch2(x).size())
        # branch 2
        x_branch2 = self.conv_branch2_f(x)

        xout = torch.cat((x_branch1, x_branch2), dim=1)
        xout = self.conv_(xout)
        x_out = self.relu(xout + self.conv_branch(x))

        return x_out

class TemporalFeatureFusionModule2(nn.Module):
    def __init__(self, in_d=None, out_d=None):
        super(TemporalFeatureFusionModule2, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)

        # branch 1
        self.conv_branch1 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        self.conv_branch1_a = adap_conv3(self.in_d, self.out_d, kz=5, pd=2)
        self.conv_branch1_b = adap_conv4(self.in_d, self.out_d, kz=5, pd=2)
        self.conv_1 = nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        # branch 2
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        self.conv_branch2_a = adap_conv1(self.in_d, self.out_d, kz=3, pd=1)
        self.conv_branch2_b = adap_conv2(self.in_d, self.out_d, kz=3, pd=1)
        self.conv_2 = nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)


    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        guiding_map = F.sigmoid(guiding_map0)
        x = guiding_map * x
        # branch 1
        x_branch1 = self.conv_branch1_a(x)
        x_branch2 = self.conv_branch1_b(x)
        xout1 = torch.cat((x_branch1, x_branch2), dim=1)
        xout1 = self.conv_1(xout1)
        x_out1 = self.relu(xout1 + self.conv_branch1(x))

        x_branch3 = self.conv_branch2_a(x_out1)
        x_branch4 = self.conv_branch2_b(x_out1)
        xout2 = torch.cat((x_branch3, x_branch4), dim=1)
        xout2 = self.conv_2(xout2)
        x_out2 = self.relu(xout2 + self.conv_branch2(x))

        return x_out2


class TemporalFeatureFusionModule3(nn.Module):
    def __init__(self, in_d=None, out_d=None):
        super(TemporalFeatureFusionModule3, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)

        # branch 1
        self.conv_branch1 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        self.conv_branch1_a = adap_conv5(self.in_d, self.out_d, kz=7, pd=3)
        self.conv_branch1_b = adap_conv6(self.in_d, self.out_d, kz=7, pd=3)
        self.conv_1 = nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        # branch 2
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        self.conv_branch2_a = adap_conv3(self.in_d, self.out_d, kz=5, pd=2)
        self.conv_branch2_b = adap_conv4(self.in_d, self.out_d, kz=5, pd=2)
        self.conv_2 = nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        # branch 3
        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)
        self.conv_branch3_a = adap_conv1(self.in_d, self.out_d, kz=3, pd=1)
        self.conv_branch3_b = adap_conv2(self.in_d, self.out_d, kz=3, pd=1)
        self.conv_3 = nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=1, stride=1, groups=1, bias=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        guiding_map = F.sigmoid(guiding_map0)
        x = guiding_map * x
        # branch 1
        x_branch1 = self.conv_branch1_a(x)
        x_branch2 = self.conv_branch1_b(x)
        xout1 = torch.cat((x_branch1, x_branch2), dim=1)
        xout1 = self.conv_1(xout1)
        x_out1 = self.relu(xout1 + self.conv_branch1(x))

        x_branch3 = self.conv_branch2_a(x_out1)
        x_branch4 = self.conv_branch2_b(x_out1)
        xout2 = torch.cat((x_branch3, x_branch4), dim=1)
        xout2 = self.conv_2(xout2)
        x_out2 = self.relu(xout2 + self.conv_branch2(x))

        x_branch5 = self.conv_branch3_a(x_out2)
        x_branch6 = self.conv_branch3_b(x_out2)
        xout3 = torch.cat((x_branch5, x_branch6), dim=1)
        xout3 = self.conv_3(xout3)
        x_out3 = self.relu(xout3 + self.conv_branch3(x))

        return x_out3





class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



#增加decoder解码器，不concat多尺度特征生成guide map,也concat多尺度特征生成最后输出
class CGNet(nn.Module):
    def __init__(self,):
        super(CGNet, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        channles = [64, 128, 256, 512, 512]
        self.bmca = BiMulCrossAttention()

        # self.tffm_x2 = TemporalFeatureFusionModule4()
        self.tffm_x3 = TemporalFeatureFusionModule3(channles[2], channles[2])
        self.tffm_x4 = TemporalFeatureFusionModule2(channles[3], channles[3])
        self.tffm_x5 = TemporalFeatureFusionModule1(channles[4], channles[4])

        self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1)
        self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1)
        self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1)
        self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1)

        self.up_layer4 = BasicConv2d(512,512,3,1,1)
        self.up_layer3 = BasicConv2d(512,512,3,1,1)
        self.up_layer2 = BasicConv2d(256,256,3,1,1)

        self.decoder = nn.Sequential(BasicConv2d(512,64,3,1,1),nn.Conv2d(64,1,3,1,1))
        self.decoder_first = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))
        self.decoder_second = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))
        self.decoder_third = nn.Sequential(BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))

        self.decoder_final = nn.Sequential(BasicConv2d(128,64,3,1,1),nn.Conv2d(64,1,1))

        # self.cgm_2 = ChangeGuideModule(256)
        # self.cgm_3 = ChangeGuideModule(512)
        # self.cgm_4 = ChangeGuideModule(512)

        #相比v2 额外的模块
        self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(1024,512,3,1,1)
        self.decoder_module3 = BasicConv2d(768,256,3,1,1)
        self.decoder_module2 = BasicConv2d(384,128,3,1,1)

    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A
    def forward(self,A,B):

        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)
        print(layer1_A.size(), layer2_A.size(), layer3_A.size(), layer4_A.size())

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        s1_2, s1_3, s1_4, s1_5 = self.bmca(layer1_A, layer1_B, layer2_A, layer2_B, layer3_A, layer3_B, layer4_A, layer4_B)
        s2_2, s2_3, s2_4, s2_5 = self.bmca(layer1_B, layer1_A, layer2_B, layer2_A, layer3_B, layer3_A, layer4_B, layer4_A)
        # layer4_s = torch.cat((layer4_B, layer4_A), dim=1)
        # layer4_s = self.conv_reduce_4(layer4_s)

        # layer1, layer2, layer3, layer4 = self.tfm(layer1_A, layer2_A, layer3_A, layer4_A, layer1_B, layer2_B, layer3_B, layer4_B)

        layer1 = torch.cat((s1_2, s2_2),dim=1)

        layer2 = torch.cat((s1_3, s2_3),dim=1)

        layer3 = torch.cat((s1_4, s2_4),dim=1)

        layer4 = torch.cat((s1_5, s2_5),dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        # # layer4 = self.up_layer4(layer4)
        # #
        # # layer3 = self.up_layer3(layer3)
        # #
        # # layer2 = self.up_layer2(layer2)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        feature_fuse=layer4_1 #需要注释！


        # layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        # layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        # layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        # feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1),dim=1)

        change_map = self.decoder(feature_fuse) #需要注释！

        # ---------------注释这两句------------------------
        # if not self.training:
        #     feature_fuse = layer4_1
        #     feature_fuse = feature_fuse.cpu().detach().numpy()
        #     for num in range(0, 511):
        #         display = feature_fuse[0, num, :, :]  # 第几张影像，第几层特征0-511
        #         plt.figure()
        #         plt.imshow(display)  # [B, C, H,W]
        #         plt.savefig('./test_result/feature_fuse-v6-LEVIR1419/' + 'v6-fuse-' + str(num) + '.png')
        # # change_map = self.decoder(torch.cat((layer1, layer2_1, layer3_1, layer4_1), dim=1))
        # change_map = self.decoder(feature_fuse)
        # ---------------注释这两句------------------------

        layer4 = self.tffm_x5(layer4, change_map)
        first_map = self.decoder_first(layer4)
        first_map = F.interpolate(first_map, size, mode='bilinear', align_corners=True)
        feature4=self.decoder_module4(torch.cat([self.upsample2x(layer4),layer3],1))
        layer3 = self.tffm_x4(feature4, change_map)
        second_map = self.decoder_second(layer3)
        second_map = F.interpolate(second_map, size, mode='bilinear', align_corners=True)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3),layer2],1))
        layer2 = self.tffm_x3(feature3, change_map)
        third_map = self.decoder_third(layer2)
        third_map = F.interpolate(third_map, size, mode='bilinear', align_corners=True)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))
        # layer1 = self.tffm_x2(feature2)

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.decoder_final(layer1)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, first_map, second_map, third_map, final_map



if __name__=='__main__':
    #测试热图
    # net=CGNet().cuda()
    # out=net(torch.rand((2,3,256,256)).cuda(),torch.rand((2,3,256,256)).cuda())

    #测试模型大小
    print('Hi~')
    input_size = 256
    model_restoration = HCGMNet()
    # model_restoration = CGNet()
    from ptflops import get_model_complexity_info
    from torchstat import stat

    # input = torch.rand((3, input_size, input_size))
    # output = model_restoration(input)
    macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    stat(model_restoration, (3, input_size, input_size))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))



from thop import profile
from thop import clever_format
testmodel=CGNet()
#print(summary(testmodel, input_size=[(1,3, 256,256), (1,3, 256,256)]))
dummy_input = torch.randn((1,3, 256,256))
flops, params = profile(testmodel, (dummy_input,dummy_input))
flops, params = clever_format([flops, params], '%.3f')
print('flops: ', flops, 'params: ', params)