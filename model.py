import random

import torch.nn as nn
import torch
from torch.nn import init

import baseline
import fusion
from CBAM import cbam
from resnet import resnet50
import torch.nn.functional as F
from losses import *
from fusion import *
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class SAB(nn.Module):
    def __init__(self, in_channel):
        super(SAB, self).__init__()
        self.IN = nn.InstanceNorm2d(in_channel)
        self.softmax = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(4, 1, kernel_size=1)
    def forward(self, x):
        x_in = self.IN(x)
        x_f = x + x_in
        x_avg = torch.mean(x_in, dim=1, keepdim=True)
        x_max = torch.max(x_in, dim=1, keepdim=True).values
        x_avg_avg = torch.mean(x_avg, dim=2, keepdim=True)
        x_avg_max = torch.max(x_avg, dim=2, keepdim=True).values
        x_max_avg = torch.mean(x_max, dim=2, keepdim=True)
        x_max_max = torch.max(x_max, dim=2, keepdim=True).values
        x_avg_avg2 = torch.mean(x_avg, dim=3, keepdim=True)
        x_avg_max2 = torch.max(x_avg, dim=3, keepdim=True).values
        x_max_avg2 = torch.mean(x_max, dim=3, keepdim=True)
        x_max_max2 = torch.max(x_max, dim=3, keepdim=True).values
        x_avg_avg = self.softmax(x_avg_avg2 * x_avg_avg)
        x_avg_max = self.softmax(x_avg_max2 * x_avg_max)
        x_max_avg = self.softmax(x_max_avg2 * x_max_avg)
        x_max_max = self.softmax(x_max_max2 * x_max_max)
        x = torch.cat([x_avg_avg, x_avg_max, x_max_avg, x_max_max], dim=1)
        x = self.conv1(x)
        x = x_f * x
        return self.relu(self.bn(x))

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()
        visible = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.conv1 = visible.conv1
        self.bn1 = visible.bn1
        self.relu = visible.relu
        self.maxpool = visible.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()
        thermal = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.conv1 = thermal.conv1
        self.bn1 = thermal.bn1
        self.relu = thermal.relu
        self.maxpool = thermal.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
from multihead_attention import *
class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', no_local='on'):
        super(base_resnet, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.base = base
        self.non_local = no_local
        self.pool = nn.AvgPool2d(2, 2)
        self.mhsa = MHSA(n_dims=2048, width=9, height=18, heads=8)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU()
        if self.non_local == 'on':
            self.NL_3 = nn.ModuleList([Non_local(1024) for i in range(3)])
            self.NL_3_idx = sorted(6 - (i + 1) for i in range(3))

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.base.layer3)):
            x = self.base.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # x = self.base.layer3(x)
        x = self.base.layer4(x)
        # for i in range(len(self.base.layer4)):
        #     if i == 1:
        #         x = self.pool(self.mhsa(x))
        #         x = self.bn(x)
        #         continue
        #     x = self.base.layer4[i](x)
        #     x = self.relu(x)
        return x


class network(nn.Module):
    def __init__(self, args, class_num, no_local='on', arch='resnet50'):
        super(network, self).__init__()
        self.thermal = thermal_module(arch=arch)
        self.visible = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.pool = GeMP()
        self.attention = baseline.attention(class_num, 2048)

    def forward(self, x_v, x_t, mode=0, train='true'):
        bsz = x_v.size(0)
        if mode == 0:
            x_v = self.visible(x_v)
            x_t = self.thermal(x_t)
            x = torch.cat((x_v, x_t), dim=0)
        elif mode == 1:
            x = self.visible(x_v)
        elif mode == 2:
            x = self.thermal(x_t)
        feat = self.base_resnet(x)
        if train == 'true':
            xv_4, xt_4 = self.attention(feat[:bsz], feat[bsz:])
            feat4_recon = torch.cat((xv_4, xt_4))

            feat_p = self.pool(feat4_recon)
            cls_id = self.classifier(self.bottleneck(feat_p))
            return {
                'cls_id': cls_id,
                'feat_p': feat_p
            }
        else:
            feat_p = self.pool(feat)
            cls_id = self.classifier(self.bottleneck(feat_p))
            feat_p_norm = F.normalize(feat_p, p=2.0, dim=1)
            return {
                'cls_id': cls_id,
                'feat_p': feat_p,
                'feat_p_norm': feat_p_norm
            }


