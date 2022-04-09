'''
    -*- coding: utf-8 -*-
    @Time    : 2022/3/22 16:07
    @Author  : Sunset
    @File    : backbone.py
    @Version : v-1.0.0
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from losses import GeMP
from resnet import resnet50

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
class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.base = resnet50(pretrained=True)
        self.NL_3 = nn.ModuleList([Non_local(1024) for i in range(3)])
        self.NL_3_idx = sorted(6 - (i + 1) for i in range(3))

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:  self.NL_3_idx = [-1]
        for i in range(len(self.base.layer3)):
            x = self.base.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        x = self.base.layer4(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class attention(nn.Module):
    def __init__(self, num_class, pool_dim):
        super(attention, self).__init__()
        self.CA = ChannelAttention(pool_dim)
        self.SA = SpatialAttention()
        self.IN = nn.InstanceNorm2d(pool_dim)
        self.BN = nn.BatchNorm2d(pool_dim)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, xv, xt):
        xv_in = self.IN(xv)
        xt_in = self.IN(xt)
        xv_bn = self.BN(xv)
        xt_bn = self.BN(xt)
        y1_v = self.SA(self.CA(xv - xv_in))
        y1_t = self.SA(self.CA(xt - xt_in))
        y2_v = self.SA(self.CA(xv - xv_bn))
        y2_t = self.SA(self.CA(xt - xt_bn))
        yv = y1_v + xt + y2_t
        yt = y1_t + xv + y2_v
        # yv = self.relu(yv)
        # yt = self.relu(yt)
        return yv, yt



class backbone(nn.Module):
    def __init__(self, args,  num_class):
        super(backbone, self).__init__()
        self.base = base_resnet()
        self.attention = attention(num_class)
        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.requires_grad_(False)

        self.classifier_v = nn.Linear(pool_dim, num_class, bias = False)
        self.classifier_t = nn.Linear(pool_dim, num_class, bias = False)
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        self.weight_sid = args.weight_sid
        self.weight_KL = args.weight_KL
        self.update_rate = args.uppdate_rate

        self.classifier = nn.Linear(pool_dim, num_class, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.pool = GeMP()

    def forward(self, xv, xt, mode=0, train='true'):
        b, c, h, w = xv.shape
        if mode == 0:
            xv = self.base(xv)
            xt = self.base(xt)
        elif mode == 1:
            x = self.base(xv)
        elif mode == 2:
            x = self.base(xt)
        if train == 'true':
            xv_recon, xt_recon = self.attention(xv, xt)
            x = torch.cat((xv_recon, xt_recon))
            feat_p = self.pool(x)
            cls_id = self.classifier(self.bottleneck(feat_p))
            feat_p_norm = F.normalize(feat_p, p=2.0, dim=1)
            return {
                'cls_id': cls_id,
                'feat_p': feat_p,
                'feat_p_norm': feat_p_norm,
                # 'loss_recon': res_v['loss']+res_t['loss']
                # 'loss_recon': loss_recon
            }
        else:
            feat_p = self.pool(x)
            cls_id = self.classifier(self.bottleneck(feat_p))
            feat_p_norm = F.normalize(feat_p, p=2.0, dim=1)
            return {
                'cls_id': cls_id,
                'feat_p': feat_p,
                'feat_p_norm': feat_p_norm,
                # 'loss_recon': res_v['loss']+res_t['loss']
                # 'loss_recon': loss_recon
            }
