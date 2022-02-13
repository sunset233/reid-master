import torch.nn as nn
import torch
from torch.nn import init
from resnet import resnet50
import torch.nn.functional as F
from losses import *

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

class cross_attention(nn.Module):
    def __init__(self):
        super(cross_attention, self).__init__()
        self.in_c1 = 64
        self.conv1 = nn.Conv2d(self.in_c1, self.in_c1//2, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_c1//2, self.in_c1, kernel_size=1)
        self.in_c2 = 12
        self.conv3 = nn.Conv2d(self.in_c2, self.in_c2 // 2, kernel_size=1)
        self.conv4 = nn.Conv2d(self.in_c2 // 2, self.in_c2, kernel_size=1)

    def get_attention(self, a):
        input_a = a
        a = a.unsqueeze(0)
        if a.size(1)==64:
            a = F.relu(self.conv1(a))
            a = self.conv2(a)
        elif a.size(1)==12:
            a = F.relu(self.conv3(a))
            a = self.conv4(a)
        a = a.squeeze(0)
        a = torch.mean(input_a*a, -1)
        a = F.softmax(a/0.05, dim=-1) + 1
        return a

    def forward(self, xv, xt): # xv: [32, 2048, 18, 9]  xt: [32, 2048, 18, 9]
        b, c, h, w = xv.size()
        x = torch.cat((xv, xt))
        x = x.view(2*b, c, -1)
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-12)
        xv_norm = x_norm[:b]
        xt_norm = x_norm[b:]
        a = torch.matmul(xv_norm.transpose(1,2), xt_norm)
        a = torch.cat((a, a.transpose(1, 2)))
        a = self.get_attention(a)
        feat = x * a.unsqueeze(1)
        feat_v = feat[:b]
        feat_t = feat[b:]
        feat_v = feat_v.view(b, c, h, w)
        feat_t = feat_t.view(b, c, h, w)
        return feat_v, feat_t
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

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
class network(nn.Module):
    def __init__(self, args, class_num, no_local='on', arch='resnet50'):
        super(network, self).__init__()
        self.thermal = thermal_module(arch=arch)
        self.visible = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.requires_grad_(False)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.pool = GeMP()

        self.non_local = no_local
        if self.non_local == 'on':
            self.NL_3 = nn.ModuleList([Non_local(1024) for i in range(3)])
            self.NL_3_idx = sorted(6-(i+1) for i in range(3))
        self.feat_cross = cross_attention()
    def forward(self, x_v, x_t, mode=0):
        if mode == 0:
            x_v = self.visible(x_v)
            x_t = self.thermal(x_t)
            x = torch.cat((x_v, x_t), dim=0)
        elif mode == 1:
            x = self.visible(x_v)
        elif mode == 2:
            x = self.thermal(x_t)
        if self.non_local == 'on':
            x = self.base_resnet.layer1(x)
            x = self.base_resnet.layer2(x)
            NL3_counter = 0
            if len(self.NL_3_idx) == 0:  self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.layer3)):
                x = self.base_resnet.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            x = self.base_resnet.layer4(x)
        else:
            x = self.base_resnet(x)
        batch_size, fdim, h, w = x.shape
        xv = x[:batch_size//2]
        xt = x[batch_size//2:]
        xv_c, xt_c = self.feat_cross(xv, xt)

        feat_p = self.pool(x)
        cls_id = self.classifier(self.bottleneck(feat_p))
        feat_p_norm = F.normalize(feat_p, p=2.0, dim=1)
        return {
            'cls_id': cls_id,
            'feat_p': feat_p,
            'feat_p_norm': feat_p_norm
        }

