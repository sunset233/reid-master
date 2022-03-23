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


class CMAlign(nn.Module):
    def __init__(self, batch_size=8, num_pos=4, temperature=50):
        super(CMAlign, self).__init__()
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.criterion = nn.TripletMarginLoss(margin=0.3, p=2.0, reduce=False)
        self.temperature = temperature

    def _random_pairs(self):
        batch_size = self.batch_size
        num_pos = self.num_pos

        pos = []
        for batch_index in range(batch_size):
            pos_idx = random.sample(list(range(num_pos)), num_pos)
            pos_idx = np.array(pos_idx) + num_pos * batch_index
            pos = np.concatenate((pos, pos_idx))
        pos = pos.astype(int)

        neg = []
        for batch_index in range(batch_size):
            batch_list = list(range(batch_size))
            batch_list.remove(batch_index)

            batch_idx = random.sample(batch_list, num_pos)
            neg_idx = random.sample(list(range(num_pos)), num_pos)

            batch_idx, neg_idx = np.array(batch_idx), np.array(neg_idx)
            neg_idx = batch_idx * num_pos + neg_idx
            neg = np.concatenate((neg, neg_idx))
        neg = neg.astype(int)

        return {'pos': pos, 'neg': neg}

    def _define_pairs(self):
        pairs_v = self._random_pairs()
        pos_v, neg_v = pairs_v['pos'], pairs_v['neg']

        pairs_t = self._random_pairs()
        pos_t, neg_t = pairs_t['pos'], pairs_t['neg']

        pos_v += self.batch_size * self.num_pos
        neg_v += self.batch_size * self.num_pos

        return {'pos': np.concatenate((pos_v, pos_t)), 'neg': np.concatenate((neg_v, neg_t))}

    def feature_similarity(self, feat_q, feat_k):
        batch_size, fdim, h, w = feat_q.shape
        feat_q = feat_q.view(batch_size, fdim, -1)
        feat_k = feat_k.view(batch_size, fdim, -1)

        feature_sim = torch.bmm(F.normalize(feat_q, dim=1).permute(0, 2, 1), F.normalize(feat_k, dim=1))
        return feature_sim

    def matching_probability(self, feature_sim):
        M, _ = feature_sim.max(dim=-1, keepdim=True)
        feature_sim = feature_sim - M  # for numerical stability
        exp = torch.exp(self.temperature * feature_sim)
        exp_sum = exp.sum(dim=-1, keepdim=True)
        return exp / exp_sum

    def soft_warping(self, matching_pr, feat_k):
        batch_size, fdim, h, w = feat_k.shape
        feat_k = feat_k.view(batch_size, fdim, -1)
        feat_warp = torch.bmm(matching_pr, feat_k.permute(0, 2, 1))
        feat_warp = feat_warp.permute(0, 2, 1).view(batch_size, fdim, h, w)

        return feat_warp

    def reconstruct(self, mask, feat_warp, feat_q):
        return mask * feat_warp + (1.0 - mask) * feat_q

    def compute_mask(self, feat):
        batch_size, fdim, h, w = feat.shape
        norms = torch.norm(feat, p=2, dim=1).view(batch_size, h * w)

        norms -= norms.min(dim=-1, keepdim=True)[0]
        norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
        mask = norms.view(batch_size, 1, h, w)

        return mask.detach()

    def compute_comask(self, matching_pr, mask_q, mask_k):
        batch_size, mdim, h, w = mask_q.shape
        mask_q = mask_q.view(batch_size, -1, 1)
        mask_k = mask_k.view(batch_size, -1, 1)
        comask = mask_q * torch.bmm(matching_pr, mask_k)

        comask = comask.view(batch_size, -1)
        comask -= comask.min(dim=-1, keepdim=True)[0]
        comask /= comask.max(dim=-1, keepdim=True)[0] + 1e-12
        comask = comask.view(batch_size, mdim, h, w)

        return comask.detach()

    def forward(self, feat_s, feat_c):
        mask = self.compute_mask(feat_s)
        feat_sim = self.feature_similarity(feat_s, feat_c)
        matching_pr = self.matching_probability(feat_sim)
        # reconstruct positive samples
        feat_recon = self.reconstruct(mask, feat_s, feat_c)
        # reconstruct negative samples
        mask_c = self.compute_mask(feat_c)
        # comask = self.compute_comask(matching_pr, feat_c, feat_s)
        feat_recon_c = self.reconstruct(mask_c, feat_c, feat_s)
        loss = torch.mean(self.criterion(feat_s, feat_recon, feat_recon_c))
        return {'feat': feat_recon, 'loss': loss}
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

    def similarity(self, feat_p, feat_q):
        feat_sim = torch.bmm(F.normalize(feat_p, dim=1).permute(0, 2, 1), F.normalize(feat_q, dim=1))
        return feat_sim

    def match_probability(self, feat_sim):
        M, _ = feat_sim.max(dim=-1, keepdim=True)
        feat_sim = feat_sim - M
        exp = torch.exp(feat_sim * 50)
        exp_sum = exp.sum(dim=-1, keepdim=True)
        relation = exp / exp_sum
        return relation
    def forward(self, xv, xt): # xv: [32, 2048, 18, 9]  xt: [32, 2048, 18, 9]

        b, c, h, w = xv.size()
        xv = xv.view(b, c, -1)
        xt = xt.view(b, c, -1)
        feat_sim = self.similarity(xv, xt)
        relation = self.match_probability(feat_sim)

        relation = F.softmax(torch.mean(relation, -1), dim=-1) + 1
        feat_v = xv * relation.unsqueeze(1) # [32, 2048, 162] @ [32, 162, 162]
        feat_t = xt * relation.unsqueeze(1)
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
class feat_exat(nn.Module):
    def __init__(self, drop_last_stride=False):
        self.resnet = resnet50(pretrained=True, drop_last_stride=drop_last_stride)
        if self.non_local == 'on':
            self.NL_3 = nn.ModuleList([Non_local(1024) for i in range(3)])
            self.NL_3_idx = sorted(6-(i+1) for i in range(3))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.relu(x)
        x = self.resnet.bn1(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        feat = self.resnet.layer3(x)
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:  self.NL_3_idx = [-1]
        for i in range(len(self.base_resnet.layer3)):
            x = self.base_resnet.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        feat = self.resnet.layer4(feat)
        feat_local = self.resnet.layer4(x)
        return feat, feat_local

class network(nn.Module):
    def __init__(self, args, class_num, no_local='on', arch='resnet50'):
        super(network, self).__init__()
        self.thermal = thermal_module(arch=arch)
        self.visible = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        # self.baseline = resnet50(pretrained=True, drop_last_stride=True)
        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.requires_grad_(False)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.pool = GeMP()
        self.non_local = no_local
        if self.non_local == 'on':
            self.NL_2 = nn.ModuleList([Non_local(512) for i in range(2)])
            self.NL_2_idx = sorted(4-(i+1) for i in range(2))
            self.NL_3 = nn.ModuleList([Non_local(1024) for i in range(3)])
            self.NL_3_idx = sorted(6-(i+1) for i in range(3))
        self.attention = baseline.attention(class_num)
        self.squeeze = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=1, stride=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.fusion = CMAlign()
    def forward(self, x_v, x_t, mode=0, train='true'):
        b, c, h, w = x_v.shape
        if mode == 0:
            x_v = self.visible(x_v)
            x_t = self.thermal(x_t)
            x = torch.cat((x_v, x_t), dim=0)
        elif mode == 1:
            x = self.visible(x_v)
        elif mode == 2:
            x = self.thermal(x_t)
        x_origin = x
        if self.non_local == 'on':
            x = self.base_resnet.layer1(x_origin)
            x = self.base_resnet.layer2(x)
            # x = self.base_resnet.layer3(x)
            NL3_counter = 0
            if len(self.NL_3_idx) == 0:  self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.layer3)):
                x = self.base_resnet.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            x_self = self.base_resnet.layer4(x)
        x = self.base_resnet(x_origin)
        if train == 'true':
            xv_recon, xt_recon = self.attention(x_self[:b], x_self[b:])
            x_self = torch.cat((xv_recon, xt_recon))
            feat_p = self.pool(x_self)
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
            feat_p = self.pool(x_self)
            cls_id = self.classifier(self.bottleneck(feat_p))
            feat_p_norm = F.normalize(feat_p, p=2.0, dim=1)
            return {
                'cls_id': cls_id,
                'feat_p': feat_p,
                'feat_p_norm': feat_p_norm
            }


