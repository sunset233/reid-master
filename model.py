import random

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
        batch_size, fdim, h, w = feat_s.shape

        pairs = self._define_pairs()
        pos_idx, neg_idx = pairs['pos'], pairs['neg']

        # feat_target_pos = feat_s[pos_idx]
        feat_sim = self.feature_similarity(feat_s, feat_c)
        matching_pr = self.matching_probability(feat_sim)
        feat_wrap = self.soft_warping(matching_pr, feat_s)
        feat_recon = self.reconstruct(mask, feat_wrap, feat_s)

        feat_wrap_c = self.soft_warping(matching_pr, feat_c)
        feat_recon_c = self.reconstruct(mask, feat_wrap_c, feat_c)
        # comask = self.compute_comask(matching_pr, mask, mask[pos_idx])
        loss = torch.mean(self.criterion(feat_s, feat_recon, feat_recon_c))


        # comask_pos = self.compute_comask(matching_pr, mask, mask[pos_idx])
        # feat_wrap_pos = self.soft_warping(matching_pr, feat_c)
        # feat_recon_pos = self.reconstruct(mask, feat_wrap_pos, feat_s)
        #
        # feat_target_neg = feat_s[neg_idx]
        # feat_sim = self.feature_similarity(feat_s, feat_target_neg)
        # matching_pr = self.matching_probability(feat_sim)
        #
        # feat_wrap = self.soft_warping(matching_pr, feat_target_neg)
        # feat_recon_neg = self.reconstruct(mask, feat_wrap, feat_s)
        # loss = torch.mean(comask_pos * self.criterion(feat_s, feat_recon_pos, feat_recon_neg))


        # feat = torch.cat([feat_v, feat_t], dim=0)
        # mask = self.compute_mask(feat)
        # batch_size, fdim, h, w = feat.shape
        #
        # pairs = self._define_pairs()
        # pos_idx, neg_idx = pairs['pos'], pairs['neg']
        #
        # # positive
        # feat_target_pos = feat[pos_idx]
        # feature_sim = self.feature_similarity(feat, feat_target_pos)
        # matching_pr = self.matching_probability(feature_sim)
        #
        # comask_pos = self.compute_comask(matching_pr, mask, mask[pos_idx])
        # feat_warp_pos = self.soft_warping(matching_pr, feat_target_pos)
        # feat_recon_pos = self.reconstruct(mask, feat_warp_pos, feat)
        #
        # # negative
        # feat_target_neg = feat[neg_idx]
        # feature_sim = self.feature_similarity(feat, feat_target_neg)
        # matching_pr = self.matching_probability(feature_sim)
        #
        # feat_warp = self.soft_warping(matching_pr, feat_target_neg)
        # feat_recon_neg = self.reconstruct(mask, feat_warp, feat)
        #
        # loss = torch.mean(comask_pos * self.criterion(feat, feat_recon_pos, feat_recon_neg))

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
        self.in_c1 = 64
        self.conv1 = nn.Conv2d(self.in_c1, self.in_c1//2, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_c1//2, self.in_c1, kernel_size=1)
        self.in_c2 = 12
        self.conv3 = nn.Conv2d(self.in_c2, self.in_c2 // 2, kernel_size=1)
        self.conv4 = nn.Conv2d(self.in_c2 // 2, self.in_c2, kernel_size=1)

    # def get_attention(self, a): # [32, 162, 162]
    #     input_a = a
    #     a = a.unsqueeze(0)
    #     if a.size(1)==64:
    #         a = F.relu(self.conv1(a))
    #         a = self.conv2(a)
    #     elif a.size(1)==12:
    #         a = F.relu(self.conv3(a))
    #         a = self.conv4(a)
    #     a = a.squeeze(0)
    #     a = torch.mean(input_a*a, -1)
    #     a = F.softmax(a/0.05, dim=-1) + 1
    #     return a

    def forward(self, xv, xt): # xv: [32, 2048, 18, 9]  xt: [32, 2048, 18, 9]
        b, c, h, w = xv.size()
        xv = xv.view(b, c, -1)
        xt = xt.view(b, c, -1)
        feat_sim = torch.bmm(normalize(xv).permute(0, 2, 1), normalize(xt))
        M, _ = feat_sim.max(dim=-1, keepdim=True)
        feat_sim = feat_sim - M
        exp = torch.exp(feat_sim * 50)
        exp_sum = exp.sum(dim=-1, keepdim = True)
        relation = exp / exp_sum
        relation = F.softmax(torch.mean(relation, -1), dim=-1) + 1
        feat_v = xv * relation.unsqueeze(1) # [32, 2048, 162] @ [32, 162, 162]
        feat_t = xt * relation.unsqueeze(1)
        feat_v = feat_v.view(b, c, h, w)
        feat_t = feat_t.view(b, c, h, w)
        return feat_v, feat_t

class feature_fusion(nn.Module):
    '''
        This section we use SMGM method to merge the features.
    '''
    def __init__(self, channels=2048, r=4):
        super(feature_fusion, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
    def forward(self, feat_self, feat_cross, mode = 0):
        '''

        :param feat_self:
        :param feat_cross:
        :return:
        '''
        if mode == 0: # make visible images
            pass
        else: # make thermal images
            pass
        pass

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
            self.NL_2 = nn.ModuleList([Non_local(512) for i in range(2)])
            self.NL_2_idx = sorted(4-(i+1) for i in range(2))
            self.NL_3 = nn.ModuleList([Non_local(1024) for i in range(3)])
            self.NL_3_idx = sorted(6-(i+1) for i in range(3))
        self.feat_cross = cross_attention()
        self.fusion = CMAlign()
    def forward(self, x_v, x_t, mode=0):
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
            # NL2_counter = 0
            # if len(self.NL_2_idx) == 0:  self.NL_2_idx = [-1]
            # for i in range(len(self.base_resnet.layer2)):
            #     x = self.base_resnet.layer2[i](x)
            #     if i == self.NL_2_idx[NL2_counter]:
            #         _, C, H, W = x.shape
            #         x = self.NL_2[NL2_counter](x)
            #         NL2_counter += 1
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
        batch_size, fdim, h, w = x.shape
        xv = x[:batch_size//2]
        xt = x[batch_size//2:]
        xv_c, xt_c = self.feat_cross(xv, xt)
        xv_s, xt_s = x_self[:batch_size//2], x_self[batch_size//2:]
        res_v = self.fusion(xv_s, xt_c)
        res_t = self.fusion(xt_s, xv_c)
        x_recon = torch.cat((res_v['feat'], res_t['feat']), dim=0)



        feat_p = self.pool(x_recon)
        cls_id = self.classifier(self.bottleneck(feat_p))
        feat_p_norm = F.normalize(feat_p, p=2.0, dim=1)
        # loss_recon = (res_v['loss'] + res_t['loss'])/2
        return {
            'cls_id': cls_id,
            'feat_p': feat_p,
            'feat_p_norm': feat_p_norm,
            # 'loss_recon': loss_recon
        }

