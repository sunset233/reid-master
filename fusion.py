# import torch.nn as nn
# import torch
# import torch.nn.functional as F
#
#
# class fusion(nn.Module):
#     def __init__(self):
#         super(fusion, self).__init__()
#
#     def compute_mask(self, feat): # 计算掩码矩阵
#         batch_size, fdim, h, w = feat.shape
#         norms = torch.norm(feat, p=2, dim=1).view(batch_size, h * w)
#
#         norms -= norms.min(dim=-1, keepdim=True)[0]
#         norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
#         mask = norms.view(batch_size, 1, h, w)
#
#         return mask.detach()
#     def compute_loss(self, x_self, x_recon):
#         return torch.mean(torch.abs(x_self - x_recon))
#
#     def forward(self, x_self, x, x_cross): # 前向传播
#         mask = self.compute_mask(x_self) # 计算掩码矩阵
#         match_pr = x_cross / x # 计算匹配可能性
#         feat_wrap = x_cross * match_pr # 自注意力和匹配性的结合
#         feat_recon = feat_wrap * mask + x_cross * (1 - mask) # 重构特征
#         loss_recon = self.compute_loss(x_self, feat_recon)
#         return {
#             'feat': feat_recon,
#             'loss': loss_recon
#         }
# def unit_test():
#     import numpy as np
#     x = torch.tensor(np.random.rand(32, 2048, 18, 9).astype(np.float32))
#     model = fusion(output=2048)
#     y = model(x, x)
#     print('output shape:', y.shape)
#     assert y.shape == (32, 2048, 18, 9), 'output shape (2,1,480,640) is expected!'
#     print('test ok!')
#
# if __name__ == '__main__':
#     unit_test()
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

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

    def forward(self, feat_v, feat_t):
        feat = torch.cat([feat_v, feat_t], dim=0)
        mask = self.compute_mask(feat)

        pairs = self._define_pairs()
        pos_idx, neg_idx = pairs['pos'], pairs['neg']

        # positive
        feat_target_pos = feat[pos_idx]
        feature_sim = self.feature_similarity(feat, feat_target_pos)
        matching_pr = self.matching_probability(feature_sim)

        comask_pos = self.compute_comask(matching_pr, mask, mask[pos_idx])
        feat_warp_pos = self.soft_warping(matching_pr, feat_target_pos)
        feat_recon_pos = self.reconstruct(mask, feat_warp_pos, feat)

        # negative
        feat_target_neg = feat[neg_idx]
        feature_sim = self.feature_similarity(feat, feat_target_neg)
        matching_pr = self.matching_probability(feature_sim)

        feat_warp = self.soft_warping(matching_pr, feat_target_neg)
        feat_recon_neg = self.reconstruct(mask, feat_warp, feat)

        loss = torch.mean(comask_pos * self.criterion(feat, feat_recon_pos, feat_recon_neg))

        return {'feat': feat_recon_pos, 'loss': loss}