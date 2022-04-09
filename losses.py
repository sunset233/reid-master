import torch.nn as nn
import numpy as np
import torch

class GeMP(nn.Module):
  def __init__(self, p=3.0, eps=1e-12):
    super(GeMP, self).__init__()
    self.p = p
    self.eps = eps

  def forward(self, x):
    p, eps = self.p, self.eps
    if x.ndim != 2:
      batch_size, fdim = x.shape[:2]
      x = x.view(batch_size, fdim, -1)
    return (torch.mean(x**p, dim=-1) + eps)**(1/p)


class CrossEntropyLabelSmooth(nn.Module):
  """Cross entropy loss with label smoothing regularizer.
  Reference:
  Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
  Equation: y = (1 - epsilon) * y + epsilon / K.
  Args:
      num_classes (int): number of classes.
      epsilon (float): weight.
  """

  def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.use_gpu = use_gpu
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    """
    Args:
        inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        targets: ground truth labels with shape (num_classes)
    """
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
    if self.use_gpu: targets = targets.cuda()
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (- targets * log_probs).mean(0).sum()
    return loss


class CenterTripletLoss(nn.Module):
  def __init__(self, k_size, margin=0):
    super(CenterTripletLoss, self).__init__()
    self.margin = margin
    self.k_size = k_size
    self.ranking_loss = nn.MarginRankingLoss(margin=margin)

  def forward(self, inputs, targets):
    n = inputs.size(0)

    # Come to centers
    centers = []
    for i in range(n):
      centers.append(inputs[targets == targets[i]].mean(0))
    centers = torch.stack(centers)

    dist_pc = (inputs - centers) ** 2
    dist_pc = dist_pc.sum(1)
    dist_pc = dist_pc.sqrt()

    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, centers, centers.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    # For each anchor, find the hardest positive and negative
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    dist_an, dist_ap = [], []
    for i in range(0, n, self.k_size):
      dist_an.append((self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean())
    dist_an = torch.stack(dist_an)

    # Compute ranking hinge loss
    y = dist_an.data.new()
    y.resize_as_(dist_an.data)
    y.fill_(1)
    loss = dist_pc.mean() + dist_an.mean()
    return loss, dist_pc.mean(), dist_an.mean()

class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim, reduction='mean'):
      super(CenterLoss, self).__init__()
      self.num_classes = num_classes
      self.feat_dim = feat_dim
      self.reduction = reduction

      self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
      """
      Args:
          x: feature matrix with shape (batch_size, feat_dim).
          labels: ground truth labels with shape (batch_size).
      """
      batch_size = x.size(0)
      distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
      distmat.addmm_(1, -2, x, self.centers.t())

      classes = torch.arange(self.num_classes).to(device=x.device, dtype=torch.long)
      labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
      mask = labels.eq(classes.expand(batch_size, self.num_classes))

      loss = distmat * mask.float()

      if self.reduction == 'mean':
        loss = loss.mean()
      elif self.reduction == 'sum':
        loss = loss.sum()

      return loss

class OriTripletLoss(nn.Module):
  """Triplet loss with hard positive/negative mining.

  Reference:
  Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
  Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

  Args:
  - margin (float): margin for triplet.
  """

  def __init__(self, batch_size, margin=0.3):
    super(OriTripletLoss, self).__init__()
    self.margin = margin
    self.ranking_loss = nn.MarginRankingLoss(margin=margin)

  def forward(self, inputs, targets):
    """
    Args:
    - inputs: feature matrix with shape (batch_size, feat_dim)
    - targets: ground truth labels with shape (num_classes)
    """
    n = inputs.size(0)

    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    # For each anchor, find the hardest positive and negative
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    dist_ap, dist_an = [], []
    for i in range(n):
      dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
      dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)

    # Compute ranking hinge loss
    y = torch.ones_like(dist_an)
    loss = self.ranking_loss(dist_an, dist_ap, y)

    # compute accuracy
    correct = torch.ge(dist_an, dist_ap).sum().item()
    return loss, correct


# Adaptive weights
def softmax_weights(dist, mask):
  max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
  diff = dist - max_v
  Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
  W = torch.exp(diff) * mask / Z
  return W


def normalize(x, axis=-1):
  """Normalizing to unit length along the specified dimension.
  Args:
    x: pytorch Variable
  Returns:
    x: pytorch Variable, same shape as input
  """
  x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
  return x


class TripletLoss_WRT(nn.Module):
  """Weighted Regularized Triplet'."""

  def __init__(self):
    super(TripletLoss_WRT, self).__init__()
    self.ranking_loss = nn.SoftMarginLoss()

  def forward(self, inputs, targets, normalize_feature=False):
    if normalize_feature:
      inputs = normalize(inputs, axis=-1)
    dist_mat = pdist_torch(inputs, inputs)

    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
    is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)
    furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
    closest_negative = torch.sum(dist_an * weights_an, dim=1)

    y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
    loss = self.ranking_loss(closest_negative - furthest_positive, y)

    # compute accuracy
    correct = torch.ge(closest_negative, furthest_positive).sum().item()
    return loss, correct


def pdist_torch(emb1, emb2):
  '''
  compute the eucilidean distance matrix between embeddings1 and embeddings2
  using gpu
  '''
  m, n = emb1.shape[0], emb2.shape[0]
  emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
  emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
  dist_mtx = emb1_pow + emb2_pow
  dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
  # dist_mtx = dist_mtx.clamp(min = 1e-12)
  dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
  return dist_mtx


def pdist_np(emb1, emb2):
  '''
  compute the eucilidean distance matrix between embeddings1 and embeddings2
  using cpu
  '''
  m, n = emb1.shape[0], emb2.shape[0]
  emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
  emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
  dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
  # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
  return dist_mtx