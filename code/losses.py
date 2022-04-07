import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# references:
#   https://github.com/ronghuaiyang/arcface-pytorch/
#   https://github.com/MuggleWang/CosFace_pytorch
#   https://github.com/clcarwin/sphereface_pytorch
class AdditiveAngularMargin(nn.Module):
    def __init__(self, num_classes, emb_dim, scale, margin, easy_margin):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.class_centers = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.class_centers)

    def forward(self, emb, label):
        cosine = F.linear(F.normalize(emb), F.normalize(self.class_centers)).clamp(-1., 1.)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # phi = torch.cos(theta + self.margin)

        one_hot = torch.zeros(cosine.size(), device=emb.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = one_hot * phi + ((1. - one_hot) * cosine)
        output *= self.scale

        return output


class LargeMarginCosine(nn.Module):
    def __init__(self, num_classes, emb_dim, scale, margin):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.scale = scale
        self.margin = margin

        self.class_centers = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.class_centers)

    def forward(self, emb, label):
        cosine = F.linear(F.normalize(emb), F.normalize(self.class_centers)).clamp(-1., 1.)
        one_hot = torch.zeros(cosine.size(), device=emb.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = self.scale * (cosine - self.margin * one_hot)
        return output


class ASoftmax(nn.Module):
    def __init__(self, num_classes, emb_dim, margin):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.margin = margin
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.lambda_min = 5.0
        self.iters = 0

        self.class_centers = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.class_centers)

        self.mlambda = [
                lambda x: x**0,
                lambda x: x**1,
                lambda x:  2 * x**2 -  1,               # derived from double angle formula
                lambda x:  4 * x**3 -  3 * x,           # cos(2x+x)
                lambda x:  8 * x.pow(4) -  8 * x.pow(2) + 1,
                lambda x: 16 * x**5 - 20 * x**3 + 5 * x
                ]

    def forward(self, emb, label):
        # [batch_size, emb_dim] x [emb_dim, num_classes] -> [batch_size, num_classes]
        cos_theta = F.linear(F.normalize(emb), F.normalize(self.class_centers)).clamp(-1., 1.)

        theta = torch.acos(cos_theta)
        with torch.no_grad():
            k = (self.margin * theta / torch.pi).floor()
        cos_m_theta = self.mlambda[self.margin](cos_theta)
        phi_theta = ((-1.0)**k * cos_m_theta) - 2 * k

        one_hot = torch.zeros(cos_theta.size(), device=emb.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # Appendix G: annealing optimization strategy (gradually reduce self.lamb)
        # reference: https://github.com/wy1iu/sphereface/blob/f5cd440a2233facf46b6529bd13231bb82f23177/tools/caffe-sphereface/src/caffe/layers/margin_inner_product_layer.cpp#L116
        self.iters += 1
        self.lamb = max(self.lambda_min, self.base * (1 + self.gamma * self.iters)**(-1 * self.power))
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta

        emb_norm = torch.linalg.vector_norm(emb, ord=2, dim=1, keepdim=True)
        output = emb_norm * output

        return output


# UniformFace
# https://openaccess.thecvf.com/content_CVPR_2019/papers/Duan_UniformFace_Learning_Deep_Equidistributed_Representation_for_Face_Recognition_CVPR_2019_paper.pdf
def uniform_loss(class_centers, label, compute_in_batch):
    if compute_in_batch:
        centers = class_centers[torch.unique(label)]
    else:
        centers = class_centers

    ci = torch.sum(torch.square(centers), dim=1, keepdim=True)
    ci_cj = torch.matmul(centers, centers.t())
    class_dist_mat = ci + ci.transpose(-1, -2) - 2*ci_cj

    class_distance = torch.triu(class_dist_mat, diagonal=1)
    class_distance = torch.sqrt(class_distance[class_distance != 0])

    num_classes = centers.size(0)
    denom = num_classes * (num_classes -1)
    uniform_loss = torch.sum(1 / (class_distance+1)) / denom

    with torch.no_grad():
        mask = torch.zeros_like(class_dist_mat).fill_diagonal_(1)
        min_dist = torch.min(class_dist_mat[mask==0].view(class_dist_mat.size(0), -1), dim=1)
        min_avg = torch.mean(min_dist[0])
        dist = torch.mean(class_distance)

    return uniform_loss, min_avg, dist

# RegularFace
# https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf
def exclusive_regularization(class_centers, label, compute_in_batch):
    if compute_in_batch:
        centers = class_centers[torch.unique(label)]
    else:
        centers = class_centers
    cosine = F.linear(F.normalize(centers), F.normalize(centers))
    mask = torch.zeros_like(cosine).fill_diagonal_(1)
    max_dist = torch.max(cosine[mask==0].view(cosine.size(0), -1).t(), dim=0)
    return torch.mean(max_dist[0])


def move_class_centers(emb, label, class_centers, beta):
    # beta is update ratio of class centers
    label = label.unsqueeze(-1)
    num_samples_per_center = torch.eq(label, label.transpose(-1, -2)).sum(dim=0)

    centers_in_batch = class_centers[label.squeeze()]
    delta = (centers_in_batch - emb) / (num_samples_per_center.unsqueeze(-1) + 1)
    class_centers.data.scatter_add_(0, label.expand(-1, delta.size(1)), -beta*delta)
