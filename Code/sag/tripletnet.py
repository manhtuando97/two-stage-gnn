import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class tripletnet(nn.Module):
    def __init__(self, model):
        super(tripletnet, self).__init__()
        self.model = model

    def forward(self, a, p, n):
        embed_a = self.model(a)
        embed_p = self.model(p)
        embed_n = self.model(n)

        # dist_p = torch.dist(embed_a, embed_p) ** 2
        # dist_n = torch.dist(embed_a, embed_n) ** 2
        dist_p = F.pairwise_distance(embed_a, embed_p, 2)
        dist_n = F.pairwise_distance(embed_a, embed_n, 2)

        return dist_p, dist_n, embed_a, embed_p, embed_n