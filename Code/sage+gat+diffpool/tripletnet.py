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
        # anchor
        a_adj = Variable(torch.Tensor([a.graph['adj']]), requires_grad=False).cuda()
        a_h0 = Variable(torch.Tensor([a.graph['feats']]), requires_grad=False).cuda()
        a_batch_num_nodes = np.array([a.graph['num_nodes']])
        a_assign_input = Variable(torch.Tensor(a.graph['assign_feats']), requires_grad=False).cuda()
        
        # positive
        p_adj = Variable(torch.Tensor([p.graph['adj']]), requires_grad=False).cuda()
        p_h0 = Variable(torch.Tensor([p.graph['feats']]), requires_grad=False).cuda()
        p_batch_num_nodes = np.array([p.graph['num_nodes']])
        p_assign_input = Variable(torch.Tensor(p.graph['assign_feats']), requires_grad=False).cuda()

        # negative
        n_adj = Variable(torch.Tensor([n.graph['adj']]), requires_grad=False).cuda()
        n_h0 = Variable(torch.Tensor([n.graph['feats']]), requires_grad=False).cuda()
        n_batch_num_nodes = np.array([n.graph['num_nodes']])
        n_assign_input = Variable(torch.Tensor(n.graph['assign_feats']), requires_grad=False).cuda()

 
        out_a, embed_a = self.model(a_h0, a_adj, a_batch_num_nodes, assign_x = a_assign_input)
        out_p, embed_p = self.model(p_h0, p_adj, p_batch_num_nodes, assign_x = p_assign_input)
        out_n, embed_n = self.model(n_h0, n_adj, n_batch_num_nodes, assign_x = n_assign_input)

        #dist_p = torch.dist(embed_a, embed_p) ** 2
        #dist_n = torch.dist(embed_a, embed_n) ** 2
        dist_p = F.pairwise_distance(embed_a, embed_p, 2)
        dist_n = F.pairwise_distance(embed_a, embed_n, 2)

        return dist_p, dist_n, embed_a, embed_p, embed_n