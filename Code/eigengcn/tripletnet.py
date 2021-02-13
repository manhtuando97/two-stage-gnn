import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np




class tripletnet(nn.Module):
    def __init__(self, model, args):
        super(tripletnet, self).__init__()
        self.model = model
        self.args = args

    def forward(self, a, p, n):
        # anchor
        a_adj = Variable(torch.Tensor([a.graph['adj']]), requires_grad=False).cuda()
        a_h0 = Variable(torch.Tensor([a.graph['feats']]), requires_grad=False).cuda()
        a_batch_num_nodes = np.array([a.graph['num_nodes']])
        a_assign_input = Variable(torch.Tensor(a.graph['assign_feats']), requires_grad=False).cuda()

        
        adj_pooled_list = []
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in self.args.pool_sizes.split('_')]
        for i in range(len(pool_sizes)):
            ind = i+1
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append(Variable(torch.from_numpy(a.graph[adj_key]).float(), requires_grad=False).cuda())
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append([a.graph[num_nodes_key]])

            pool_matrices_list = []
            for j in range(self.args.num_pool_matrix):
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
                u = Variable( torch.from_numpy(a.graph[pool_adj_key]).float(), requires_grad = False).cuda()
                u_ = u.unsqueeze(0)

                pool_matrices_list.append(u_)

            pool_matrices_dic[i] = pool_matrices_list 

        pool_matrices_list = []
        if self.args.num_pool_final_matrix > 0:
    
            for j in range(self.args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                u = Variable( torch.from_numpy(a.graph[pool_adj_key]).float(), requires_grad = False).cuda()
                u_ = u.unsqueeze(0)
                pool_matrices_list.append(u_)

            pool_matrices_dic[ind] = pool_matrices_list 

        embed_a = self.model(a_h0, a_adj, adj_pooled_list, a_batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)


        
        # positive
        p_adj = Variable(torch.Tensor([p.graph['adj']]), requires_grad=False).cuda()
        p_h0 = Variable(torch.Tensor([p.graph['feats']]), requires_grad=False).cuda()
        p_batch_num_nodes = np.array([p.graph['num_nodes']])
        p_assign_input = Variable(torch.Tensor(p.graph['assign_feats']), requires_grad=False).cuda()

        adj_pooled_list = []
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in self.args.pool_sizes.split('_')]
        for i in range(len(pool_sizes)):
            ind = i+1
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append(Variable(torch.from_numpy(p.graph[adj_key]).float(), requires_grad=False).cuda())
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append([p.graph[num_nodes_key]])

            pool_matrices_list = []
            for j in range(self.args.num_pool_matrix):
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
                u = Variable(torch.from_numpy(p.graph[pool_adj_key]).float(), requires_grad = False).cuda()
                u_ = u.unsqueeze(0)
                pool_matrices_list.append(u_)

            pool_matrices_dic[i] = pool_matrices_list 

        pool_matrices_list = []
        if self.args.num_pool_final_matrix > 0:
    
            for j in range(self.args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                u = Variable(torch.from_numpy(p.graph[pool_adj_key]).float(), requires_grad = False).cuda()
                u_ = u.unsqueeze(0)
                pool_matrices_list.append(u_)

            pool_matrices_dic[ind] = pool_matrices_list 
 

        embed_p = self.model(p_h0, p_adj, adj_pooled_list, p_batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)



        # negative
        n_adj = Variable(torch.Tensor([n.graph['adj']]), requires_grad=False).cuda()
        n_h0 = Variable(torch.Tensor([n.graph['feats']]), requires_grad=False).cuda()
        n_batch_num_nodes = np.array([n.graph['num_nodes']])
        n_assign_input = Variable(torch.Tensor(n.graph['assign_feats']), requires_grad=False).cuda()

        adj_pooled_list = []
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in self.args.pool_sizes.split('_')]
        for i in range(len(pool_sizes)):
            ind = i+1
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append(Variable(torch.from_numpy(n.graph[adj_key]).float(), requires_grad=False).cuda())
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append([n.graph[num_nodes_key]])

            pool_matrices_list = []
            for j in range(self.args.num_pool_matrix):
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)
                u = Variable(torch.from_numpy(n.graph[pool_adj_key]).float(), requires_grad = False).cuda()
                u_ = u.unsqueeze(0)

                pool_matrices_list.append(u_)

            pool_matrices_dic[i] = pool_matrices_list 

        pool_matrices_list = []
        if self.args.num_pool_final_matrix > 0:
    
            for j in range(self.args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                u = Variable(torch.from_numpy(n.graph[pool_adj_key]).float(), requires_grad = False).cuda()
                u_ = u.unsqueeze(0)

                pool_matrices_list.append(u_)

            pool_matrices_dic[ind] = pool_matrices_list 

        embed_n = self.model(n_h0, n_adj, adj_pooled_list, n_batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)


 
        
        #dist_p = torch.dist(embed_a, embed_p) ** 2
        #dist_n = torch.dist(embed_a, embed_n) ** 2
        dist_p = F.pairwise_distance(embed_a, embed_p, 2)
        dist_n = F.pairwise_distance(embed_a, embed_n, 2)

        return dist_p, dist_n, embed_a, embed_p, embed_n