import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from encoders import GraphConv

import numpy as np

from set2set import Set2Set

### TODO ###
# Implement other versions of DGATLayer to account for different versions of mechanisms

# 1 GAT Head accounting for directed and edges, can be used for Ver. 0, Ver. 1
class DGATHead(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, dropout=0.0, neg_input_slope = 0.2, concat=True):
        super(DGATHead, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leakyRELU_neg_input_slope = neg_input_slope
        # if the layer containing this head has concat: hidden layer (yes) ~ return activation, final layer (no) ~ return w/o activation
        self.concat = concat

        # the w and a parameters of GAT
        self.w = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.leakyRELU_neg_input_slope)

    def forward(self, input, adj):
        #print(input)
        
        h = torch.mm(input[0], self.w)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1,N).view(N*N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2* self.output_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 *torch.ones_like(e)
        # attention from edges (undirected case)/from incoming edges only(directed case)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        # if the layer needs 'concat', returns the activation 
        if self.concat:
            return F.elu(h_prime)
        else: # otherwise, return raw
            return h_prime

# use the transpose of the adjacency matrix
class DGATHead_T(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, dropout=0.0, neg_input_slope = 0.2, concat=True):
        super(DGATHead_T, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leakyRELU_neg_input_slope = neg_input_slope
        # if the layer containing this head has concat: hidden layer (yes) ~ return activation, final layer (no) ~ return w/o activation
        self.concat = concat

        # the w and a parameters of GAT
        self.w = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.leakyRELU_neg_input_slope)

    def forward(self, input, adj):
        h = torch.mm(input, self.w)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1,N).view(N*N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2* self.out_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 *torch.ones(e)
        # transpose the matrix to account for attention from outcoming edges as well
        adj_T = torch.t(adj)
        attention = torch.where(adj_T > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        # if the layer needs 'concat', returns the activation 
        if self.concat:
            return F.elu(h_prime)
        else: # otherwise, return raw
            return h_prime

# attention mechanism V3 (description: in the meeting slides 20200616)
class DGATHead_V3(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, dropout=0.0, neg_input_slope = 0.2, concat=True):
        super(DGATHead_V3, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leakyRELU_neg_input_slope = neg_input_slope
        # if the layer containing this head has concat: hidden layer (yes) ~ return activation, final layer (no) ~ return w/o activation
        self.concat = concat

        # the w and a parameters of GAT
        self.w = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # the a and b paramters of Ver.3 (refer slides 20200616)
        self.a_coeff = nn.Parameter(torch.ones(1))
        
        self.b_coeff = nn.Parameter(torch.ones(1))
        
        self.leakyrelu = nn.LeakyReLU(self.leakyRELU_neg_input_slope)

    def forward(self, input, adj):
        h = torch.mm(input[0], self.w)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1,N).view(N*N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2* self.output_dim)
        #e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        e = torch.matmul(a_input, self.a).squeeze(2)

        # linear transformation of the adj matrix
        adj_transformed = self.a_coeff * adj + self.b_coeff
        e_transformed = self.leakyrelu(adj_transformed) * e

        zero_vec = -9e15 *torch.ones_like(e)
        
        attention = torch.where(adj> 0, e_transformed, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        # if the layer needs 'concat', returns the activation 
        if self.concat:
            return F.elu(h_prime)
        else: # otherwise, return raw
            return h_prime

# attention mechanism V3.1 (description: in the meeting slides)
class DGATHead_V3_1(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, dropout=0.0, neg_input_slope = 0.2, concat=True):
        super(DGATHead_V3_1, self).__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leakyRELU_neg_input_slope = neg_input_slope
        # if the layer containing this head has concat: hidden layer (yes) ~ return activation, final layer (no) ~ return w/o activation
        self.concat = concat

        # the w and a parameters of GAT
        self.w = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # the a, b, c and d paramters of Ver.3.1 (refer slides 20200616)
        self.a_coeff = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.a_coeff.data, gain=1.414)
        self.b_coeff = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.b_coeff.data, gain=1.414)

        self.c_coeff = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.c_coeff.data, gain=1.414)
        self.d_coeff = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.d_coeff.data, gain=1.414)


        self.leakyrelu = nn.LeakyReLU(self.leakyRELU_neg_input_slope)

    def forward(self, input, adj):
        h = torch.mm(input, self.w)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1,N).view(N*N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2* self.out_dim)
        e_raw = torch.matmul(a_input, self.a).squeeze(2)
        e = self.leakyrelu(c_coeff * e_raw + d_coeff)


        # linear transformation of the adj matrix
        adj_transformed = a_coeff * adj + b_coeff
        
        # attention coefficients on version 3.1
        e_transformed = self.leakyrelu(adj_transformed) * e

        zero_vec = -9e15 *torch.ones(e)
        
        attention = torch.where(adj > 0, e_transformed, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        # if the layer needs 'concat', returns the activation 
        if self.concat:
            return F.elu(h_prime)
        else: # otherwise, return raw
            return h_prime



# a Layer of GAT, consisting of several attention heads. Each attention head is implemented at 'DGATHead'
class DGATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, neg_input_slope=0.2, n_heads=4, concat=True):
        super(DGATLayer, self).__init__()
        self.dropout = dropout
        self.concat = concat   
        self.n_heads = n_heads  

        self.attentions = [DGATHead_V3(input_dim, output_dim, dropout=dropout, neg_input_slope=neg_input_slope, concat=self.concat) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for m in self.modules():
            if isinstance(m, DGATHead_V3):
                m.w.data = init.xavier_uniform(m.w.data, gain=1.414)
                if m.a is not None:
                    m.a.data = init.constant(m.a.data, 0.0)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        if self.concat: # concatenation
            
            out = torch.cat([attention(x, adj) for attention in self.attentions], dim=2) 
        else: # average
            
            attentions = [attention(x, adj) for attention in self.attentions]
            sum = attentions[0]
            for att_head in range(1, self.n_heads):
                sum += attentions[att_head]
            avg = sum / self.n_heads
            out = F.elu(avg)
        return out

class DGATEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim,
                       num_layers=2, num_heads=[4, 6], pred_hidden_dims=[], neg_input_slopes=[0.2, 0.2], dropouts=[0.0, 0.0], final_dim='output_dim', concat=True):

        '''
        list of # heads, list of negative input slopes, list of dropout parameters
        len(num_heads) = len(neg_input_slopes) = len(dropouts) = num_layers
        ''' 

        super(DGATEncoderGraph, self).__init__()
        self.dropout = dropouts
        self.bias = True
        self.num_layers = num_layers
        self.num_aggs = 1

        # specify whether the last vector is either number of classes or embedding_dim
        self.final_dim = final_dim

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                 input_dim, hidden_dim, embedding_dim, num_layers,
                 num_heads, neg_input_slopes, dropouts)

        self.pred_input_dim = embedding_dim
        
        # final prediction model, output dimension is equal to the number of classes
        self.pred_model = self.build_pred_layers(self.pred_input_dim, label_dim, num_aggs=self.num_aggs)
        
        # final mapping model, output dimension is equal to embedding_dim
        self.map_model = self.build_pred_layers(self.pred_input_dim, embedding_dim, num_aggs=self.num_aggs)
  

    # aggregate node embeddings
    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers,
                          num_heads, neg_input_slopes, dropouts):
        print("num layers: " + str(num_layers))
        conv_first = DGATLayer(input_dim = input_dim, output_dim=hidden_dim, n_heads=num_heads[0], dropout=dropouts[0], concat=True)
        if num_layers >= 3:
            conv_block = nn.ModuleList(
                     [DGATLayer(input_dim=hidden_dim * num_heads[i-1], output_dim=hidden_dim, n_heads=num_heads[i], dropout=dropouts[i], concat=True) for i in range(1, num_layers-1)])
        else:
            conv_block = None
        conv_last = DGATLayer(input_dim=hidden_dim*num_heads[-2], output_dim=embedding_dim, n_heads=num_heads[-1], dropout=dropouts[-1], concat=False)
        return conv_first, conv_block, conv_last

    # soft-clusters assign
    def build_assign_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
             normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim = input_dim, output_dim = hidden_dim, add_self=add_self,
                 normalize_embedding=normalize)
        conv_block = nn.ModuleList(
                 [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                            normalize_embedding=normalize, dropout=dropout)
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                 normalize_embedding=normalize)

        return conv_first, conv_block, conv_last
       

    def build_pred_layers(self, pred_input_dim, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        pred_model = nn.Linear(pred_input_dim, label_dim)
        return pred_model

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last):

        '''
        Perform forward propagation with attention.
        Output: embedding matrix dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        
        x_all = [x]
        if conv_block != None:
            for i in range(len(conv_block)):
                x = conv_block[i](x, adj)
                x_all.append(x)
        x = conv_last(x, adj)
        
        x_all.append(x)
        #x_tensor = torch.cat(x_all, dim=2)
        #return x_tensor
        return x


    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # conv
        x = self.conv_first(x, adj)
        
        
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            
        out = self.conv_last(x,adj)
        out_max, _ = torch.max(out, dim=1)

        if self.final_dim != 'output_dim':
            output = self.pred_model(out_max)
        else:
            output = self.map_model(out_max)
        #output = output[0]
        

        return output


class SoftPoolingDGATEncoder(DGATEncoderGraph):
    # when this model is instantiated, embedding_dim = args.output_dim

    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, assign_hidden_dim,
            num_layers=2, num_heads=[4, 6], neg_input_slopes=[0.2, 0.2], dropouts=[0.0, 0.0], 
            assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, final_dim = 'output_dim'):

        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingDGATEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.concat = concat

        # specify whether the last vector is either number of classes or embedding_dim
        self.final_dim = final_dim

        # Graph Attention Layers
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    num_heads, neg_input_slopes, dropouts=dropouts)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment of soft clusters
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_assign_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            #assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred_input_dim = assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, assign_dim, num_aggs=1)


            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), 
                label_dim, num_aggs=self.num_aggs)
        self.map_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), 
                embedding_dim, num_aggs=self.num_aggs)


        for m in self.modules():
            if isinstance(m, DGATHead):
                m.w.data = init.xavier_uniform(m.w.data, gain=nn.init.calculate_gain('relu'))
                
                m.a.data = init.constant(m.a.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            #embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            embedding_mask = None
        else:
            embedding_mask = None

        out_all = []

        
        #embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last)

      
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                #embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                embedding_mask = None
            else:
                embedding_mask = None

            #self.assign_tensor = self.gcn_forward(x_a, adj, self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i], embedding_mask)
            self.assign_tensor = self.gcn_forward(x_a, adj, self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i])



            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))

            '''
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask
            '''
            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x
        
            embedding_tensor = self.gcn_forward(x, adj, 
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i])


            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                #out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)


        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        if self.final_dim != 'output_dim':
            ypred = self.pred_model(output)
        else:
            ypred = self.map_model(output)
        return ypred

    # this loss is only for original diff-pool setting (final vector = predicted class probabilities)
    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingDGATEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.Tensor(1).cuda())
            #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[1-adj_mask.byte()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            #print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss


        
