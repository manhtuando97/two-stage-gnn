import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


import argparse
import os
import pickle
import random
import shutil
import time
import math

import cross_val
import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
from tripletnet import tripletnet
from triplet_sampler import TripletSampler

from sklearn.neighbors import KNeighborsClassifier

import pickle


# Train the classify ppi graphs based on triplet loss

# evaluate is based on kNN (on validation and test sets)
def evaluate(train_dataset, val_dataset, model, args, name='Validation', max_num_examples=None, writer=False, iteration=0):
    model.eval()
    

    train_labels = []
    train_embeddings = []

    # this could be validation or test set, but name of variables are 'val_...'
    val_labels = []
    val_embeddings = []

         
    labels = []
    preds = []

    for c in val_dataset.keys():
        for graph in val_dataset[c]:
            
            adj = Variable(torch.Tensor([graph.graph['adj']]), requires_grad=False).cuda()
            h0 = Variable(torch.Tensor([graph.graph['feats']]), requires_grad=False).cuda()
            batch_num_nodes = np.array([graph.graph['num_nodes']])
            assign_input = Variable(torch.Tensor(graph.graph['assign_feats']), requires_grad=False).cuda()

            feat = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            #feat = model(graph)
            _, indices = torch.max(feat, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(graph.graph['label'])
                                                            
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    
    return result

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name

def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'



def train(dataset, model, ensemble_model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True, iteration = 0):
    writer_batch_idx = [0, 3, 6, 9]
    #print("Number of training graphs: ", len(dataset[0]) + len(dataset[1]))
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    print("model parameters")
    print(model.parameters())
    
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    
    best_test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []
    val_accs = []
    val_epochs = []
    test_accs = []
    test_epochs = []
    

    # Triplet Sampler of train graphs
    tripletsampler_tr = TripletSampler(dataset)

    # Triplet Net
    TNet = tripletnet(model)

    # Triplet loss function
    criterion = torch.nn.MarginRankingLoss(margin = args.alpha)

    
    for epoch in range(args.num_epochs):
        
        tripletsampler_tr.shuffle()        
       
        total_time = 0
        avg_loss = 0.0
        pretrain_loss = 0.0
        
        iter = 0

        model.train()
        #print('Epoch: ', epoch)
        while not tripletsampler_tr.end():
            begin_time = time.time()
            model.zero_grad()

            if args.triplet == 'exhaust':
                one_triplet = tripletsampler_tr.exhaust_sample()
            else:
                one_triplet = tripletsampler_tr.sampler()


            dist_p, dist_n, embed_a, embed_p, embed_n = TNet(one_triplet['anchor'], one_triplet['pos'], one_triplet['neg'])  
                        
            target = torch.FloatTensor(dist_p.size()).fill_(-1)
            target = Variable(target).cuda()
            L2_loss = args.l2_regularize * (embed_a.norm(2) + embed_p.norm(2) + embed_n.norm(2))
            L1_loss = args.l1_regularize * (embed_a.norm(1) + embed_p.norm(1) + embed_n.norm(1))
            C_loss = 0           
            
            if args.w > 0:
                X = torch.cat((embed_a, embed_p))
                X = torch.cat((X, embed_n))
                X = torch.transpose(X,0,1)
                row_means = torch.mean(X, 1)
                row_means = torch.reshape(row_means, (args.output_dim, 1))
                X_ = X - row_means
                
                X_T = torch.transpose(X_,0,1)
                
                K = torch.mm(X_, X_T) 
                K = K/ 3
                C = torch.zeros([args.output_dim, args.output_dim])
                for i in range(args.output_dim):
                    C[i][i] = math.exp(-i**2 / args.gamma ** 2)
                C_loss += args.w * torch.norm(C)            

            loss = criterion(dist_p, dist_n, target) + L2_loss + L1_loss + C_loss
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss.item()
            pretrain_loss += loss.item()
            
            
            #print('Iter: ', iter, ', pos dist: {0:.5f}'.format(dist_p.cpu().data[0].item()), ' neg_dist: {0:.5f}'.format(dist_n.cpu().data[0].item()),  ', loss: {0:.5f}'.format(loss.cpu().data.item()))
            elapsed = time.time() - begin_time
            total_time += elapsed
                                        
        avg_loss /= iter 
        pretrain_loss /= iter
        print("average training loss after pre-train at epoch " + str(epoch) + " is: " + str(pretrain_loss))

        

        
    # now, train 1 more time with the final mapping layer
    # Triplet Sampler of train graphs
    tripletsampler_tr = TripletSampler(dataset)
    
    
    in_feat = model.map_model.in_features
    pred_layers = []
    pred_layers.append(nn.Linear(in_feat, 64).cuda())
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(64, 32).cuda())
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(32, 2).cuda())
    pred_model = nn.Sequential(*pred_layers)
   
    model.map_model = pred_model
    optimizer_2 = torch.optim.Adam(model.parameters(), lr=0.0005)
        
    result = evaluate(dataset, dataset, model, args, name='Train', max_num_examples=100)
    train_acc = result['acc']
    print("train acc before post-train : " + str(train_acc))

    max_train_acc = 0
    for epoch in range(args.num_epochs * 2):
        posttrain_loss = 0.0
        iter = 0
        
        tripletsampler_tr.shuffle()              
      
        model.train()
        
        while not tripletsampler_tr.end():
            begin_time = time.time()
            #ensemble_model.zero_grad()

            # zero the parameter gradients
            optimizer_2.zero_grad()

            if args.triplet == 'exhaust':
                one_triplet = tripletsampler_tr.exhaust_sample()
            else:
                one_triplet = tripletsampler_tr.sampler()

            anchor_graph = one_triplet['anchor']
            
            label = Variable(torch.LongTensor([int(anchor_graph.graph['label'])])).cuda()
            
            adj = Variable(torch.Tensor([anchor_graph.graph['adj']]), requires_grad=False).cuda()
            h0 = Variable(torch.Tensor([anchor_graph.graph['feats']]), requires_grad=False).cuda()
            batch_num_nodes = np.array([anchor_graph.graph['num_nodes']])
            assign_input = Variable(torch.Tensor(anchor_graph.graph['assign_feats']), requires_grad=False).cuda()

            pred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            
            lamda = torch.tensor(0.).cuda()
            l2_reg = torch.tensor(0.).cuda()
            for param in model.parameters():
                l2_reg += torch.norm(param).cuda()

            loss = F.cross_entropy(F.softmax(pred), label)
            
            posttrain_loss += loss.item()
            iter += 1
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer_2.step()


        posttrain_loss /= iter
        print("training loss at post phase at epoch " + str(epoch) + " is: " + str(posttrain_loss))

        result = evaluate(dataset, dataset, model, args, name='Train', max_num_examples=100)
        train_acc = result['acc']
        print("train acc at epoch " + str(epoch) + " post-train: " + str(train_acc))
        if train_acc > max_train_acc:
            max_train_acc = train_acc


         
    # after training twice, evaluate on validation and test set
    result = evaluate(dataset, dataset, model, args, name='Train', max_num_examples=100)
    train_acc = result['acc']
    print("train acc: " + str(train_acc))
                
    val_acc = 0
    if val_dataset is not None:
        val_result = evaluate(dataset, val_dataset, model, args, name='Validation', iteration=iteration)
        val_acc = val_result['acc']
    if test_dataset is not None:
        test_result = evaluate(dataset, test_dataset, model, args, name='Test', iteration=iteration)
        test_acc = test_result['acc']

       
    return max_train_acc, test_acc, val_acc

def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graph[train_idx:]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


def pkl_task(args, feat=None):
    print("should not reach here")



def benchmark_task_val(args, writer=None, feat='learnable'):
    print("ppi task")
    all_vals = []
    all_tests = []
    all_trains = []

    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    print(len(graphs))

    if args.feature_type == 'learnable':
        # input node features as learnable features
        node_features = torch.empty(args.max_nodes, args.input_dim, requires_grad = True).type(torch.FloatTensor).cuda()
        nn.init.normal_(node_features, std = 2)

        print('Using learnable features')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = node_features[u]

    elif args.feature_type == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif args.feature_type == 'node-label' and 'label' in graphs[0].nodes[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = np.array(G.nodes[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(args.num_iterations):
        train_dataset, test_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
                cross_val.split_train_val(graphs, args, i, max_nodes=args.max_nodes, feat = 'learnable')
        
        if args.method == 'soft-assign':
            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim, final_dim = 'output_dim').cuda()

       
        
        else:
            print('Method: base')
            model = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args, final_dim = 'output_dim').cuda()
        
                    
        # Ensemble model
        # Final mapping layer
        model_2 = encoders.FinalFC(input_dim = args.output_dim, hidden_dims = [64, 32, 16], embedding_dim = args.num_classes)
        
        ensemble_model = encoders.Ensemble(model, model_2)

            
        if args.val == True:
            train_acc, test_acc, val_acc = train(train_dataset, model, ensemble_model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer, iteration=i)
            all_vals.append(val_acc)
            all_tests.append(test_acc)
            all_trains.append(train_acc)
        else:
            train_acc, test_acc, val_acc = train(train_dataset, model, ensemble_model, args, val_dataset=None, test_dataset=test_dataset,
            writer=writer, iteration=i)
            all_tests.append(test_acc)
            all_trains.append(train_acc)

    # Report performance on train
    all_trains_mean = np.mean(all_trains, axis=0)
    all_trains_std = np.std(all_trains, axis=0)
    print("Train performance " + str(args.num_iterations) + " times: " + str(all_trains_mean) + " with std " + str(all_trains_std))
    

    # Report performance on validation
    if args.val == True:
        all_vals_mean = np.mean(all_vals, axis=0)
        all_vals_std = np.std(all_vals, axis=0)
        print("Validation performance " + str(args.num_iterations) + " times: " + str(all_vals_mean) + " with std " + str(all_vals_std))
    
    # Report performance on test
    all_tests_mean = np.mean(all_tests, axis=0)
    all_tests_std = np.std(all_tests, axis=0)
    print("Test performance " + str(args.num_iterations) + " times: " + str(all_tests_mean) + " with std " + str(all_tests_std))


def attack_darpa_task_val(args, writer=None, feat='learnable'):
    all_vals = []
    all_tests = []
    all_trains = []

    if args.datadir == 'attack':
        print('attack data')
        graphs = load_data.read_attackgraph(args.datadir, args.top, max_nodes=args.max_nodes)
    elif args.datadir == 'darpa' or args.datadir == 'taxi':
        print(args.task + ' data')
        graphs = load_data.read_supplementarygraph(args.datadir, args.task, max_nodes=args.max_nodes)
        


    if feat == 'learnable':
        # input node features as learnable features
        node_features = torch.empty(args.max_nodes, args.input_dim, requires_grad = True).type(torch.FloatTensor).cuda()
        nn.init.normal_(node_features, std = 2)

        print('Using learnable features')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = node_features[u]

    elif feat == 'node-labels':
        print('Using node labels as features')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = np.array(G.nodes[u]['label'])


    for i in range(args.num_iterations):
        train_dataset, test_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
                cross_val.split_train_val(graphs, args, i, max_nodes=args.max_nodes, feat = 'learnable')
        
        if args.method == 'soft-assign':
            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim, final_dim = 'output_dim').cuda()
        
        else:
            print('Method: base')
            model = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args, final_dim='output_dim').cuda()

        # Ensemble model
        # Final mapping layer
        model_2 = encoders.FinalFC(input_dim = args.output_dim, hidden_dims = [64, 32, 16], embedding_dim = args.num_classes)
        
        ensemble_model = encoders.Ensemble(model, model_2)

            
        if args.val == True:
            train_acc, test_acc, val_acc = train(train_dataset, model, ensemble_model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer, iteration=i)
            all_vals.append(val_acc)
            all_tests.append(test_acc)
            all_trains.append(train_acc)
        else:
            train_acc, test_acc, val_acc = train(train_dataset, model, ensemble_model, args, val_dataset=None, test_dataset=test_dataset,
            writer=writer, iteration=i)
            all_tests.append(test_acc)
            all_trains.append(train_acc)

    # Report performance on train
    all_trains_mean = np.mean(all_trains, axis=0)
    all_trains_std = np.std(all_trains, axis=0)
    print("Train performance " + str(args.num_iterations) + " times: " + str(all_trains_mean) + " with std " + str(all_trains_std))
    

    # Report performance on validation
    if args.val == True:
        all_vals_mean = np.mean(all_vals, axis=0)
        all_vals_std = np.std(all_vals, axis=0)
        print("Validation performance " + str(args.num_iterations) + " times: " + str(all_vals_mean) + " with std " + str(all_vals_std))
    
    # Report performance on test
    all_tests_mean = np.mean(all_tests, axis=0)
    all_tests_std = np.std(all_tests, axis=0)
    print("Test performance " + str(args.num_iterations) + " times: " + str(all_tests_mean) + " with std " + str(all_tests_std))

    
    
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')

    parser.add_argument('--task', dest='task',
            help='Whether attack task or ppi task')    
    parser.add_argument('--top', dest='top', type=int,
            help='Top 2 or top 3 for attack data')

    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')

    # whether to have a validation sets
    parser.add_argument('--val', dest='val', type=bool,
                        help='Whether to have a validation set')


    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--iterations', dest='num_iterations', type=int,
            help='Number of iterations in benchmark_test_val')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature_type', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=True,
            help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    # triplet loss 
    parser.add_argument('--alpha', dest='alpha', type=float,
            help='margin alpha in the triplet loss')
    parser.add_argument('--triplet', dest='triplet',
            help='how triplet is sampled')

    # norm regularization
    parser.add_argument('--l2_regularize', dest='l2_regularize', type=float,
                       help='coefficient of L2 regularization, to control parameters')
    parser.add_argument('--l1_regularize', dest='l1_regularize', type=float,
                       help='coefficient of L1 reguarization, to enforce sparsity')

    # variance regularization
    parser.add_argument('--w', dest='w', type=float,
            help='weight of covariance matrix regularization')
    parser.add_argument('--gamma', dest='gamma', type=float,
            help='decay rate of variance in the covariance matrix (refer to draft)')

    parser.set_defaults(task='ppi',
                        datadir='data',
                        logdir='log',
                        val=True,
                        dataset='syn1v2',
                        max_nodes=1000,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=1,
                        num_epochs=10,
                        num_iterations=5,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=16,
                        hidden_dim=20,
                        output_dim=10,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1, 
                        alpha = 1.5,
                        triplet = 'random negative',
                        l1_regularize=0,
                        l2_regularize=0,
                        w=0,
                        gamma=2
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        if prog_args.task == 'ppi':
            benchmark_task_val(prog_args)
        else:
            attack_darpa_task_val(prog_args)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        print("there is nothing to do")

    

if __name__ == "__main__":
    main()

