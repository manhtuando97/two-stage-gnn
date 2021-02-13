import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable            

import argparse
import os
import pickle
import random
import time

import encoders as encoders
import gen.feat as featgen
from graph_sampler import GraphSampler
import load_data
from coarsen_pooling_with_last_eigen_padding import Graphs as gp
import graph 
import time
import pickle
import cross_val
from tripletnet import tripletnet
from triplet_sampler import TripletSampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def evaluate(train_dataset, val_dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()


    train_labels = []
    train_embeddings = []

    val_labels = []
    val_embeddings = []


    for batch_idx, data in enumerate(train_dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        train_labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()

        adj_pooled_list = []
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        for i in range(len(pool_sizes)):
            ind = i+1
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append( Variable(data[adj_key].float(), requires_grad = False ).cuda())
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append(data[num_nodes_key])

            pool_matrices_list = []
            for j in range(args.num_pool_matrix):
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)

                pool_matrices_list.append(Variable( data[pool_adj_key].float(), requires_grad = False).cuda())

            pool_matrices_dic[i] = pool_matrices_list 

        pool_matrices_list = []
        if args.num_pool_final_matrix > 0:

            for j in range(args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                pool_matrices_list.append(Variable( data[pool_adj_key].float(), requires_grad = False).cuda())

            pool_matrices_dic[ind] = pool_matrices_list 


        feat =model( h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
        learned_feat = feat[0].cpu().data.numpy()
        train_embeddings.append(learned_feat)

    
    for batch_idx, data in enumerate(val_dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        val_labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()

        adj_pooled_list = []
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        for i in range(len(pool_sizes)):
            ind = i+1
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append( Variable(data[adj_key].float(), requires_grad = False ).cuda())
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append(data[num_nodes_key])

            pool_matrices_list = []
            for j in range(args.num_pool_matrix):
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)

                pool_matrices_list.append(Variable( data[pool_adj_key].float(), requires_grad = False).cuda())

            pool_matrices_dic[i] = pool_matrices_list 

        pool_matrices_list = []
        if args.num_pool_final_matrix > 0:

            for j in range(args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                pool_matrices_list.append(Variable( data[pool_adj_key].float(), requires_grad = False).cuda())

            pool_matrices_dic[ind] = pool_matrices_list 


        feat =model( h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
        learned_feat = feat[0].cpu().data.numpy()
        val_embeddings.append(learned_feat)

        
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_embeddings, train_labels)

    val_preds = neigh.predict(val_embeddings)
    train_preds = neigh.predict(train_embeddings)
    
    result = {'prec': metrics.precision_score(val_labels, val_preds, average='macro'),
              'recall': metrics.recall_score(val_labels, val_preds, average='macro'),
              'acc': metrics.accuracy_score(val_labels, val_preds),
              'F1': metrics.f1_score(val_labels, val_preds, average="micro"),
              'train acc': metrics.accuracy_score(train_labels, train_preds)}

    return result


def evaluate_mlp(train_dataset, val_dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()


    train_labels = []
    train_embeddings = []

    val_labels = []
    val_embeddings = []


    for batch_idx, data in enumerate(train_dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        train_labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()

        adj_pooled_list = []
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        for i in range(len(pool_sizes)):
            ind = i+1
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append( Variable(data[adj_key].float(), requires_grad = False ).cuda())
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append(data[num_nodes_key])

            pool_matrices_list = []
            for j in range(args.num_pool_matrix):
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)

                pool_matrices_list.append(Variable( data[pool_adj_key].float(), requires_grad = False).cuda())

            pool_matrices_dic[i] = pool_matrices_list 

        pool_matrices_list = []
        if args.num_pool_final_matrix > 0:

            for j in range(args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                pool_matrices_list.append(Variable( data[pool_adj_key].float(), requires_grad = False).cuda())

            pool_matrices_dic[ind] = pool_matrices_list 


        feat =model( h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
        learned_feat = feat[0].cpu().data.numpy()
        train_embeddings.append(learned_feat)

    
    for batch_idx, data in enumerate(val_dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        val_labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()

        adj_pooled_list = []
        batch_num_nodes_list = []
        pool_matrices_dic = dict()
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        for i in range(len(pool_sizes)):
            ind = i+1
            adj_key = 'adj_pool_' + str(ind)
            adj_pooled_list.append( Variable(data[adj_key].float(), requires_grad = False ).cuda())
            num_nodes_key = 'num_nodes_' + str(ind)
            batch_num_nodes_list.append(data[num_nodes_key])

            pool_matrices_list = []
            for j in range(args.num_pool_matrix):
                pool_adj_key = 'pool_adj_' + str(i) + '_' + str(j)

                pool_matrices_list.append(Variable( data[pool_adj_key].float(), requires_grad = False).cuda())

            pool_matrices_dic[i] = pool_matrices_list 

        pool_matrices_list = []
        if args.num_pool_final_matrix > 0:

            for j in range(args.num_pool_final_matrix):
                pool_adj_key = 'pool_adj_' + str(ind) + '_' + str(j)

                pool_matrices_list.append(Variable( data[pool_adj_key].float(), requires_grad = False).cuda())

            pool_matrices_dic[ind] = pool_matrices_list 


        feat =model( h0, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic)
        learned_feat = feat[0].cpu().data.numpy()
        val_embeddings.append(learned_feat)

        
    val_size = len(val_embeddings)
        
    # Initialization of the final classifier
    in_feat = len(learned_feat)
    pred_layers = []
    pred_layers.append(nn.Linear(in_feat, 64).cuda())
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(64, 32).cuda())
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(32, 2).cuda())
    pred_model = nn.Sequential(*pred_layers)

    # The to-be-finetuned model and the optimizer
    optimizer_2 = torch.optim.Adam(pred_model.parameters(), lr=0.001)

    # Train on the train embeddings
    for i in range(len(train_embeddings)):
        pred_prob = pred_model(Variable(torch.Tensor(train_embeddings[i]), requires_grad=False).cuda())
        #print(pred_prob)
        pred_prob = torch.unsqueeze(pred_prob, 0)
        loss = F.cross_entropy(pred_prob, Variable(torch.LongTensor([int(train_labels[i])])).cuda())
        loss.backward()
        optimizer_2.step()
        optimizer_2.zero_grad()

    # Make predictions on the val/test embeddings
    correct = 0
    for i in range(len(val_embeddings)):
        pred_prob = pred_model(Variable(torch.Tensor(val_embeddings[i]), requires_grad=False).cuda())
        pred = pred_prob.argmax(dim=0)
        #print(pred)
        correct += pred.eq(torch.Tensor(val_labels[i]).cuda()).sum().item()

    
    result = {'acc': correct/len(val_embeddings)}


    return result


def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, mask_nodes = True):
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr= args.lr, weight_decay = args.weight_decay)
    iter = 0
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}


    # Triplet Sampler of train graphs
    tripletsampler_tr = TripletSampler(dataset)

    # Triplet Net
    TNet = tripletnet(model, args)

    # Triplet loss function
    criterion = torch.nn.MarginRankingLoss(margin = args.alpha)

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()

        tripletsampler_tr.shuffle()

        while not tripletsampler_tr.end():


            time1 = time.time()
            model.zero_grad()
            one_triplet = tripletsampler_tr.sampler()
            dist_p, dist_n, embed_a, embed_p, embed_n = TNet(one_triplet['anchor'], one_triplet['pos'], one_triplet['neg'])  
            target = torch.FloatTensor(dist_p.size()).fill_(-1)
            target = Variable(target).cuda()

            loss = criterion(dist_p, dist_n, target)
            loss.backward()

            time3 = time.time()

            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
        #print("done training")

        
        elapsed = time.time() - begin_time
        
        eval_time = time.time()
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=True,
            num_workers=args.num_workers)

        

        val_dataset_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=True,
            num_workers=args.num_workers)


        if val_dataset is not None:
            val_result = evaluate(train_dataset_loader, val_dataset_loader, model, args, name='Validation')
            val_acc = val_result['acc']

        test_dataset_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=True,
            num_workers=args.num_workers)


        if test_dataset is not None:
            test_result = evaluate(train_dataset_loader, test_dataset_loader, model, args, name='Test')
            test_acc = test_result['acc']
        
    return model, test_acc, val_acc



def prepare_data(graphs, graphs_list, args, test_graphs=None, max_nodes=0, seed=0):

    zip_list = list(zip(graphs,graphs_list))
    random.Random(seed).shuffle(zip_list)
    graphs, graphs_list = zip(*zip_list)

    test_graphs_list = []

    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
        train_graphs_list = graphs_list[:train_idx]
        val_graphs_list = graphs_list[train_idx: test_idx]
        test_graphs_list = graphs_list[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        train_graphs_list = graphs_list[:train_idx]    
        val_graphs = graphs[train_idx:]    
        val_graphs_list = graphs_list[train_idx: ] 


    test_dataset_loader = []
 
    train_dataset_sampler = GraphSampler(train_graphs,train_graphs_list, args.num_pool_matrix,args.num_pool_final_matrix,normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type, norm = args.norm)



    val_dataset_sampler = GraphSampler(val_graphs, val_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type, norm = args.norm)



    if len(test_graphs)>0:
        test_dataset_sampler = GraphSampler(test_graphs, test_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,normalize=False, max_num_nodes=max_nodes,
                features=args.feature_type, norm = args.norm)


    return train_dataset_sampler, val_dataset_sampler, test_dataset_sampler, \
            train_dataset_sampler.max_num_nodes, train_dataset_sampler.feat_dim


def benchmark_task_val(args, feat='node-label', pred_hidden_dims = [50]):

    all_vals = []
    all_tests = []

    data_out_dir = 'data/data_preprocessed/' + args.bmname + '/pool_sizes_' + args.pool_sizes 
    if args.normalize ==0:
        data_out_dir = data_out_dir + '_nor_' + str(args.normalize)


    data_out_dir = data_out_dir + '/'
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    graph_list_file_name = data_out_dir + 'graphs_list.p' 
    dataset_file_name = data_out_dir + 'dataset.p'

    # load the pre-processed datasets or do pre-processin
    if os.path.isfile(graph_list_file_name) and os.path.isfile(dataset_file_name):
        print('Files exist, reading from stored files....')
        #print('Reading file from', data_out_dir)
        print("dataset file name " + dataset_file_name)
        with open(dataset_file_name, 'rb') as f:
            graphs = pickle.load(f)
        with open(graph_list_file_name, 'rb') as f:
            graphs_list = pickle.load(f)
        #print('Data loaded!')
    else:
        print('No files exist, preprocessing datasets...')


        graphs = load_data.read_graphfile(args.datadir,args.bmname, max_nodes =args.max_nodes)
        #print('Data length before filtering: ', len(graphs))

        dataset_copy = graphs.copy()

        len_data = len(graphs)
        graphs_list = []
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        #print('pool_sizes: ', pool_sizes )

        for i in range(len_data):

            adj = nx.adjacency_matrix(dataset_copy[i])
            
            if adj.shape[0] < args.min_nodes or adj.shape[0]> args.max_nodes or adj.shape[0]!= dataset_copy[i].number_of_nodes():
                graphs.remove(dataset_copy[i])
                
            else:
                
                number_of_nodes = dataset_copy[i].number_of_nodes()

                coarsen_graph = gp(adj.todense().astype(float), pool_sizes)
                print("coarsen_graph")
                print(coarsen_graph)
                # if args.method == 'wave':
                result = coarsen_graph.coarsening_pooling(args.normalize)
                if result == 1:
                    graphs_list.append(coarsen_graph)
                else:
                    graphs.remove(dataset_copy[i])


        #print('Data length after filtering: ', len(graphs), len(graphs_list))
        #print('Dataset preprocessed, dumping....')
        with open(dataset_file_name, 'wb') as f:
            pickle.dump(graphs, f)
        with open(graph_list_file_name, 'wb') as f:
            pickle.dump(graphs_list, f)

        #print('Dataset dumped!')

    if args.feature_type == 'learnable':
        # input node features as learnable features
        node_features = torch.empty(args.max_nodes, args.input_dim, requires_grad = True).type(torch.FloatTensor).cuda()
        nn.init.normal_(node_features, std = 2)

        print('Using learnable features')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = node_features[u]

    
    elif feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']

    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])

    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(args.num_iterations):

        train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim = \
                        prepare_data(graphs, graphs_list, args, test_graphs = None,max_nodes=args.max_nodes, seed = i)
            
                       
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        model = encoders.WavePoolingGcnEncoder(max_num_nodes, input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                             args.num_pool_matrix, args.num_pool_final_matrix,pool_sizes =  pool_sizes, pred_hidden_dims = pred_hidden_dims,
                             concat = args.concat,bn=args.bn, dropout=args.dropout, mask = args.mask,args=args)

        _, test_acc, val_acc = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset)
        all_vals.append(val_acc)
        all_tests.append(test_acc)

    # Report performance on validation
    all_vals_mean = np.mean(all_vals, axis=0)
    all_vals_std = np.std(all_vals, axis=0)
    print("Validation performance " + str(args.num_iterations) + " times: " + str(all_vals_mean) + " with std " + str(all_vals_std))
    
    # Report performance on test
    all_tests_mean = np.mean(all_tests, axis=0)
    all_tests_std = np.std(all_tests, axis=0)
    print("Test performance " + str(args.num_iterations) + " times: " + str(all_tests_mean) + " with std " + str(all_tests_std))



def taxi_task_val(args, writer=None, feat='learnable', pred_hidden_dims = [50]):
    all_vals = []
    all_tests = []
    all_trains = []

    
    print(args.task + ' data')
    graphs = load_data.read_supplementarygraph(args.datadir, args.task, max_nodes=args.max_nodes)


    data_out_dir = 'taxi/data_preprocessed/' + args.bmname + '/pool_sizes_' + args.pool_sizes 
    if args.normalize ==0:
        data_out_dir = data_out_dir + '_nor_' + str(args.normalize)


    data_out_dir = data_out_dir + '/'
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)


    graph_list_file_name = data_out_dir + 'graphs_list.p' 
    dataset_file_name = data_out_dir + 'dataset.p'

        
    # load the pre-processed datasets or do pre-processin
    if os.path.isfile(graph_list_file_name) and os.path.isfile(dataset_file_name):
        print('Files exist, reading from stored files....')
        #print('Reading file from', data_out_dir)
        with open(dataset_file_name, 'rb') as f:
            graphs = pickle.load(f)
        with open(graph_list_file_name, 'rb') as f:
            graphs_list = pickle.load(f)
        #print('Data loaded!')
    else:
        print('No files exist, preprocessing datasets...')


        #graphs = load_data.read_graphfile(args.datadir,args.bmname, max_nodes =args.max_nodes)
        #print('Data length before filtering: ', len(graphs))

        dataset_copy = graphs.copy()

        len_data = len(graphs)
        graphs_list = []
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        #print('pool_sizes: ', pool_sizes )

        for i in range(len_data):

            adj = nx.adjacency_matrix(dataset_copy[i])
            
            # print('Adj shape',adj.shape)
            if adj.shape[0] < args.min_nodes or adj.shape[0]> args.max_nodes or adj.shape[0]!= dataset_copy[i].number_of_nodes():
                graphs.remove(dataset_copy[i])
                
            else:
                
                number_of_nodes = dataset_copy[i].number_of_nodes()

                coarsen_graph = gp(adj.todense().astype(float), pool_sizes)
                # if args.method == 'wave':
                result = coarsen_graph.coarsening_pooling(args.normalize)
                if result == 1:
                    graphs_list.append(coarsen_graph)
                else:
                    graphs.remove(dataset_copy[i])
                #graphs_list.append(coarsen_graph)


        #print('Data length after filtering: ', len(graphs), len(graphs_list))
        #print('Dataset preprocessed, dumping....')
        with open(dataset_file_name, 'wb') as f:
            pickle.dump(graphs, f)
        with open(graph_list_file_name, 'wb') as f:
            pickle.dump(graphs_list, f)

        #print('Dataset dumped!')


    if args.feature_type == 'learnable':
        # input node features as learnable features
        node_features = torch.empty(args.max_nodes, args.input_dim, requires_grad = True).type(torch.FloatTensor).cuda()
        nn.init.normal_(node_features, std = 2)

        print('Using learnable features')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = node_features[u]

    elif args.feature_type == 'node-labels':
        print('Using node labels as features')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = np.array(G.nodes[u]['label'])


    for i in range(args.num_iterations):
        train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim = \
                        prepare_data(graphs, graphs_list, args, test_graphs = None,max_nodes=args.max_nodes, seed = i)
            
                       
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        model = encoders.WavePoolingGcnEncoder(max_num_nodes, input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                             args.num_pool_matrix, args.num_pool_final_matrix,pool_sizes =  pool_sizes, pred_hidden_dims = pred_hidden_dims,
                             concat = args.concat,bn=args.bn, dropout=args.dropout, mask = args.mask,args=args)

        _, test_acc, val_acc = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset)
        all_vals.append(val_acc)
        all_tests.append(test_acc)

    # Report performance on validation
    all_vals_mean = np.mean(all_vals, axis=0)
    all_vals_std = np.std(all_vals, axis=0)
    print("Validation performance " + str(args.num_iterations) + " times: " + str(all_vals_mean) + " with std " + str(all_vals_std))
    
    # Report performance on test
    all_tests_mean = np.mean(all_tests, axis=0)
    all_tests_std = np.std(all_tests, axis=0)
    print("Test performance " + str(args.num_iterations) + " times: " + str(all_tests_mean) + " with std " + str(all_tests_std))


            
                 

def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    parser.add_argument('--task', dest='task',
            help='Name of the taxi dataset')

    # whether to have a validation sets
    parser.add_argument('--val', dest='val', type=bool,
                        help='Whether to have a validation set')

    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--iterations', dest='num_iterations', type=int,
            help='Number of iterations in benchmark_test_val')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
            help='Ratio of number of graphs testing set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
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
    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')

    # triplet loss 
    parser.add_argument('--alpha', dest='alpha', type=float,
            help='margin alpha in the triplet loss')
    parser.add_argument('--triplet', dest='triplet',
            help='how triplet is sampled')



    parser.add_argument('--pool_sizes', type = str,
                        help = 'pool_sizes', default = '10')
    parser.add_argument('--num_pool_matrix', type =int,
                        help = 'num_pooling_matrix', default = 1)
    parser.add_argument('--min_nodes', type = int,
                        help = 'min_nodes', default = 12)

    parser.add_argument('--weight_decay', type = float,
                        help = 'weight_decay', default = 0.0)
    parser.add_argument('--num_pool_final_matrix', type = int,
                        help = 'number of final pool matrix', default = 0)

    parser.add_argument('--normalize', type = int,
                        help = 'nomrlaized laplacian or not', default = 0)
    parser.add_argument('--pred_hidden', type = str,
                        help = 'pred_hidden', default = '50')

    
    parser.add_argument('--num_shuffle', type = int,
                        help = 'total num_shuffle', default = 10)
    parser.add_argument('--shuffle', type = int,
                        help = 'which shuffle, choose from 0 to 9', default=0)
    parser.add_argument('--concat', type = int,
                        help = 'whether concat', default = 1)
    parser.add_argument('--feat', type = str,
                        help = 'which feat to use', default = 'node-label')
    parser.add_argument('--mask', type = int,
                        help = 'mask or not', default = 1)
    parser.add_argument('--norm', type = str,
                        help = 'Norm for eigens', default = 'l2')

    parser.add_argument('--with_test', type = int,
                        help = 'with test or not', default = 0)
    parser.add_argument('--con_final', type = int,
                        help = 'con_final', default = 1)
    parser.add_argument('--cuda', dest = 'cuda',
                        help = 'cuda', default = 0)
    parser.set_defaults(max_nodes=1000,
			task='ppi',
                        feature_type='learnable',
                        datadir = 'data',
                        val=True,
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=10,
                        num_iterations=1,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=6,
                        num_gc_layers=2,
                        dropout=0.0,
                        alpha = 1.5,
                        num_pool_matrix=1,
                        num_pool_final_matrix=1,
                        shuffle=0,
                        num_shuffle=10,
                        weight_decay=0,
                        mask=1,
                        norm='l2',
                        pool_sizes=10,
                        with_test=1
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()
    seed = 1
    print(prog_args)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('bmname: ', prog_args.bmname)
    print('num_classes: ', prog_args.num_classes)
    # print('method: ', prog_args.method)
    print('batch_size: ', prog_args.batch_size)
    print('num_pool_matrix: ', prog_args.num_pool_matrix)
    print('num_pool_final_matrix: ', prog_args.num_pool_final_matrix)
    print('epochs: ', prog_args.num_epochs)
    print('learning rate: ', prog_args.lr)
    print('num of gc layers: ', prog_args.num_gc_layers)
    print('output_dim: ', prog_args.output_dim)
    print('hidden_dim: ', prog_args.hidden_dim)
    print('pred_hidden: ', prog_args.pred_hidden)
    # print('if_transpose: ', prog_args.if_transpose)
    print('dropout: ', prog_args.dropout)
    print('weight_decay: ', prog_args.weight_decay)
    print('shuffle: ', prog_args.shuffle)
    print('Using batch normalize: ', prog_args.bn)
    print('Using feat: ', prog_args.feat)
    print('Using mask: ', prog_args.mask)
    print('Norm for eigens: ', prog_args.norm)
    # print('Combine pooling results: ', prog_args.pool_m)
    print('With test: ', prog_args.with_test)

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)
   

   
    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]
    if prog_args.bmname is not None:
        
        if prog_args.task == 'ppi':
            benchmark_task_val(prog_args, pred_hidden_dims = pred_hidden_dims, feat = prog_args.feat)
        else:
            taxi_task_val(prog_args)



if __name__ == "__main__":
    main()
