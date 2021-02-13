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
import shutil
import time

import cross_val
import encoders
import encoders_GAT
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data

# evaluate based on classification result (on validation and test sets)
def evaluate(dataset, model, args, iteration, name='Validation', max_num_examples=None):
    model.eval()

    
    embeddings = []
    
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
        output, ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        embeddings.append(output.cpu().data.numpy())
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    #print(name, " accuracy:", result['acc'])

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



def train(dataset, model, args, iteration, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True):
    writer_batch_idx = [0, 3, 6, 9]
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []



    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        iter = 0
        model.train()
        #print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            
            label = Variable(data['label'].long(), requires_grad=False).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

            output, ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss.item()
            #if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            total_time += elapsed
            
        avg_loss /= iter
        print("training loss at epoch " + str(epoch) + " is: " + str(avg_loss))

        
    result = evaluate(dataset, model, args, iteration, name='Train', max_num_examples=100)
    train_acc = result['acc']
    train_accs.append(result['acc'])
    train_epochs.append(epoch)
    if val_dataset is not None:
        val_result = evaluate(val_dataset, model, args, iteration, name='Validation')
        val_acc = val_result['acc']
    if test_dataset is not None:
        test_result = evaluate(test_dataset, model, args, iteration, name='Test')
        test_acc = test_result['acc']
          
    return train_acc, test_acc, val_acc

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
        val_graphs = graphs[train_idx:]
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


def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    all_tests = []

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
                cross_val.split_train_val_normal(graphs, args, i, max_nodes=args.max_nodes, feat = 'node-label')
        
        if args.method == 'soft-assign':
            #print('Method: diff-pool')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim, final_dim='number_classes').cuda()
        
        elif args.method == 'base':
            print('Method: sage')
            model = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args, final_dim='number_classes').cuda()

        elif args.method == 'GAT':
            print('Method: gat')
            model = encoders_GAT.DGATEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                     num_layers = args.num_gc_layers, args=args, final_dim = 'number_classes').cuda()

        else:
            print('No valid method specified. Default: soft-assign.')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim, final_dim = 'number_classes').cuda()


        if args.val == True:
            train_acc, test_acc, val_acc = train(train_dataset, model, args, i, val_dataset=val_dataset, test_dataset=test_dataset, writer=writer)
            all_vals.append(val_acc)
            all_tests.append(test_acc)
        else:
            train_acc, test_acc, val_acc = train(train_dataset, model, args, i, val_dataset=None, test_dataset=test_dataset, writer=writer)
            all_tests.append(test_acc)

    # Report performance on validation
    if args.val == True:
        all_vals_mean = np.mean(all_vals, axis=0)
        all_vals_std = np.std(all_vals, axis=0)
        print("Validation performance " + str(args.num_iterations) + " times: " + str(all_vals_mean) + " with std " + str(all_vals_std))
    
    # Report performance on test
    all_tests_mean = np.mean(all_tests, axis=0)
    all_tests_std = np.std(all_tests, axis=0)
    print("Test performance " + str(args.num_iterations) + " times: " + str(all_tests_mean) + " with std " + str(all_tests_std))


def taxi_task_val(args, writer=None, feat='learnable'):
    all_vals = []
    all_tests = []
    all_trains = []

    if args.datadir == 'attack':
        print('attack data')
        graphs = load_data.read_attackgraph(args.datadir, args.top, max_nodes=args.max_nodes)
    elif args.datadir == 'darpa' or args.datadir == 'taxi':
        print(args.task + ' data')
        graphs = load_data.read_supplementarygraph(args.datadir, args.task, max_nodes=args.max_nodes)
        

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
        train_dataset, test_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
                cross_val.split_train_val_normal(graphs, args, i, max_nodes=args.max_nodes, feat = 'learnable')
        
        if args.method == 'soft-assign':
            print('Method: diff-pool')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim, final_dim = 'number_classes').cuda()
        
        elif args.method == 'base':
            print('Method: sage')
            model = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args, final_dim='number_classes').cuda()

        elif args.method == 'GAT':
            print('Method: gat')
            model = encoders_GAT.DGATEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                     num_layers = args.num_gc_layers, args=args, final_dim = 'number_classes').cuda()

        else:
            print('No valid method specified. Default: soft-assign.')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim, final_dim = 'number_classes').cuda()

            
        if args.val == True:
            train_acc, test_acc, val_acc = train(train_dataset, model, args, i, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)
            all_vals.append(val_acc)
            all_tests.append(test_acc)
            all_trains.append(train_acc)
        else:
            train_acc, test_acc, val_acc = train(train_dataset, model, args, i, val_dataset=None, test_dataset=test_dataset,
            writer=writer)
            all_tests.append(test_acc)
            all_trains.append(train_acc)
    
    # Report performance on train
    all_trains_mean = np.mean(all_trains, axis=0)
    all_trains_std = np.std(all_trains, axis=0)
    print("Training performance " + str(args.num_iterations) + " times: " + str(all_trains_mean) + " with std " + str(all_trains_std))

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
    parser.add_argument('--max_nodes', dest='max_nodes', type=int,
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

    parser.set_defaults(task='ppi',
                        datadir='data',
                        logdir='log',
                        val=True,
                        dataset='syn1v2',
                        max_nodes=1000,
                        cuda='1',
                        feature_type='learnable',
                        lr=0.001,
                        clip=2.0,
                        batch_size=1,
                        num_epochs=10,
                        num_iterations=1,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=2,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
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
            taxi_task_val(prog_args)
    elif prog_args.pkl_fname is not None:
        print("there is nothing to do")
    elif prog_args.dataset is not None:
        print("there is nothing to do")


if __name__ == "__main__":
    main()

