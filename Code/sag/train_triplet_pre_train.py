from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import torch
from torch.autograd import Variable
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import argparse
import os
from network import Net
from tripletnet import tripletnet
from triplet_sampler import TripletSampler



# Test & Evaluation code
def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

# evaluate using kNN
def evaluate(train_loader, val_loader, model, device):
    model.eval()
    correct = 0.
    loss = 0.

    train_embeddings = []
    train_labels = []
    val_embeddings = []
    val_labels = []

    for data in train_loader:
        data = data.to(device)
        out = model(data)
        learned_feat = out[0].cpu().data.numpy()
        train_embeddings.append(learned_feat)
        train_labels.append(data.y.long().numpy())


    for data in val_loader:
        data = data.to(device)
        out = model(data)
        learned_feat = out[0].cpu().data.numpy()
        val_embeddings.append(learned_feat)
        val_labels.append(data.y.long().numpy())

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

# evaluate using MLP classifier
def evaluate_mlp(train_loader, val_loader, model, device):
    model.eval()
    correct = 0.
    loss = 0.

    train_embeddings = []
    train_labels = []
    val_embeddings = []
    val_labels = []

    for data in train_loader:
        data = data.to(device)
        out = model(data)
        learned_feat = out[0].cpu().data.numpy()
        train_embeddings.append(learned_feat)
        train_labels.append(data.y.long().numpy())


    for data in val_loader:
        data = data.to(device)
        out = model(data)
        learned_feat = out[0].cpu().data.numpy()
        val_embeddings.append(learned_feat)
        val_labels.append(data.y.long().numpy())

    # Initialization of the final classifier
    in_feat = len(learned_feat)
    pred_layers = []
    pred_layers.append(nn.Linear(in_feat, 64).to(device))
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(64, 32).to(device))
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(32, 2).to(device))
    pred_model = nn.Sequential(*pred_layers)

    # The to-be-finetuned model and the optimizer
    optimizer_2 = torch.optim.Adam(pred_model.parameters(), lr=0.001)

    # Train on the train embeddings
    for i in range(len(train_embeddings)):
        pred_prob = pred_model(Variable(torch.Tensor(train_embeddings[i]), requires_grad=False))
        # print(pred_prob)
        pred_prob = torch.unsqueeze(pred_prob, 0)
        loss = F.cross_entropy(pred_prob, Variable(torch.LongTensor([int(train_labels[i])])))
        loss.backward()
        optimizer_2.step()
        optimizer_2.zero_grad()

    # Make predictions on the val/test embeddings
    correct = 0
    for i in range(len(val_embeddings)):
        pred_prob = pred_model(Variable(torch.Tensor(val_embeddings[i]), requires_grad=False))
        pred = pred_prob.argmax(dim=0)
        # print(pred)
        correct += pred.eq(torch.Tensor(val_labels[i])).sum().item()



    result = {'acc': correct /len(val_embeddings)}

    return result


# Hyperparameters & Setup

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD')
parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
parser.add_argument('--epochs', type=int, default=100000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv')
parser.add_argument('--num_features', type=int, default=64, help='Dimension of input features')
parser.add_argument('--final_dim', type=int, default=64, help='Dimension of final embeddings')
parser.add_argument('--alpha', type=float, default=1.5, help='Margin in the triplet loss')


args = parser.parse_args()
device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = 'cuda:0'
dataset = TUDataset(os.path.join('data',args.dataset), name=args.dataset)
num_classes = dataset.num_classes
num_features = dataset.num_features

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)

# Train
min_loss = 1e10
patience = 0

val_accs = []
test_accs = []

for iter in range(args.iterations):
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model = Net(num_features, args.nhid, args.final_dim, args.pooling_ratio, args.dropout_ratio).to(device)

    # Triplet Net
    TNet = tripletnet(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Triplet Sampler of train graphs
    tripletsampler_tr = TripletSampler(training_set)

    # Triplet loss function
    criterion = torch.nn.MarginRankingLoss(margin=args.alpha)

    # FIRST STAGE: TRAINING WITH TRIPLET LOSS
    for epoch in range(args.epochs):

        tripletsampler_tr.shuffle()

        model.train()
        while not tripletsampler_tr.end():
            one_triplet = tripletsampler_tr.sampler()

            # data = data.to(args.device)
            dist_p, dist_n, embed_a, embed_p, embed_n = TNet(one_triplet['anchor'].to(device),
                                                             one_triplet['pos'].to(device),
                                                             one_triplet['neg'].to(device))
            target = torch.FloatTensor(dist_p.size()).fill_(-1)
            target = Variable(target).to(device)

            loss = criterion(dist_p, dist_n, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # SECOND STAGE: FINE-TUNING

    # Triplet Sampler of train graphs
    tripletsampler_tr = TripletSampler(training_set)
    # Train & Validation Loader
    train_loader = DataLoader(training_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=True)

    # Initialization of the final classifier
    in_feat = args.final_dim
    pred_layers = []
    pred_layers.append(nn.Linear(in_feat, 64).to(device))
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(64, 32).to(device))
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(32, 2).to(device))
    pred_model = nn.Sequential(*pred_layers)

    # The to-be-finetuned model and the optimizer
    model.map2_model = pred_model
    optimizer_2 = torch.optim.Adam(model.parameters(), lr=0.001)

    # Accuracy before fine-tuning
    result = evaluate(training_set, training_set, model, name='Train', max_num_examples=100)
    train_acc = result['acc']
    print("train acc before post-train : " + str(train_acc))

    # Fine-tuning
    for epoch in range(args.epochs):
        posttrain_loss = 0.0
        iter = 0

        model.train()

        for i, data in enumerate(train_loader):
            # zero the parameter gradients
            optimizer_2.zero_grad()
            data = data.to(device)
            out = model(data)

            loss = F.nll_loss(out, data.y)
            # print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer_2.step()
            optimizer_2.zero_grad()

        val_acc, val_loss = test(model, val_loader)
        # print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), 'latest.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > patience:
            break

        result = evaluate(training_set, training_set, model, name='Train', max_num_examples=100)
        train_acc = result['acc']
        print("train acc at epoch " + str(epoch) + " post-train: " + str(train_acc))

    # Evaluate the results
    test_result = evaluate(training_set, test_set, model, name='Test', max_num_examples=100)
    test_accs.append(test_result['acc'])

    val_result = evaluate(training_set, validation_set, model, name='Validation', max_num_examples=100)
    val_accs.append(val_result['acc'])

# Print avg and std
# Validation avg. and std
print('Validation accuracy of ' + str(args.iterations) + ' times is: ' + str(np.mean(val_accs)) + ' with std: ' + str(np.std(val_accs)))
# Test avg. and std
print('Test accuracy of ' + str(args.iterations) + ' times is: ' + str(np.mean(test_accs)) + ' with std: ' + str(np.std(test_accs)))