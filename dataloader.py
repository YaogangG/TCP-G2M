import dgl
import torch
import numpy as np
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, LINKXDataset
from utils import *
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import DataLoader, Data, ClusterData, ClusterLoader

homo_datasets = ['cora', 'citeseer', 'pubmed', 'a-photo', 'a-computer', 'cs', 'phy', 'cora-full']
hetero_datasets = ['texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel', 'actor', 'penn94', 'arxiv-year', 'arating', 'romanempire']


def load_data(dataset, param):

    if dataset == 'cora':
        g = dgl.data.CoraGraphDataset('./data')[0]
    elif dataset == 'citeseer':
        g = dgl.data.CiteseerGraphDataset('./data')[0]
    elif dataset == 'pubmed':
        g = dgl.data.PubmedGraphDataset('./data')[0]
    elif dataset == 'a-photo':
        g = dgl.data.AmazonCoBuyPhotoDataset('./data')[0]
    elif dataset == 'cs':
        g = dgl.data.CoauthorCSDataset(raw_dir='./data')[0]
    elif dataset == 'phy':
        g = dgl.data.CoauthorPhysicsDataset('./data')[0]
    elif dataset == 'cora-full':
        g = dgl.data.CoraFullDataset('./data')[0]
    elif dataset == 'texas':
        g = dgl.data.TexasDataset('./data')[0]
    elif dataset == 'cornell':
        g = dgl.data.CornellDataset('./data')[0]
    elif dataset == 'wisconsin':
        g = dgl.data.WisconsinDataset('./data')[0]
    elif dataset == 'squirrel':
        g = dgl.data.SquirrelDataset('./data')[0]
    elif dataset == 'actor':
        g = dgl.data.ActorDataset('./data')[0]
    elif dataset == 'chameleon':
        g = dgl.data.ChameleonDataset('./data')[0]
    elif dataset == 'arating':
        g = dgl.data.AmazonRatingsDataset('./data')[0]
    elif dataset == 'romanempire':
        g = dgl.data.RomanEmpireDataset('./data')[0]
    elif dataset == 'penn94':
        graph = LINKXDataset(root='./data', name='penn94')[0]
        g = to_dgl(graph)
    elif dataset == 'arxiv-year':
        data = NodePropPredDataset(name='ogbn-arxiv',root='./data')
        print(data)
        print(data.num_classes)
        print(data.graph.keys())
        data.name = 'arxiv-year'
        split_index = data.get_idx_split()
        label = even_quantile_labels(
            data.graph['node_year'].flatten(), 5, verbose=False
        )
        data.label = torch.as_tensor(label).reshape(-1, 1)
        graph = Data(x=torch.from_numpy(data.graph['node_feat']).float(),
                     edge_index=torch.from_numpy(data.graph['edge_index']).long(),
                     y=data.label.long().squeeze(1)
                     )
        g = to_dgl(graph)
        data.num_features = data.graph['node_feat'].shape[1]
        data.num_classes = data.label.max() + 1
        print(data.name)

    labels = g.ndata['label']
    g = dgl.remove_self_loop(g)
    features = g.ndata['feat']

    g = dgl.add_self_loop(g)

    return g, labels


def get_mask(g, param):
    if param['dataset'] in hetero_datasets:
        if param['dataset'] == 'arxiv-year':
            train_mask, val_mask, test_mask = rand_train_test_idx(g.ndata['label'], seed=2)
            train_mask = train_mask
            val_mask = val_mask
            test_mask = test_mask
        else:
            train_mask = g.ndata['train_mask'][:, param['split_id']]
            val_mask = g.ndata['val_mask'][:, param['split_id']]
            test_mask = g.ndata['test_mask'][:, param['split_id']]
    elif param['dataset'] in ['cora', 'citeseer', 'pubmed']:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    else:
        labels = g.ndata['label']
        n_class = int(labels.max().item() + 1)
        nrange = torch.arange(labels.shape[0]).to(device)
        train_mask = torch.zeros(labels.shape[0], dtype=bool).to(device)

        for y in range(n_class):
            label_mask = (g.ndata['label'] == y)
            train_mask[nrange[label_mask][torch.randperm(label_mask.sum())[:20]]] = True

        val_mask = ~train_mask
        val_mask[nrange[val_mask][torch.randperm(val_mask.sum())[500:]]] = False
        test_mask = ~(train_mask | val_mask)
        test_mask[nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]] = False

    return train_mask.nonzero()[:, 0], val_mask.nonzero()[:, 0], test_mask.nonzero()[:, 0]


def load_out_t(out_t_dir):
    return torch.from_numpy(np.load(out_t_dir.joinpath("out.npz"))["arr_0"])
