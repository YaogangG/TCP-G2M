import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv, SAGEConv


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout_ratio):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
            h_list.append(h)
        return h, h_list


class GCN(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout_ratio, activation):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(in_dim, out_dim, activation=activation))
        else:
            self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(GraphConv(hidden_dim, out_dim, activation=activation))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.norms[l](h)
                h = self.dropout(h)
            h_list.append(h)
        return h, h_list


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout_ratio, activation,
                 num_heads=8, attn_drop=0.3, negative_slope=0.2, residual=False):
        super().__init__()
        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        heads = ([num_heads] * num_layers) + [1]

        from dgl.nn import GATConv
        self.layers.append(GATConv(in_dim, hidden_dim, heads[0], dropout_ratio, attn_drop,
                                   negative_slope, False, activation))
        for l in range(1, num_layers - 1):
            self.layers.append(GATConv(hidden_dim * heads[l - 1], hidden_dim, heads[l],
                                       dropout_ratio, attn_drop, negative_slope, residual, activation))
        self.layers.append(GATConv(hidden_dim * heads[-2], out_dim, heads[-1],
                                   dropout_ratio, attn_drop, negative_slope, residual, None))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
            h_list.append(h)
        return h, h_list


class GraphSAGE(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout_ratio, activation,
                 aggregator_type='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.activation = activation

        if num_layers == 1:
            self.layers.append(SAGEConv(in_dim, out_dim, aggregator_type=aggregator_type))
        else:
            self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type=aggregator_type))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_type))
            self.layers.append(SAGEConv(hidden_dim, out_dim, aggregator_type=aggregator_type))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
            h_list.append(h)
        return h, h_list


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.model_name = param['teacher'] if param['distill_mode'] == 0 else param['student']

        if "MLP" in self.model_name:
            self.encoder = MLP(
                num_layers=param['num_layers'],
                in_dim=param['feat_dim'],
                hidden_dim=param['hidden_dim'],
                out_dim=param['label_dim'],
                dropout_ratio=param.get('dropout_t', param.get('dropout_s', 0.5)),
            ).to(device)
        elif "GCN" in self.model_name:
            self.encoder = GCN(
                num_layers=param['num_layers'],
                in_dim=param['feat_dim'],
                hidden_dim=param['hidden_dim'],
                out_dim=param['label_dim'],
                dropout_ratio=param['dropout_s'],
                activation=F.relu,
            ).to(device)
        elif "GAT" in self.model_name:
            self.encoder = GAT(
                num_layers=param['num_layers'],
                in_dim=param['feat_dim'],
                hidden_dim=param['hidden_dim'],
                out_dim=param['label_dim'],
                dropout_ratio=param['dropout_s'],
                activation=F.relu,
                num_heads=param.get('num_heads', 8),
            ).to(device)
        elif "SAGE" in self.model_name:
            self.encoder = GraphSAGE(
                num_layers=param['num_layers'],
                in_dim=param['feat_dim'],
                hidden_dim=param['hidden_dim'],
                out_dim=param['label_dim'],
                dropout_ratio=param['dropout_s'],
                activation=F.relu,
                aggregator_type=param.get('sage_aggregator', 'mean'),
            ).to(device)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def forward(self, g, feats):
        if "MLP" in self.model_name:
            out, h_list = self.encoder(feats)
        else:
            out, h_list = self.encoder(g, feats)
        return out, h_list
