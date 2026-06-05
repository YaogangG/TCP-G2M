import os
import dgl
import torch
import shutil
import random
import numpy as np
import yaml
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Correct spelling.

    # Optional: make nondeterministic operations raise errors, which is useful for debugging.
    torch.use_deterministic_algorithms(True)

    # cuBLAS determinism for CUDA 10.2+.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8".

    # DGL random seed.
    dgl.random.seed(seed)

def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass

def idx_split(idx, ratio, seed=0):
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]

    return idx1, idx2


def graph_split(idx_train, idx_val, idx_test, rate, seed):
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1:N1+N2]
    obs_idx_test = obs_idx_all[N1 + N2:]

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind

def get_evaluator(dataset):
    def evaluator(out, labels):
        pred = out.argmax(1)
        return pred.eq(labels).float().mean().item()

    return evaluator

def to_dgl(data):
    r"""Converts a :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
    object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The data object.

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
        >>> x = torch.randn(5, 3)
        >>> edge_attr = torch.randn(6, 2)
        >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes=5, num_edges=6,
            ndata_schemes={'x': Scheme(shape=(3,))}
            edata_schemes={'edge_attr': Scheme(shape=(2, ))})

        >>> data = HeteroData()
        >>> data['paper'].x = torch.randn(5, 3)
        >>> data['author'].x = torch.ones(5, 3)
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        >>> data['author', 'cites', 'paper'].edge_index = edge_index
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes={'author': 5, 'paper': 5},
            num_edges={('author', 'cites', 'paper'): 5},
            metagraph=[('author', 'paper', 'cites')])
    """
    import dgl

    from torch_geometric.data import Data, HeteroData

    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        elif 'adj' in data:
            row, col, _ = data.adj.coo()
        elif 'adj_t' in data:
            row, col, _ = data.adj_t.t().coo()
        else:
            row, col = [], []

        g = dgl.graph((row, col), num_nodes=data.num_nodes)

        for attr in data.node_attrs():
            if attr == 'x':
                g.ndata['feat'] = data[attr]
            elif attr == 'y':
                g.ndata['label'] = data[attr]
            else:
                g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, edge_store in data.edge_items():
            if edge_store.get('edge_index') is not None:
                row, col = edge_store.edge_index
            else:
                row, col, _ = edge_store['adj_t'].t().coo()

            data_dict[edge_type] = (row, col)

        g = dgl.heterograph(data_dict)

        for node_type, node_store in data.node_items():
            for attr, value in node_store.items():
                g.nodes[node_type].data[attr] = value

        for edge_type, edge_store in data.edge_items():
            for attr, value in edge_store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = np.ones(vals.shape[0]) * -1
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, seed=1234):
    """ randomly splits label into train/valid/test splits """
    train_idx, test_idx = train_test_split(np.arange(len(label)), train_size=train_prop, random_state=seed)
    val_idx, test_idx = train_test_split(test_idx, train_size=train_prop, random_state=seed)
    train_mask = torch.zeros(len(label), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(label), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(len(label), dtype=torch.bool)
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask

def get_train_config(config_path, model=None, dataset=None):
    """Load a YAML training configuration.

    The release configuration intentionally contains only representative default
    settings. Missing dataset/model entries fall back to the global section.
    """
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)

    specific_config = dict(full_config.get("global", {}))

    # Backward compatibility: get_train_config(config_path, dataset)
    if dataset is None:
        dataset = model
        model = None

    dataset_config = full_config.get(dataset, {}) if dataset is not None else {}
    if isinstance(dataset_config, dict):
        if model is not None and isinstance(dataset_config.get(model), dict):
            specific_config.update(dataset_config[model])
        else:
            # Allow flat dataset-level entries as a fallback.
            flat_items = {k: v for k, v in dataset_config.items() if not isinstance(v, dict)}
            specific_config.update(flat_items)

    return specific_config

import torch

def normalize_pair(A: torch.Tensor,
                   B: torch.Tensor,
                   method: str = "zscore",
                   center: bool = True,
                   symmetrize: bool = False,
                   eps: float = 1e-8):
    """
    Normalize A and B to the same scale to reduce scale effects in similarity computation.

    Args:
        A, B: tensors with the same shape, e.g., N x N Gram matrices.
        method:
            - "zscore": global (X - mean) / std.
            - "fro": X / ||X||_F, optionally after mean-centering when center=True.
            - "corr": correlation-style normalization D^{-1/2} X D^{-1/2}, mainly for Gram matrices.
        center: whether to mean-center before Frobenius normalization.
        symmetrize: if True, use (X + X^T) / 2. Recommended for Gram matrices.
        eps: numerical stability constant.
    Returns:
        A_norm, B_norm
    """
    assert A.shape == B.shape, "A and B must have the same shape"
    dev = A.device
    A = A.to(dev)
    B = B.to(dev)

    if symmetrize:
        A = 0.5 * (A + A.t())
        B = 0.5 * (B + B.t())

    if method.lower() == "zscore":
        Am = A.mean()
        As = A.std(unbiased=False).clamp_min(eps)
        Bm = B.mean()
        Bs = B.std(unbiased=False).clamp_min(eps)
        A_norm = (A - Am) / As
        B_norm = (B - Bm) / Bs

    elif method.lower() == "fro":
        if center:
            A = A - A.mean()
            B = B - B.mean()
        A_norm = A / (A.norm(p='fro').clamp_min(eps))
        B_norm = B / (B.norm(p='fro').clamp_min(eps))

    elif method.lower() == "corr":
        # Normalize Gram matrices to correlation-style matrices.
        diagA = A.diag().clamp_min(eps).sqrt()
        diagB = B.diag().clamp_min(eps).sqrt()
        A_norm = A / (diagA.unsqueeze(1) * diagA.unsqueeze(0) + eps)
        B_norm = B / (diagB.unsqueeze(1) * diagB.unsqueeze(0) + eps)
        # Numerical cleanup: set NaNs to zero and diagonal entries to one.
        A_norm = torch.nan_to_num(A_norm, nan=0.0)
        B_norm = torch.nan_to_num(B_norm, nan=0.0)
        A_norm.fill_diagonal_(1.0)
        B_norm.fill_diagonal_(1.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return A_norm, B_norm


def compute_dynamic_lambda(logits_t: torch.Tensor,
                           T: float = 4.0,
                           lam_min: float = 0.1,
                           lam_max: float = 0.9,
                           ema_state: dict = None,
                           ema_key: str = "lambda",
                           ema_beta: float = 0.9,
                           eps: float = 1e-12):
    """
    Dynamically adjust the KD mixing coefficient lambda using the teacher's temperature-scaled normalized entropy.
    - Lower uncertainty means stronger KD weight and a larger lambda.
    - Higher uncertainty means stronger supervised weight and a smaller lambda.
    Returns: lambda_scalar (float), u_mean (float in [0,1]).
    """

    with torch.no_grad():
        # Temperature-scaled probabilities.
        p_t = F.softmax(logits_t / T, dim=1)
        p_t = p_t.clamp_min(eps)

        # Per-sample entropy H(p) = -sum p log p.
        ent = -(p_t * p_t.log()).sum(dim=1)  # [N]

        # Normalize to [0, 1].
        num_classes = p_t.size(1)
        ent_norm = ent / math.log(num_classes + eps)  # [N]
        u_mean = ent_norm.mean()

        # Linear interpolation: lower uncertainty pushes lambda closer to lam_max.
        lam = lam_min + (lam_max - lam_min) * (1.0 - u_mean.item())

        # Optional EMA smoothing.
        if ema_state is not None:
            prev = ema_state.get(ema_key, lam)
            lam = ema_beta * prev + (1.0 - ema_beta) * lam
            ema_state[ema_key] = lam

        # Boundary protection.
        lam = float(max(lam_min, min(lam_max, lam)))

    return lam, float(u_mean)


def evidence_lambda_ds(
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    idx_l = None,
    T: float = 4.0,
    lam_min: float = 0.1,
    lam_max: float = 0.9,
    alpha_T: float = 1.0,
    eps: float = 1e-12,
):
    """
    Compute a batch-level lambda with a simple Dempster-Shafer evidence model:
      - Frame of discernment: Θ = {T (trust teacher), L (trust labels)}.
      - Evidence source 1: teacher confidence supports T.
      - Evidence source 2: teacher accuracy on labels distributes mass between T and L.

    Returns:
        lam_label: weight for supervised CE.
        info: intermediate quantities for debugging or plotting.
    """
    # Compute on the labeled subset when idx_l is provided, usually the training nodes.
    if idx_l is not None:
        logits_t = logits_t[idx_l]
        labels = labels[idx_l]

    # --------- Evidence source 1: teacher confidence ----------
    with torch.no_grad():
        p_t = F.softmax(logits_t / T, dim=1).clamp_min(eps)
        ent = -(p_t * p_t.log()).sum(dim=1)  # [N]
        num_classes = p_t.size(1)
        ent_norm = ent / math.log(num_classes + eps)  # Normalize to [0, 1].
        u_mean = ent_norm.mean().item()               # Average uncertainty.
        s_t = 1.0 - u_mean                             # Average confidence in [0, 1].

        # Evidence source 1 supports {T} and Θ only.
        m1_T = float(alpha_T * s_t)
        m1_T = max(0.0, min(1.0, m1_T))
        m1_L = 0.0
        m1_U = 1.0 - m1_T

        # --------- Evidence source 2: teacher accuracy on labels ----------
        pred_t = logits_t.argmax(dim=1)
        acc_t = pred_t.eq(labels).float().mean().item()

        # Here acc_t is treated as mass supporting {T}, while 1-acc_t supports {L}.
        m2_T = float(acc_t)
        m2_L = float(1.0 - acc_t)
        m2_U = 0.0

        # --------- Dempster combination ----------
        # Conflict mass, considering only the disjoint case T ∩ L = ∅.
        K_conf = m1_T * m2_L + m1_L * m2_T
        denom = max(eps, 1.0 - K_conf)

        # Unnormalized combined mass.
        mT_num = m1_T * m2_T + m1_T * m2_U + m1_U * m2_T
        mL_num = m1_L * m2_L + m1_L * m2_U + m1_U * m2_L
        mU_num = m1_U * m2_U

        mT = mT_num / denom
        mL = mL_num / denom
        mU = mU_num / denom

        # --------- Pignistic transform to decision probabilities ----------
        # BetP(T) = m(T) + 0.5 m(Θ)
        # BetP(L) = m(L) + 0.5 m(Θ)
        betT = mT + 0.5 * mU
        betL = mL + 0.5 * mU

        # Map to the [lam_min, lam_max] interval.
        lam = betL
        lam = lam_min + (lam_max - lam_min) * lam
        lam = float(max(lam_min, min(lam_max, lam)))

        info = {
            "m1": {"T": m1_T, "L": m1_L, "U": m1_U},
            "m2": {"T": m2_T, "L": m2_L, "U": m2_U},
            "m_comb": {"T": mT, "L": mL, "U": mU},
            "bet": {"T": betT, "L": betL},
            "u_mean": u_mean,
            "acc_t": acc_t,
            "lambda": lam,
        }

    return lam, info

def competition_lambda(
        logits_t: torch.Tensor,       # teacher logits (batch)
        labels: torch.Tensor,         # ground truth labels (batch)
        T: float = 4.0,
        gamma: float = 8.0,
        eps: float = 1e-12
    ):
    """
    Competition-based adjustment: dynamically adjust lambda between trusting the teacher and trusting labels.
    λ_i = sigmoid( γ ( confidence_i - error_i ) )

    - confidence_i: maximum probability after teacher softening.
    - error_i: teacher cross-entropy or margin with respect to the ground-truth label.
    """
    with torch.no_grad():

        # Soft teacher probability
        p_t = F.softmax(logits_t / T, dim=1).clamp_min(eps)

        # 1) Teacher confidence; larger values indicate stronger trust in the teacher.
        conf, _ = p_t.max(dim=1)      # [N] 0~1

        # 2) Teacher-label disagreement; larger values indicate stronger trust in labels.
        ce = F.nll_loss(p_t.log(), labels, reduction='none')  # [N]

        # Min-max normalize to [0, 1].
        ce_norm = (ce - ce.min()) / (ce.max() - ce.min() + eps)

        # Use a sigmoid function to determine lambda allocation.
        lam = torch.sigmoid(gamma * (conf - ce_norm))

        # Larger lambda means more trust in labels; smaller lambda means more trust in the teacher.
        return lam   # shape [N]


def _get_gt_mask(logits: torch.Tensor, labels: torch.Tensor):
    # logits: [N, C], labels: [N]
    return torch.zeros_like(logits).scatter_(1, labels.view(-1, 1), 1).bool()

def _get_other_mask(logits: torch.Tensor, labels: torch.Tensor):
    return ~_get_gt_mask(logits, labels)

def _cat_mask(p: torch.Tensor, gt_mask: torch.Tensor, other_mask: torch.Tensor):
    # Collapse the C-class distribution into two groups: [p(target), p(others)].
    p_t = (p * gt_mask.float()).sum(dim=1, keepdim=True)
    p_o = (p * other_mask.float()).sum(dim=1, keepdim=True)
    return torch.cat([p_t, p_o], dim=1)

def decoupled_kd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    labels: torch.Tensor,
    T: float = 4.0,
    alpha: float = 1.0,
    beta: float = 8.0,
) -> torch.Tensor:
    """
    DKD (Decoupled Knowledge Distillation), using a common reproduction-style implementation.
      loss = (alpha * TCKD + beta * NCKD) * T^2

    TCKD: KL over the binary distribution of target class versus non-target classes.
    NCKD: KL over non-target classes after masking the target logit with a large negative offset.
    """
    # masks
    gt_mask = _get_gt_mask(logits_s, labels)      # [N, C] bool
    other_mask = ~gt_mask                         # [N, C] bool

    # ----- TCKD -----
    p_s = F.softmax(logits_s / T, dim=1)
    p_t = F.softmax(logits_t / T, dim=1)

    p_s_2 = _cat_mask(p_s, gt_mask, other_mask)   # [N, 2]
    p_t_2 = _cat_mask(p_t, gt_mask, other_mask)   # [N, 2]

    tckd_loss = F.kl_div(
        (p_s_2 + 1e-12).log(),     # log-prob from student
        p_t_2.detach(),            # prob from teacher
        reduction="batchmean",
        log_target=False,
    )

    # ----- NCKD -----
    # Mask the target class by subtracting a sufficiently large constant so that its softmax probability is near zero.
    LARGE = 1000.0
    logits_s_nt = logits_s / T - LARGE * gt_mask.float()
    logits_t_nt = logits_t / T - LARGE * gt_mask.float()

    logp_s_nt = F.log_softmax(logits_s_nt, dim=1)
    p_t_nt = F.softmax(logits_t_nt, dim=1)

    nckd_loss = F.kl_div(
        logp_s_nt,                 # log-prob
        p_t_nt.detach(),           # prob
        reduction="batchmean",
        log_target=False,
    )

    return (alpha * tckd_loss + beta * nckd_loss) * (T * T)

