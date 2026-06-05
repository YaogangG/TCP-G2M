# ppr_precompute_optimized.py
import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import dgl
import dgl.function as fn

@torch.no_grad()
def _sym_norm_with_self_loop(g: dgl.DGLGraph) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """
    Return the graph with self-loops and symmetric normalized edge weights w_ij = 1/sqrt(d_i * d_j).
    Compatible with older DGL versions; does not rely on g.has_self_loop.
    """
    # For compatibility, remove existing self-loops and add exactly one self-loop per node.
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # Degree computation and symmetric normalization.
    deg = g.in_degrees().float().clamp_min(1)
    norm = torch.pow(deg, -0.5)
    # Use local_var to avoid mutating the original graph fields.
    gl = g.local_var()
    gl.ndata["_norm"] = norm
    gl.apply_edges(lambda e: {"_w": e.src["_norm"] * e.dst["_norm"]})
    # Return the graph without persistent edge fields together with the edge-weight tensor.
    ew = gl.edata["_w"]
    return g, ew


def _spmm(g: dgl.DGLGraph, edge_w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Sparse multiplication: Y = S * X, where S is the symmetric normalized adjacency with self-loops and edge_w stores edge weights.
    Use local_var to avoid persistent writes to graph data fields and to maintain compatibility with older DGL versions.
    """
    gl = g.local_var()
    gl.edata["_w"] = edge_w
    gl.srcdata["_h"] = X
    gl.update_all(fn.u_mul_e("_h", "_w", "_m"), fn.sum("_m", "_h"))
    return gl.dstdata["_h"]



@torch.no_grad()
def da_appnp_features(
    g: dgl.DGLGraph,
    X: torch.Tensor,
    alpha0: float = 0.2,
    beta: float = 0.5,              # Degree-adaptive strength; 0 reduces to a constant alpha.
    K: int = 10,
    tol: float = 0.0,               # Early-stopping threshold, e.g., 1e-4; 0 disables early stopping.
    mode: str = "concat",           # ["concat", "replace"]
    chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Degree-Adaptive APPNP:
      Z_{t+1} = (1-α_i) * S Z_t + α_i * X,  α_i = clip( alpha0 * (deg_i/avg_deg)^(-beta) ).
    Supports feature-dimension chunking and node-wise early stopping based on row-norm changes.
    """
    assert mode in ("concat", "replace")
    dev = X.device if device is None else device
    g = g.to(dev)
    X = X.to(dev)

    # Normalized adjacency.
    g, ew = _sym_norm_with_self_loop(g)
    ew = ew.to(dev)

    # Adaptive alpha_i.
    deg = g.in_degrees().float().to(dev).clamp_min(1.0)
    avg_deg = deg.mean()
    alpha_i = alpha0 * torch.pow(deg / avg_deg, -beta)
    alpha_i = torch.clamp(alpha_i, 0.02, 0.9).view(-1, 1)   # Avoid extreme values.
    one_minus_alpha = 1.0 - alpha_i

    def _prop(Z0: torch.Tensor) -> torch.Tensor:
        Z = Z0.clone()
        active = torch.ones(Z.size(0), dtype=torch.bool, device=dev)  # Early-stopping mask.
        for _ in range(K):
            Z_next = _spmm(g, ew, Z)
            Z_next = one_minus_alpha * Z_next + alpha_i * Z0
            if tol > 0:
                diff = (Z_next - Z).norm(p=2, dim=1)
                newly_stable = diff < tol
                active = active & (~newly_stable)
                # Once all nodes are stable, stop unnecessary updates.
                if not active.any():
                    Z = Z_next
                    break
            Z = Z_next
        return Z

    d = X.size(1)
    if chunk_size is None or d <= chunk_size:
        X_ppr = _prop(X)
    else:
        outs = []
        for s in range(0, d, chunk_size):
            e = min(d, s + chunk_size)
            outs.append(_prop(X[:, s:e]))
        X_ppr = torch.cat(outs, dim=1)

    X_new = torch.cat([X, X_ppr], dim=1) if mode == "concat" else X_ppr
    return X_new, X_ppr


@torch.no_grad()
def tc_ppr_features(
    g: dgl.DGLGraph,
    X: torch.Tensor,
    logits_teacher: torch.Tensor,   # [N, C] teacher outputs, either logits or probabilities.
    beta: float = 0.2,              # Teacher-guidance strength.
    alpha: float = 0.2,
    K: int = 10,
    mode: str = "concat",
    chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    proj_dim: Optional[int] = None, # Optional projection from teacher probabilities to the feature space.
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher-Calibrated PPR:
      X0 = (1-β) X + β * P_t W
      Z_{t+1} = (1-α) S Z_t + α X0
    """
    assert mode in ("concat", "replace")
    dev = X.device if device is None else device
    g = g.to(dev)
    X = X.to(dev)
    P = torch.softmax(logits_teacher.to(dev), dim=1)

    # Project teacher probabilities to the feature space.
    N, d = X.shape
    C = P.shape[1]
    if proj_dim is None:
        # Simple linear mapping to d dimensions.
        W = torch.empty(C, d, device=dev)
        torch.nn.init.xavier_uniform_(W)
        Tproj = P @ W
    else:
        W = torch.empty(C, proj_dim, device=dev)
        torch.nn.init.xavier_uniform_(W)
        Tproj = P @ W
        # If proj_dim differs from d, map the projection to d dimensions.
        if proj_dim != d:
            W2 = torch.empty(proj_dim, d, device=dev)
            torch.nn.init.xavier_uniform_(W2)
            Tproj = Tproj @ W2

    X0 = (1 - beta) * X + beta * Tproj

    # Normalized adjacency.
    g, ew = _sym_norm_with_self_loop(g)
    ew = ew.to(dev)

    def _prop(Z0: torch.Tensor) -> torch.Tensor:
        Z = Z0.clone()
        for _ in range(K):
            Z = _spmm(g, ew, Z)
            Z = (1 - alpha) * Z + alpha * Z0
        return Z

    if chunk_size is None or d <= chunk_size:
        X_ppr = _prop(X0)
    else:
        outs = []
        for s in range(0, d, chunk_size):
            e = min(d, s + chunk_size)
            outs.append(_prop(X0[:, s:e]))
        X_ppr = torch.cat(outs, dim=1)

    X_new = torch.cat([X, X_ppr], dim=1) if mode == "concat" else X_ppr
    return X_new, X_ppr


@torch.no_grad()
def maybe_cache_optimized(
    cache_path: str,
    tensor_pair_fn,
    *args, **kwargs
):
    """
    Generic caching wrapper: save (X_new, X_ppr) to disk and move loaded tensors back to the input tensor device.
    """
    os.makedirs(Path(cache_path).parent, exist_ok=True)
    # Use the first tensor argument as the target device for loading cached tensors.
    first_tensor = None
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor):
            first_tensor = a
            break
    dev = first_tensor.device if first_tensor is not None else torch.device("cpu")

    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location=dev)
        return data["x_new"].to(dev), data["x_ppr"].to(dev)

    x_new, x_ppr = tensor_pair_fn(*args, **kwargs)
    torch.save({"x_new": x_new.cpu(), "x_ppr": x_ppr.cpu()}, cache_path)
    return x_new, x_ppr


@torch.no_grad()
def compute_role_features(
    g: dgl.DGLGraph,
    device: Optional[torch.device] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Structural role-prior features computed without labels:
      - deg_z        : z-score of node degree.
      - log_deg_z    : z-score of log(1+degree).
      - nbr_deg_z    : z-score of average neighbor degree.

    Returns: a tensor with shape [N, 3].
    """
    dev = device if device is not None else (g.device if hasattr(g, "device") else torch.device("cpu"))
    g = g.to(dev)

    # Node degree. Since this is a role feature, using in-degrees with self-loops is acceptable.
    deg = g.in_degrees().float().clamp_min(0.0)  # [N]
    N = deg.shape[0]

    # Average neighbor degree computed with one message-passing step.
    gl = g.local_var()
    gl.ndata["_deg"] = deg
    gl.update_all(fn.copy_u("_deg", "m"), fn.mean("m", "_nbr_deg"))
    nbr_deg = gl.ndata["_nbr_deg"]  # [N]

    def zscore(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std(unbiased=False).clamp_min(eps)
        return (x - mean) / std

    deg_z = zscore(deg)
    log_deg_z = zscore(torch.log1p(deg))
    nbr_deg_z = zscore(nbr_deg)

    role_feat = torch.stack([deg_z, log_deg_z, nbr_deg_z], dim=1)  # [N, 3]
    return role_feat