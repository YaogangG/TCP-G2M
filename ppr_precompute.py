# ppr_precompute.py
import torch
import dgl
from dgl.nn import APPNPConv
from typing import Optional

@torch.no_grad()
def appnp_ppr_features(
    g: dgl.DGLGraph,
    x: torch.Tensor,
    alpha: float = 0.2,
    k: int = 10,
    mode: str = "concat",   # ["concat", "replace"]
    chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """
    Feature preprocessing with APPNP-based approximate PPR diffusion. Supports large graphs on GPU or CPU.
    - g: DGLGraph. Self-loops are recommended; if unsure, add them with dgl.add_self_loop.
    - x: [N, d] original node features located on the target device.
    - alpha: teleport probability, commonly 0.1 to 0.2.
    - k: number of propagation steps, commonly 5 to 10.
    - mode: "concat" appends diffused features to the original features; "replace" uses diffused features only.
    - chunk_size: optional feature-dimension chunk size for memory-constrained settings.
    - device: computation device; defaults to x.device.

    Returns:
      x_new: preprocessed features; x_ppr: diffused features for reuse or optional caching.
    """
    assert mode in ("concat", "replace")
    dev = x.device if device is None else device
    appnp = APPNPConv(k=k, alpha=alpha).to(dev)
    g = g.to(dev)

    def _prop(feat: torch.Tensor) -> torch.Tensor:
        # APPNPConv internally performs K iterations: Z_{t+1}=(1-a)S Z_t + a Z0.
        return appnp(g, feat)

    d = x.size(1)
    if chunk_size is None or d <= chunk_size:
        x_ppr = _prop(x)
    else:
        # Propagate feature chunks to reduce peak memory when d is large.
        outs = []
        for s in range(0, d, chunk_size):
            e = min(d, s + chunk_size)
            outs.append(_prop(x[:, s:e]))
        x_ppr = torch.cat(outs, dim=1)

    if mode == "concat":
        x_new = torch.cat([x, x_ppr], dim=1)
    else:  # "replace"
        x_new = x_ppr
    return x_new, x_ppr


@torch.no_grad()
def maybe_cache_ppr(
    cache_path: str,
    g: dgl.DGLGraph,
    x: torch.Tensor,
    alpha: float,
    k: int,
    mode: str,
    chunk_size: Optional[int],
    device: Optional[torch.device],
):
    """
    Load cached features if available; otherwise compute and save them.
    Returns:(x_new, x_ppr)
    """
    import os
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location=x.device)
        return data["x_new"], data["x_ppr"]

    x_new, x_ppr = appnp_ppr_features(
        g, x, alpha=alpha, k=k, mode=mode, chunk_size=chunk_size, device=device
    )
    torch.save(
        {"x_new": x_new.cpu(), "x_ppr": x_ppr.cpu(),
         "alpha": alpha, "k": k, "mode": mode},
        cache_path,
    )
    return x_new, x_ppr
