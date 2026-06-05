import argparse
import warnings
from pathlib import Path
import torch.optim as optim
from models import *
from dataloader import *
from train_and_eval import *
import sys
import os
from ppr_precompute import maybe_cache_ppr
from ppr_precompute_optimized import da_appnp_features, tc_ppr_features, maybe_cache_optimized, compute_role_features


warnings.filterwarnings("ignore", category=Warning)

def assign_obs_feats(feats: torch.Tensor, idx_obs: torch.Tensor, obs_feats_new: torch.Tensor):
    """
    Write the updated features from an observed subgraph back to the full-graph feature tensor:
    - If the feature dimension is unchanged (replace mode), overwrite the corresponding rows directly.
    - If the feature dimension increases (concat mode), allocate an expanded full-graph feature tensor.
    """
    N, d = feats.shape
    _, d_new = obs_feats_new.shape
    device = torch.device(param['device'])

    if d_new == d:
        # The feature dimension is unchanged; overwrite the observed rows directly.
        feats = feats.clone()
        feats[idx_obs] = obs_feats_new.to(device)
        return feats

    # The feature dimension increases (e.g., concat mode); expand the full-graph features.
    feats_new = feats.new_zeros((N, d_new))
    # Keep the original features in the first d dimensions.
    feats_new[:, :d] = feats
    # Fill observed nodes with the updated features, including appended dimensions if any.
    feats_new[idx_obs] = obs_feats_new.to(device)
    return feats_new

def main(param, out_t=None):
    model = Model(param).to(device)
    if param['distill_mode'] == 0:
        optimizer = optim.AdamW(model.parameters(), lr=param['learning_rate'], weight_decay=param["weight_decay"])
    elif param["distill_mode"] == 1:
        optimizer = optim.AdamW(model.parameters(), lr=param['learning_rate'], weight_decay=param["weight_decay"])
    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(param["dataset"])

    if param['distill_mode'] == 0:
        out, test_acc, test_val, test_best, tea_feats = train_teacher(param, model, g, feats, labels, indices, criterion_l, evaluator, optimizer)

        if os.path.exists(out_t_dir) == 0:
            os.makedirs(out_t_dir)

        np.savez(out_t_dir.joinpath("out.npz"), out.cpu().detach())
        torch.save(tea_feats, out_t_dir.joinpath('feat.pt'))

        return test_acc, test_val, test_best
    else:
        out_t = load_out_t(out_t_dir).to(device)

        test_acc, test_val, test_best = train_student(param, model, g, feats, feats_tea, labels, out_t, indices, criterion_l, criterion_t, evaluator, optimizer)

        return test_acc, test_val, test_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="chameleon")
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--split_id", type=float, default=1, help="heterophilic datasets split of dgl")
    parser.add_argument("--dropout_s", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=4)
    parser.add_argument("--lamb", type=float, default=0.5)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=1500)
    parser.add_argument("--feat_t_dim", type=int, default=0)

    parser.add_argument("--distill_mode", type=int, default=0, help="0 only train teacher, 1 load teacher and train student")
    parser.add_argument("--exp_setting", type=str, default="tran", help="[tran, ind]")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--split_rate", type=float, default=0.2)
    parser.add_argument("--model_config_path", default=".conf.yaml", help="Path to model configeration")

    parser.add_argument("--feat_distill", type=int, default=0, help="0: feat_distill off; 1: on")

    # PPR-related arguments.
    parser.add_argument("--use_ppr", type=int, default=0, help="0: off, 1: on")
    parser.add_argument("--ppr_mode", type=str, default="concat", choices=["concat", "replace"])
    parser.add_argument("--ppr_alpha", type=float, default=0.2)
    parser.add_argument("--ppr_k", type=int, default=0)
    parser.add_argument("--ppr_chunk", type=int, default=0, help="0 means no chunk; otherwise feature-dim chunk size")

    parser.add_argument("--use_tc_ppr", type=int, default=1, help="0: off, 1: on")

    parser.add_argument("--tc_alpha", type=float, default=0.2)

    parser.add_argument("--tc_ind_gamma", type=float, default=2.0,
                        help="inductive: attenuation strength, s = max(smin, mean(exposure)^gamma)")
    parser.add_argument("--tc_ind_smin", type=float, default=0.2,
                        help="inductive: minimum tc-ppr strength (avoid turning off)")
    parser.add_argument("--tc_ind_clip", type=float, default=3.0)
    parser.add_argument("--tc_ind_smax", type=float, default=1e-4,
                        help="inductive: maximum tc-ppr strength (do not turn off, but cap the influence)")

    parser.add_argument("--use_da_appnp", type=int, default=0, help="0: off, 1: on")
    parser.add_argument("--use_role_prior", type=int, default=0, help="0: off, 1: on")

    parser.add_argument("--kd_type", type=str, default="kl",
                        help="kl: vanilla KD (KLDiv); dkd: decoupled KD")

    parser.add_argument(
        "--lambda_mode",
        type=str,
        default="pcgrad",
        help="fixed | evidence | pcgrad",
    )

    parser.add_argument(
        "--log_grad_conflict",
        type=int,
        default=0,
        help="0: off, 1: log gradient conflict stats between CE and KD (cos<0)",
    )
    parser.add_argument(
        "--grad_conflict_report_every",
        type=int,
        default=200,
        help="report conflict stats every N epochs (only when log_grad_conflict=1)",
    )

    parser.add_argument("--ablation_mode", type=int, default=0,
                        help="0: HindG2M; 1: valinna MLPs; 2: GLNN;")

    args = parser.parse_args()

    param = {}
    if args.model_config_path is not None:
        args.model_config_path = '.conf.yaml'
        param = get_train_config(args.exp_setting + args.model_config_path, args.student, args.dataset)
    param = dict(args.__dict__, **param)

    device = torch.device(param['device'])
    set_seed(param['seed'])

    g, labels = load_data(args.dataset, param)
    g, labels = g.to(device), labels.to(device)

    results = []
    for i in range(5 if param['dataset'] == 'penn94' else 10):
        out_dir = './out/'
        if param['dataset'] in ['cora','citeseer','pubmed','a-photo','a-computer', 'cs','phy']:
            param['seed'] = i
            set_seed(param['seed'])
            out_t_dir = Path.cwd().joinpath(out_dir, args.dataset, args.teacher, param['exp_setting'], str(param['seed']))
        else:
            param['split_id'] = i
            out_t_dir = Path.cwd().joinpath(out_dir, args.dataset, args.teacher, param['exp_setting'], str(param['split_id']))

        idx_train, idx_val, idx_test = get_mask(g, param)
        if args.exp_setting == "tran":
            # In the transductive setting, indices contain train/validation/test splits only.
            indices = (idx_train, idx_val, idx_test)
            idx_obs = None
            idx_test_ind = None
        elif args.exp_setting == "ind":
            # In the inductive setting, graph_split returns observed-split indices and held-out inductive test indices.
            indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, i)
            obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

        # Original node features.
        feats = g.ndata["feat"].to(device)

        # Teacher features are loaded for interface compatibility; they are not modified here.
        if param['distill_mode'] == 1:
            feats_tea = torch.load(out_t_dir.joinpath('feat.pt'), map_location=device)

        # ============== PPR / DA-APPNP / TC-PPR feature preprocessing ==============
        # Use param['exp_setting'] to determine whether the current setting is transductive or inductive.
        exp_setting = param['exp_setting']

        # ---------- Option A: standard PPR preprocessing ----------
        if param.get("use_ppr", 0) == 1:
            cache_dir = Path("./out") / args.dataset / "__ppr__"
            base_tag = f"a{param.get('ppr_alpha', 0.2)}_k{param.get('ppr_k', 10)}_{param.get('ppr_mode', 'concat')}"
            # Transductive runs can reuse one cache; inductive runs need split-specific caches because subgraphs differ.
            if exp_setting == "ind":
                tag = f"{base_tag}_ind_split{param['split_id']}"
            else:
                tag = base_tag
            cache_path = cache_dir / f"ppr_{tag}.pt"

            if exp_setting == "tran":
                # Transductive setting: run PPR on the full graph.
                x_new, _ = maybe_cache_ppr(
                    str(cache_path),
                    g,
                    feats,
                    alpha=float(param.get("ppr_alpha", 0.2)),
                    k=int(param.get("ppr_k", 10)),
                    mode=str(param.get("ppr_mode", "concat")),
                    chunk_size=int(param.get("ppr_chunk", 0)) or None,
                    device=device,
                )
                feats = x_new.to(device)
            else:
                # Inductive setting: run PPR only on the observed subgraph to avoid using held-out test structure.
                g_obs = g.subgraph(idx_obs).to(device)            # observed subgraph
                obs_feats = feats[idx_obs]                        # features of observed nodes only
                x_obs_new, _ = maybe_cache_ppr(
                    str(cache_path),
                    g_obs,
                    obs_feats,
                    alpha=float(param.get("ppr_alpha", 0.2)),
                    k=int(param.get("ppr_k", 10)),
                    mode=str(param.get("ppr_mode", "concat")),
                    chunk_size=int(param.get("ppr_chunk", 0)) or None,
                    device=device,
                )
                # Write back only to idx_obs; held-out inductive nodes keep original features to avoid structural leakage.
                feats = assign_obs_feats(feats, idx_obs, x_obs_new)

        # ---------- Option B: Teacher-Calibrated PPR (TC-PPR) ----------
        # TC-PPR is used only during student training because teacher logits are unavailable during teacher training.
        if param.get("use_tc_ppr", 0) == 1 and param['distill_mode'] == 1:
            # Load teacher logits on the full graph.
            out_t = load_out_t(out_t_dir)        # [N, C]
            logits_t_full = out_t.to(device)

            cache_dir = Path("./out") / args.dataset / "__ppr_opt__"
            base_tag = f"tc_b{param.get('tc_beta', 0.2)}_a{param.get('tc_alpha', 0.2)}_K{param.get('tc_K', 10)}_{param.get('ppr_mode', 'concat')}"
            if exp_setting == "ind":
                if param['dataset'] in ['cora','citeseer','pubmed','a-photo','a-computer']:
                    tag = f"{base_tag}_ind_split{param['seed']}"
                else:
                    tag = f"{base_tag}_ind_split{param['split_id']}"
            else:
                tag = base_tag
            cache_path = cache_dir / f"{tag}.pt"

            if exp_setting == "tran":
                # Transductive setting: run TC-PPR on the full graph using all teacher logits.
                feats, _ = maybe_cache_optimized(
                    str(cache_path),
                    tc_ppr_features,
                    g, feats, logits_t_full,
                    beta=float(param.get("tc_beta", 0.2)),
                    alpha=float(param.get("tc_alpha", 0.2)),
                    K=int(param.get("tc_K", 10)),
                    mode=str(param.get("ppr_mode", "concat")),
                    chunk_size=int(param.get("ppr_chunk", 0)) or None,
                    device=device,
                    proj_dim=int(param.get("tc_proj_dim", 0)) or None,
                )
            else:
                idx_obs = idx_obs.to(device)
                g_obs = g.subgraph(idx_obs).to(device)

                # 1) Store the observed features before TC-PPR; clone to avoid later in-place side effects.
                obs_feats_before = feats[idx_obs].clone()
                logits_obs = logits_t_full[idx_obs]

                # 2) Compute TC-PPR features. In the inductive setting, replace mode avoids dimensional shifts and is better suited for residual mixing.
                obs_feats_new, _ = maybe_cache_optimized(
                    str(cache_path),
                    tc_ppr_features,
                    g_obs, obs_feats_before, logits_obs,
                    beta=float(param.get("tc_beta", 0.2)),
                    alpha=float(param.get("tc_alpha", 0.2)),
                    K=int(param.get("tc_K", 10)),
                    mode="replace",  # Use replace mode in the inductive setting to avoid extra distribution shift from concatenation.
                    chunk_size=int(param.get("ppr_chunk", 0)) or None,
                    device=device,
                    proj_dim=int(param.get("tc_proj_dim", 0)) or None,
                )

                # 3) Estimate observed structural exposure in the inductive setting and derive the TC-PPR attenuation coefficient s.
                deg_full = g.out_degrees().float().to(device).clamp_min(1.0)  # [N]
                deg_obs = g_obs.out_degrees().float().to(device)  # [N_obs]
                exposure_obs = (deg_obs / deg_full[idx_obs]).clamp(0.0, 1.0)  # [N_obs]

                gamma = float(param.get("tc_ind_gamma", 2.0))
                s_min = float(param.get("tc_ind_smin", 0.2))
                s = float(torch.clamp(exposure_obs.mean().pow(gamma), min=s_min, max=1.0))
                s_max = float(param.get("tc_ind_smax", 0.05))  # A conservative default is recommended.
                s = min(s, s_max)

                # 4) In the inductive setting, attenuate TC-PPR instead of disabling it, and mix it back with the original observed features.
                # --- Residual-Norm tc-ppr injection (inductive) ---
                delta = obs_feats_new - obs_feats_before  # tc-ppr residual

                # LayerNorm over feature dim (no learnable params, very stable)
                delta = (delta - delta.mean(dim=1, keepdim=True)) / (delta.std(dim=1, keepdim=True) + 1e-6)

                # final: keep original feature as anchor, inject normalized residual
                clip_c = float(param.get("tc_ind_clip", 3.0))
                obs_feats_new = obs_feats_before + (s * delta).clamp(-clip_c, clip_c)

                # 5) Write back to the full graph; held-out inductive nodes keep original features to avoid leakage.
                feats = assign_obs_feats(feats, idx_obs, obs_feats_new)

        # ---------- Structural role prior ----------
        if param.get("use_role_prior", 0) == 1:
            # Transductive setting: compute role features on the full graph.
            if exp_setting == "tran":
                role_feat = compute_role_features(g, device=device)  # [N, 3]
            else:
                # Inductive setting: compute role features only on the observed subgraph to avoid using held-out test structure.
                g_obs = g.subgraph(idx_obs).to(device)
                role_obs = compute_role_features(g_obs, device=device)  # [|idx_obs|, 3]

                # Initialize full-graph role channels with zeros, then fill the observed subgraph entries.
                N = g.num_nodes()
                dr = role_obs.shape[1]
                role_feat = feats.new_zeros((N, dr))
                role_feat[idx_obs] = role_obs

            # Append role features to the original node features.
            feats = torch.cat([feats, role_feat.to(device)], dim=1)

        # Update model input and output dimensions.
        param["feat_dim"] = feats.shape[1]
        param["label_dim"] = labels.int().max().item() + 1

        # Enter the teacher or student training stage.
        test_acc, test_val, test_best = main(param)
        results.append(round(test_val, 4))


    print(f'Student({args.student}):{results}')
    print(f'Mean:{np.mean(results):.4f},STD:{np.std(results):.4f}')
    print(' Configuration: ', param)







