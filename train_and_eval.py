import dgl
import copy
import torch

from utils import *
import torch.nn.functional as F
import math
import time
import gc

import torch
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

class TriadMonitor:
    """No-op diagnostics hook for the anonymized reviewer release."""
    @classmethod
    def get(cls, param):
        return cls()
    def observe_pre(self, *args, **kwargs):
        return None
    def observe_post(self, *args, **kwargs):
        return None
    def curvature_after_step(self, *args, **kwargs):
        return None
    def report_and_reset(self, *args, **kwargs):
        return None



device = torch.device('cuda:0')

def _flat_grad(grads):
    """
    Flatten a list/tuple of parameter gradients into one 1-D vector.
    Skip None gradients.
    """
    vec = []
    for g in grads:
        if g is None:
            continue
        vec.append(g.reshape(-1))
    if len(vec) == 0:
        return None
    return torch.cat(vec, dim=0)


# Training for teacher GNNs
def train(model, data, feats, labels, criterion, optimizer, idx_train):
    model.train()

    logits, feats_map = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return logits, loss.item()


def train_mini_batch(
    model,
    feats,
    feats_tea,
    labels,
    out_t_all,
    idx_l,
    criterion_l,
    criterion_t,
    optimizer,
    param,
    exposure=None,   # Optional inductive exposure gate.
):
    """
    PCGrad with inductive-aware KD.

    Key changes for the inductive setting:
      - Transductive: KL-KD is computed on idx_l only, providing a strong supervised distillation signal.
      - Inductive: KL-KD is computed over all currently visible input nodes, which provides a smoothing prior and reduces mismatch.
      - Inductive: optional exposure gate or ind_kd_scale can further suppress boundary mismatch.
    """
    import torch
    import torch.nn.functional as F

    model.train()

    # ---------- forward ----------
    logits, feats_stu = model(None, feats)
    out = logits.log_softmax(dim=1)

    # ---------- CE on labeled ----------
    loss_l = criterion_l(out[idx_l], labels[idx_l])

    # ---------- KD ----------
    tau = float(param.get("tau", 1.0))
    kd_type = param.get("kd_type", "kl")  # "kl" or "dkd"
    exp_setting = param.get("exp_setting", "tran")  # "tran" or "ind"

    if kd_type == "dgkd":
        logits_s_l = logits[idx_l]
        logits_t_l = out_t_all[idx_l].to(logits_s_l.device)
        labels_l = labels[idx_l]

        alpha = float(param.get("dgkd_alpha", 15.0))
        beta = float(param.get("dgkd_beta", 3.0))

        loss_t = dgkd_loss(
            logits_mlp=logits_s_l,
            logits_gnn=logits_t_l,
            target=labels_l,
            alpha=alpha,
            beta=beta,
            temperature=tau,
        )

    elif kd_type == "dkd":
        # Decoupled KD loss on labeled nodes only.
        logits_s_l = logits[idx_l]
        logits_t_l = out_t_all[idx_l].to(logits_s_l.device)
        labels_l = labels[idx_l]
        alpha = float(param.get("dkd_alpha", 1.0))
        beta = float(param.get("dkd_beta", 8.0))
        loss_t = decoupled_kd_loss(logits_s_l, logits_t_l, labels_l, T=tau, alpha=alpha, beta=beta)

    else:
        # KL KD
        student_logp_T = (logits / tau).log_softmax(dim=1)
        teacher_logp_T = (out_t_all / tau).log_softmax(dim=1).to(student_logp_T.device)

        if exp_setting == "ind":
            # ==============================
            # Core inductive choice: compute KL over all currently visible nodes.
            # In the inductive branch, feats/out_t_all are usually restricted to the observed subgraph, so held-out test nodes are not leaked.
            # ==============================
            loss_t = (tau * tau) * criterion_t(student_logp_T, teacher_logp_T)

            # Optional 1: globally down-weight KD in the inductive setting; default 1.0 has no effect.
            ind_kd_scale = float(param.get("ind_kd_scale", 1.0))
            if ind_kd_scale != 1.0:
                loss_t = loss_t * ind_kd_scale

            # Optional 2: exposure gate, which requires passing exposure from train_student.
            # Let only structurally stable labeled nodes activate KD; otherwise KD is suppressed toward zero.
            if exposure is not None and int(param.get("use_exp_gate", 0)) == 1:
                thr = float(param.get("exp_threshold", 0.8))
                exp_l = exposure[idx_l]  # [B]
                kd_ratio = (exp_l >= thr).float().mean()  # 0~1
                loss_t = loss_t * kd_ratio

        else:
            # Transductive setting: keep labeled-only KL as a stronger supervised distillation signal.
            loss_t = (tau * tau) * criterion_t(
                student_logp_T[idx_l],
                teacher_logp_T[idx_l],
            )

    # Feature distillation is not used in this release.
    loss_f = torch.tensor(0.0, device=logits.device, requires_grad=False)

    # ---------- combine / optimize ----------
    ablation_mode = int(param.get("ablation_mode", 0))
    lambda_mode = param.get("lambda_mode", "fixed")
    lamb = float(param.get("lamb", 0.5))

    optimizer.zero_grad()

    # ablation_mode == 1: only CE
    if ablation_mode == 1:
        loss = loss_l
        loss.backward()
        optimizer.step()

        return loss_l.item(), loss_t.item(), loss_f.item()

    # ---------- PCGrad multi-objective ----------
    if lambda_mode == "pcgrad":
        pcgrad_normalize = int(param.get("pcgrad_normalize", 1))         # 1: L2 normalize
        pcgrad_symmetric = int(param.get("pcgrad_symmetric", 1))         # 1: symmetric projection
        pcgrad_restore_scale = int(param.get("pcgrad_restore_scale", 1)) # 1: restore avg scale

        loss_ce = loss_l
        loss_kd = loss_t

        params = [p for p in model.parameters() if p.requires_grad]

        grads_ce = torch.autograd.grad(
            loss_ce, params, retain_graph=True, allow_unused=True
        )
        grads_kd = torch.autograd.grad(
            loss_kd, params, retain_graph=False, allow_unused=True
        )

        # ---- Tragic Triad diagnostics (PCGrad) ----
        triad = TriadMonitor.get(param)
        triad.observe_pre(grads_ce, grads_kd, params, cos=None, loss_total_detached=(loss_l + loss_t).detach())
        final_grads = []
        post_grads_ce = []
        post_grads_kd = []
        for g_ce, g_kd in zip(grads_ce, grads_kd):
            if g_ce is None and g_kd is None:
                final_grads.append(None)
                continue
            if g_ce is None:
                final_grads.append(g_kd)
                continue
            if g_kd is None:
                final_grads.append(g_ce)
                continue

            g1 = g_ce
            g2 = g_kd

            g1f = g1.view(-1)
            g2f = g2.view(-1)

            if pcgrad_normalize:
                n1 = g1f.norm() + 1e-12
                n2 = g2f.norm() + 1e-12
                g1n = g1 / n1
                g2n = g2 / n2
            else:
                n1 = g1f.norm() + 1e-12
                n2 = g2f.norm() + 1e-12
                g1n = g1
                g2n = g2

            dot = torch.dot(g1n.view(-1), g2n.view(-1))

            # projected grads for post-statistics (default: no conflict => unchanged)
            g1_post = g1n
            g2_post = g2n

            if dot < 0:
                denom_22 = g2n.view(-1).pow(2).sum() + 1e-12
                g1p = g1n - dot / denom_22 * g2n
                g1_post = g1p

                if pcgrad_symmetric:
                    dot2 = torch.dot(g2n.view(-1), g1n.view(-1))
                    denom_11 = g1n.view(-1).pow(2).sum() + 1e-12
                    g2p = g2n - dot2 / denom_11 * g1n
                    g2_post = g2p
                    g = 0.5 * (g1p + g2p)
                else:
                    # asymmetric: only project CE gradient, keep KD as is
                    g2_post = g2n
                    g = 0.5 * (g1p + g2n)
            else:
                g = 0.5 * (g1n + g2n)

            # restore update scale if needed
            if pcgrad_normalize and pcgrad_restore_scale:
                g = g * (0.5 * (n1 + n2))

            # Save post-projection gradients in the original scale for global cosine computation.
            if pcgrad_normalize:
                # g1n = g1/n1, so scale back to comparable magnitude with original grads
                post_grads_ce.append(g1_post * n1)
                post_grads_kd.append(g2_post * n2)
            else:
                post_grads_ce.append(g1_post)
                post_grads_kd.append(g2_post)

            final_grads.append(g)
        # ---- Tragic Triad diagnostics (post-projection) ----
        triad.observe_post(post_grads_ce, post_grads_kd)


        for p, g in zip(params, final_grads):
            if g is not None:
                p.grad = g

        optimizer.step()

        # ---- curvature (must be after optimizer.step) ----
        if int(param.get("log_grad_curvature", 0)) == 1:
            triad.curvature_after_step(
                model=model,
                feats=feats,
                labels=labels,
                out_t_all=out_t_all,
                idx_l=idx_l,
                criterion_l=criterion_l,
                criterion_t=criterion_t,
                tau=float(param.get("tau", 4.0)),
                params=params
            )

        return loss_l.item(), loss_t.item(), loss_f.item()

    # ---------- fallback: weighted sum ----------
    if ablation_mode in (0, 2):
        loss = lamb * loss_l + (1.0 - lamb) * loss_t
    else:
        loss = loss_l

    loss.backward()
    optimizer.step()

    return loss_l.item(), loss_t.item(), loss_f.item()




def meta_kd_epoch(
    param,
    model,
    feats_enh,      # (N_task, d_new) enhanced features, e.g., PPR/role features.
    feats_raw,      # (N_task, d_raw) raw features.
    labels,         # (N_task,)
    out_t_all,      # (N_task, C) teacher logits on these nodes.
    idx_train,      # (N_train,)
    criterion_l,
    criterion_t,
    optimizer,
):
    """
    Main training variant used when ablation_mode = 0:
      - inner loop: use enhanced features with CE + KD on the support split.
      - outer loop: use pseudo-inductive features with CE + KD on the query split.
      - then update the meta-model with a first-order MAML-style update.
    """
    device = feats_enh.device
    model.train()

    N_train = idx_train.size(0)
    if N_train == 0:
        return 0.0, 0.0, 0.0

    # -------- support/query split --------
    meta_split_rate = float(param.get("meta_split_rate", 0.7))
    meta_split_rate = min(max(meta_split_rate, 0.1), 0.9)

    perm = torch.randperm(N_train, device=device)
    cut = int(meta_split_rate * N_train)
    if cut <= 0:
        cut = max(1, N_train // 2)
    if cut >= N_train:
        cut = N_train - 1

    idx_train = idx_train.to(device)
    support = idx_train[perm[:cut]]
    query   = idx_train[perm[cut:]]

    # -------- inner model and optimizer --------
    inner_model = copy.deepcopy(model).to(device)
    inner_model.train()

    inner_lr    = float(param.get("meta_inner_lr", param.get("lr", 0.01)))
    inner_steps = int(param.get("meta_inner_steps", 1))
    inner_opt   = torch.optim.SGD(inner_model.parameters(), lr=inner_lr)

    tau  = float(param.get("tau", 1.0))
    lamb = float(param["lamb"])

    # ======= Inner loop: CE + KD on the support split =======
    loss_l_s_last = 0.0
    loss_t_s_last = 0.0

    for _ in range(inner_steps):
        logits_s, _ = inner_model(None, feats_enh)
        out_s = logits_s.log_softmax(dim=1)

        loss_ce_s = criterion_l(out_s[support], labels[support])

        student_logp_T_s = (logits_s / tau).log_softmax(dim=1)
        teacher_logp_T   = (out_t_all / tau).log_softmax(dim=1).to(device)
        loss_kd_s = (tau * tau) * criterion_t(
            student_logp_T_s[support], teacher_logp_T[support]
        )

        loss_inner = loss_ce_s * lamb + loss_kd_s * (1.0 - lamb)

        inner_opt.zero_grad()
        loss_inner.backward()
        inner_opt.step()

        loss_l_s_last = loss_ce_s.item()
        loss_t_s_last = loss_kd_s.item()

    # ======= Outer loop: pseudo-inductive features on the query split =======
    N_task, d_new = feats_enh.shape
    _,     d_raw = feats_raw.shape

    feats_q = feats_enh.clone()
    feats_q[query, :d_raw] = feats_raw[query]
    if d_new > d_raw:
        feats_q[query, d_raw:] = 0.0

    logits_q, _ = inner_model(None, feats_q)
    out_q = logits_q.log_softmax(dim=1)

    loss_ce_q = criterion_l(out_q[query], labels[query])

    student_logp_T_q = (logits_q / tau).log_softmax(dim=1)
    teacher_logp_T   = (out_t_all / tau).log_softmax(dim=1).to(device)
    loss_kd_q = (tau * tau) * criterion_t(
        student_logp_T_q[query], teacher_logp_T[query]
    )

    loss_outer = loss_ce_q * lamb + loss_kd_q * (1.0 - lamb)

    # ======= First-order MAML: update the meta-model using inner-model gradients =======
    optimizer.zero_grad()
    loss_outer.backward()

    for p_meta, p_inner in zip(model.parameters(), inner_model.parameters()):
        if p_inner.grad is not None:
            if p_meta.grad is None:
                p_meta.grad = p_inner.grad.detach().clone()
            else:
                p_meta.grad.copy_(p_inner.grad.detach())

    optimizer.step()

    loss_l_total = loss_l_s_last + loss_ce_q.item()
    loss_t_total = loss_t_s_last + loss_kd_q.item()
    loss_f_total = 0.0

    return float(loss_l_total), float(loss_t_total), float(loss_f_total)


# Testing for teacher GNNs
def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval):
    model.eval()

    with torch.no_grad():
        logits, tea_feats = model.forward(data, feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out[idx_eval], labels[idx_eval])
        acc = evaluator(out[idx_eval], labels[idx_eval])

    return logits, loss.item(), acc


# Training for student MLPs
def evaluate_mini_batch(model, feats, labels, criterion, evaluator):
    model.eval()

    with torch.no_grad():
        logits, _ = model.forward(None, feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, labels)
        acc = evaluator(out, labels)

    return loss.item(), acc


def train_teacher(param, model, g, feats, labels, indices, criterion, evaluator, optimizer):
    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_g = g.subgraph(idx_obs).to(device)

    g.to(device)

    es = 0
    val_best = 0
    test_best = 0
    test_val = 0

    for epoch in range(1, param['max_epoch'] + 1):
        if param['exp_setting'] == 'tran':
            out, loss = train(model, g, feats, labels, criterion, optimizer, idx_train)
            _, train_loss, train_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_train)
            _, _, val_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_val)
            _, _, test_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_test)
        else:
            out, loss = train(model, obs_g, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
            _, train_loss, train_acc = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator,
                                                obs_idx_train)
            _, _, val_acc = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
            _, _, test_acc = evaluate(model, g, feats, labels, criterion, evaluator, idx_test_ind)

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1

        if es == 50:
            print("Early stopping!")
            break

        if epoch % 1 == 0:
            print("\033[0;30;46m [{}] CLA: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m"
                  .format(epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_val, test_best))

    model.load_state_dict(state)
    inference_time = 9999
    if param['exp_setting'] == 'tran':
        start_time = time.time()
        out, _, _ = evaluate(model, g, feats, labels, criterion, evaluator, idx_val)
        end_time = time.time()
    else:
        start_time = time.time()
        obs_out, _, _ = evaluate(model, obs_g, obs_feats, obs_labels, criterion, evaluator, obs_idx_val)
        out, _, _ = evaluate(model, g, feats, labels, criterion, evaluator, idx_test_ind)
        out[idx_obs] = obs_out
        end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time * 1000} ms")
    print("Test Val:{:.4f}".format(test_val))

    # Run one full-graph forward pass to obtain teacher logits and features.
    model.eval()
    with torch.no_grad():
        out, tea_feat = model(g, feats)

    return out, test_acc, test_val, test_best, tea_feat


def train_student(
    param,
    model,
    g,
    feats,
    feats_tea,   # Kept for interface compatibility.
    labels,
    out_t_all,
    indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
):
    """
    Main student-training function:

    ablation_mode = 0 and use_meta = 1: meta-style KD.
    Other cases:
      ablation_mode = 0 and use_meta = 0: standard logit distillation with CE + KD.
      ablation_mode = 1: plain MLP trained with CE only.
      ablation_mode = 2: standard logit distillation with CE + KD, without meta training.
    """
    device = feats.device
    ablation_mode = int(param.get("ablation_mode", 0))
    use_meta = int(param.get("use_meta", 1))   # New hyperparameter: 1 enables meta distillation, 0 disables it.

    if param["exp_setting"] == "tran":
        idx_train, idx_val, idx_test = indices

        # Meta-training task: full graph.
        task_feats_enh  = feats                          # (N, d_new)
        task_labels     = labels                         # (N,)
        task_out_t_all  = out_t_all                      # (N, C)
        task_feats_raw  = g.ndata["feat"].to(device)     # (N, d_raw)
        task_idx_train  = idx_train

    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

        # Observed-subgraph tensors for meta or standard training.
        obs_feats_enh  = feats[idx_obs]                  # (N_obs, d_new)
        obs_labels     = labels[idx_obs]                 # (N_obs,)
        obs_out_t      = out_t_all[idx_obs]              # (N_obs, C)
        obs_feats_raw  = g.ndata["feat"].to(device)[idx_obs]  # (N_obs, d_raw)

        # Meta-training task: observed subgraph.
        task_feats_enh  = obs_feats_enh
        task_labels     = obs_labels
        task_out_t_all  = obs_out_t
        task_feats_raw  = obs_feats_raw
        task_idx_train  = obs_idx_train             # [0, N_obs)

    es = 0
    val_best = 0.0
    test_val = 0.0
    test_best = 0.0
    state = copy.deepcopy(model.state_dict())

    max_epoch = int(param["max_epoch"])

    for epoch in range(1, max_epoch + 1):
        # ----------------- One training step -----------------
        if ablation_mode == 0 and use_meta == 1:
            # Meta-KD branch; used only when ablation_mode=0 and use_meta=1.
            loss_l, loss_t, loss_f = meta_kd_epoch(
                param,
                model,
                task_feats_enh,
                task_feats_raw,
                task_labels,
                task_out_t_all,
                task_idx_train,
                criterion_l,
                criterion_t,
                optimizer,
            )
        else:
            # Standard branch: plain MLP or standard KD, including ablation_mode=0 with use_meta=0.
            if param["exp_setting"] == "tran":
                # Transductive setting: train on idx_train nodes of the full graph.
                loss_l, loss_t, loss_f = train_mini_batch(
                    model,
                    feats,
                    feats_tea,
                    labels,
                    out_t_all,
                    idx_train,
                    criterion_l,
                    criterion_t,
                    optimizer,
                    param,
                )
            else:
                # Inductive setting: train on obs_idx_train nodes of the observed subgraph.
                loss_l, loss_t, loss_f = train_mini_batch(
                    model,
                    obs_feats_enh,
                    None,
                    obs_labels,
                    obs_out_t,
                    obs_idx_train,   # [0, N_obs)
                    criterion_l,
                    criterion_t,
                    optimizer,
                    param,
                )

        loss = loss_l + loss_t + loss_f

        # ----------------- Evaluation -----------------
        if param["exp_setting"] == "tran":
            train_loss, train_acc = evaluate_mini_batch(
                model,
                feats[idx_train],
                labels[idx_train],
                criterion_l,
                evaluator,
            )
            _, val_acc = evaluate_mini_batch(
                model,
                feats[idx_val],
                labels[idx_val],
                criterion_l,
                evaluator,
            )
            _, test_acc = evaluate_mini_batch(
                model,
                feats[idx_test],
                labels[idx_test],
                criterion_l,
                evaluator,
            )
        else:
            train_loss, train_acc = evaluate_mini_batch(
                model,
                obs_feats_enh[obs_idx_train],
                obs_labels[obs_idx_train],
                criterion_l,
                evaluator,
            )
            _, val_acc = evaluate_mini_batch(
                model,
                obs_feats_enh[obs_idx_val],
                obs_labels[obs_idx_val],
                criterion_l,
                evaluator,
            )
            _, test_acc = evaluate_mini_batch(
                model,
                feats[idx_test_ind],
                labels[idx_test_ind],
                criterion_l,
                evaluator,
            )

        # ----------------- Early stopping and best checkpoint tracking -----------------
        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1

        if es == 50:
            print("Early stopping!")
            break

        # print(
        #     "\033[0;1;41m[{}][{}] mode:{} use_meta:{} CLA: {:.5f}, KD: {:.5f}, LF: {:.5f}, "
        #     "Train Loss: {:.5f}, Train Acc: {:.4f}, Val Acc: {:.4f}, "
        #     "Test Acc: {:.4f}, Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
        #         param.get("split_id", 0),
        #         epoch,
        #         ablation_mode,
        #         use_meta,
        #         loss_l,
        #         loss_t,
        #         loss_f,
        #         train_loss,
        #         train_acc,
        #         val_acc,
        #         test_acc,
        #         val_best,
        #         test_val,
        #         test_best,
        #     )
        # )
    # PCGrad diagnostic report hook; no-op in the anonymized release unless diagnostics are enabled.
    TriadMonitor.get(param).report_and_reset(epoch=1)

    model.load_state_dict(state)
    model.eval()

    # Inference time.
    if param["exp_setting"] == "tran":
        start_time = time.time()
        _, val_acc = evaluate_mini_batch(
            model, feats[idx_val], labels[idx_val], criterion_l, evaluator
        )
        end_time = time.time()
    else:
        start_time = time.time()
        _, val_acc = evaluate_mini_batch(
            model, obs_feats_enh, obs_labels, criterion_l, evaluator
        )
        end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time * 1000} ms")

    return test_acc, test_val, test_best




