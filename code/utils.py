import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import laion_clap
import wandb
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.utils.checkpoint import checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from model import AudioImageModel
from hyperparams import CLIP_CKPT_PATH, CLAP_CKPT_PATH

def init_wandb(project_name, wandb_run_name, NUM_EPOCHS, LR, train_loader, METHOD, alpha, TAU, n_iters, image_model, audio_model, track_grads=True):
    wandb.init(
       project=project_name,
       name=wandb_run_name,
       config={
           "num_epochs": NUM_EPOCHS,
           "lr": LR,
           "batch_size": train_loader.batch_size,
           "method": METHOD,
           "alpha": alpha,
           "tau": TAU,
           "n_iters": n_iters,
           "unfreeze_epoch": 0,
       }
    )

    if track_grads:
        wandb.watch(image_model, log="gradients", log_freq=100)
        wandb.watch(audio_model, log="gradients", log_freq=100)   

def load_model_for_eval(
    ckpt_path,
    method,
    device,
    gradient_ckpt=False,
    clip_ckpt_path=CLIP_CKPT_PATH,
    clap_ckpt_path=CLAP_CKPT_PATH,
):

    model = AudioImageModel(
        method=method,
        # device=device,
        gradient_ckpt=gradient_ckpt,
        clip_ckpt_path=clip_ckpt_path,
        clap_ckpt_path=clap_ckpt_path,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    return model

def get_image_embeddings(image_model, imgs, proj_img, method):
    image_local_embeddings, image_global_embeddings = None, None
    if "baseline" in method or "global" in method:
        image_global_embeddings = image_model.encode_image(imgs, full_rep=False)
    if "sinkhorn" in method or "softmax" in method:
        image_local_embeddings = image_model.encode_image(imgs, full_rep=True)
        image_local_embeddings = proj_img(image_local_embeddings)

    return image_local_embeddings, image_global_embeddings

def get_audio_embeddings(audio_model, audio_data, proj_audio, method, subsample_factor=8):
    # audio_data: list of audio waveforms of shape [len] wheren len < 10 * 48000
    audio_local_embeddings, audio_global_embeddings = None, None
    if "baseline" in method or "global" in method:
        audio_global_embeddings = audio_model.get_audio_embedding_from_data(x=audio_data, use_tensor=True, data_filling='pad', full_rep=False)
    if "sinkhorn" in method or "softmax" in method:
        audio_local_embeddings = audio_model.get_audio_embedding_from_data(x=audio_data, use_tensor=True, data_filling='pad', full_rep=True)
        audio_local_embeddings = proj_audio(audio_local_embeddings)
    if audio_local_embeddings is not None:
        audio_local_embeddings = audio_local_embeddings.permute(0, 2, 1)
        audio_local_embeddings = nn.MaxPool1d(kernel_size=subsample_factor, stride=subsample_factor)(audio_local_embeddings)
        audio_local_embeddings = audio_local_embeddings.permute(0, 2, 1)

    return audio_local_embeddings, audio_global_embeddings    

def sinkhorn_batch(logits, n_iters=20):
    Q = logits.exp()
    if torch.isinf(Q).any():
        print(Q)
    for _ in range(n_iters):
        Q = Q / (Q.sum(dim=-1, keepdim=True) + 1e-6)  # row norm
        Q = Q / (Q.sum(dim=-2, keepdim=True) + 1e-6)  # col norm
    return Q

def sinkhorn_batch_log(logits, n_iters=20):
    Q = logits
    for _ in range(n_iters):
        Q = Q - torch.logsumexp(Q, -1, keepdim=True)
        Q = Q - torch.logsumexp(Q, -2, keepdim=True)
    return Q.exp()

def sinkhorn_batch_log_rectangle(logits, n_iters=20):
    B, N, M = logits.shape
    log_r = -torch.log(torch.tensor(N, device=logits.device))
    log_c = -torch.log(torch.tensor(M, device=logits.device))

    Q = logits
    for _ in range(n_iters):
        Q = Q - torch.logsumexp(Q, dim=-1, keepdim=True) + log_r
        Q = Q - torch.logsumexp(Q, dim=-2, keepdim=True) + log_c
        
    return Q.exp()

def sinkhorn_forward(logits, n_iters=20):
    if not logits.requires_grad:
        return sinkhorn_batch_log_rectangle(logits, n_iters=n_iters)
    iters_t = torch.tensor(n_iters, device=logits.device)
    return checkpoint(sinkhorn_batch_log_rectangle, logits, iters_t, use_reentrant=False)

def sinkhorn_matrix(T, V, logit_scale_local, tau=0.07, n_iters=20, pad_to_square=True, detach=False, return_A_only=False):    
    B, N, D = T.shape
    M = V.shape[1]

    T = F.normalize(T, dim=-1)
    V = F.normalize(V, dim=-1)
    # S_all = torch.einsum('itd,jpd->ijtp', T, V) / tau
    S_all = torch.einsum('itd,jpd->ijtp', T, V) * logit_scale_local.exp()
    S_flat = S_all.reshape(B * B, N, M)
    if pad_to_square:
        print("Padding to square")
        if N > M:
            S_flat = F.pad(S_flat, (0, 0, 0, 0, 0, N - M), mode="constant", value=-1e9) # pad to square matrix with neg inf for log
        elif M > N:
            S_flat = F.pad(S_flat, (0, 0, 0, M - N), mode="constant", value=-1e9) # pad to square matrix with neg inf for log
    # A_flat = sinkhorn_batch_log(S_flat, n_iters=n_iters)
    if detach:
        A_flat = sinkhorn_forward(S_flat.detach(), n_iters=n_iters)
    else:
        A_flat = sinkhorn_forward(S_flat, n_iters=n_iters)
    s_matrix = (A_flat * S_flat) # B, N, M
    if return_A_only:
        return A_flat
    return s_matrix

def sinkhorn_score(T, V, logit_scale_local, tau=0.07, n_iters=20, pad_to_square=True, detach=False):
    B = T.shape[0]
    logits = torch.zeros(B, B, device=T.device)
    s_matrix = sinkhorn_matrix(T, V, logit_scale_local, tau=tau, n_iters=n_iters, pad_to_square=pad_to_square, detach=detach)
    scores = s_matrix.sum(dim=(-1, -2))  # [B*B]
    logits = scores.view(B, B)
    return logits    

def sinkhorn_infonce_loss(T, V, logit_scale_local, tau=0.07, n_iters=20, pad_to_square=True, detach=False):
    B, N, D = T.shape
    
    logits = sinkhorn_score(T, V, logit_scale_local, tau=tau, n_iters=n_iters, pad_to_square=pad_to_square, detach=detach)
    labels = torch.arange(B, device=T.device)
    loss_t2i = F.cross_entropy(logits, labels)
    loss_i2t = F.cross_entropy(logits.t(), labels)

    return (loss_t2i + loss_i2t) / 2    

def sinkhorn_infonce_loss_chunked(T, V, logit_scale_local, tau=0.07, n_iters=20, chunk_size=4, pad_to_square=True, detach=False, neg_batch=False, regularization=False, reg_lambda=0):
    B, N, D = T.shape
    B2, M, _ = V.shape
    # assert B % chunk_size == 0
    
    T = F.normalize(T, dim=-1)
    V = F.normalize(V, dim=-1)

    def compute_logits_chunk(T_chunk_slice, V_full_slice, detach):
        # S_chunk = torch.einsum('itd,jpd->ijtp', T_chunk_slice, V_full_slice) / tau
        S_chunk = torch.einsum('itd,jpd->ijtp', T_chunk_slice, V_full_slice) * logit_scale_local.exp()
        
        curr_chunk = T_chunk_slice.shape[0]
        S_flat = S_chunk.reshape(curr_chunk * B2, N, M)
        
        if detach:
            A_flat = sinkhorn_batch_log_rectangle(S_flat.detach(), n_iters=n_iters)
        else:
            A_flat = sinkhorn_batch_log_rectangle(S_flat, n_iters=n_iters)

        reg_loss = 0.0
        if regularization:
            reg_loss = sinkhorn_regularization(A_flat[:, 1:])
            # reg_loss = sinkhorn_regularization((A_flat * S_flat))
        
        scores = (A_flat * S_flat).sum(dim=(-1, -2))
        return scores.reshape(curr_chunk, B2), reg_loss

    logits_list = []
    reg_losses = []
    for i in range(0, B, chunk_size):
        T_chunk = T[i:i + chunk_size]

        # chunk_logits = compute_logits_chunk(T_chunk, V, detach)
        
        chunk_logits, reg_loss = checkpoint(
            compute_logits_chunk, 
            T_chunk, 
            V, 
            detach,
            use_reentrant=False
        )
        logits_list.append(chunk_logits)
        reg_losses.append(reg_loss)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.arange(B, device=T.device)
    total_reg_loss = None
    if regularization:
        total_reg_loss = torch.stack(reg_losses).mean()

    if neg_batch:
        assert B == B2
        assert B % 2 == 0
        true_batch_size = int(B/2)
        loss_i2t = torch.nn.CrossEntropyLoss()(logits[:true_batch_size, :], labels[:true_batch_size])
        loss_t2i = torch.nn.CrossEntropyLoss()(logits[:, :true_batch_size].t(), labels[:true_batch_size])
    else:    
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

    if regularization:
        return ((loss_i2t + loss_t2i) / 2) + (0.5 * total_reg_loss)
    else:
        return (loss_i2t + loss_t2i) / 2

def sinkhorn_infonce_loss_chunked_distributed(T, V, logit_scale_local, tau=0.07, n_iters=20, chunk_size=4, pad_to_square=True, detach=False):
    """
    Distributed-correct Sinkhorn InfoNCE.
    Each rank computes loss only for its local rows
    """
    B, N, D = T.shape
    M = V.shape[1]

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_batch = B // world_size

    assert B % world_size == 0, "Global batch must be divisible by world_size"

    T = F.normalize(T, dim=-1)
    V = F.normalize(V, dim=-1)

    logits_list = []

    for i in range(0, B, chunk_size):
        T_chunk = T[i:i + chunk_size]                 
        curr_chunk = T_chunk.shape[0]
        
        # S_chunk = torch.einsum('itd,jpd->ijtp', T_chunk, V) / tau
        S_chunk = torch.einsum('itd,jpd->ijtp', T_chunk, V) * logit_scale_local.exp()
        S_flat = S_chunk.reshape(curr_chunk * B, N, M)

        if pad_to_square:
            if N > M:
                S_flat = F.pad(
                    S_flat, (0, 0, 0, 0, 0, N - M),
                    mode="constant", value=-1e9
                )
            elif M > N:
                S_flat = F.pad(
                    S_flat, (0, 0, 0, M - N),
                    mode="constant", value=-1e9
                )

        if detach:
            A_flat = sinkhorn_batch_log_rectangle(
                S_flat.detach(), n_iters=n_iters
            )
        else:
            A_flat = sinkhorn_batch_log_rectangle(
                S_flat, n_iters=n_iters
            )

        scores = (A_flat * S_flat).sum(dim=(-1, -2))
        logits_chunk = scores.reshape(curr_chunk, B)

        logits_list.append(logits_chunk)

    # Full B x B matrix
    logits = torch.cat(logits_list, dim=0)

    # slice only local rows
    start = rank * local_batch
    end = start + local_batch

    local_logits_i2t = logits[start:end]          # [local_batch, B]
    local_logits_t2i = logits.t()[start:end]      # [local_batch, B]

    local_labels = torch.arange(start, end, device=T.device)
    loss_i2t = F.cross_entropy(local_logits_i2t, local_labels)
    loss_t2i = F.cross_entropy(local_logits_t2i, local_labels)

    loss = (loss_i2t + loss_t2i) / 2

    return loss


def softmax_score(T, V, logit_scale_local, tau=0.07):
    B, N, D = T.shape
    M = V.shape[1]

    T = F.normalize(T, dim=-1)
    V = F.normalize(V, dim=-1)

    logits = torch.zeros(B, B, device=T.device)
    # S_all = torch.einsum('itd,jpd->ijtp', T, V) / tau
    S_all = torch.einsum('itd,jpd->ijtp', T, V) * logit_scale_local.exp()
    S_flat = S_all.reshape(B * B, N, M)
    A_flat = F.softmax(S_flat, dim=-2)
    scores = (A_flat * S_flat).sum(dim=(-1, -2))
    logits = scores.view(B, B)
    return logits

def softmax_infonce_loss(T, V, logit_scale_local, tau=0.07):
    B, N, D = T.shape
    
    logits = softmax_score(T, V, logit_scale_local, tau=tau)
    labels = torch.arange(B, device=T.device)
    loss_t2i = F.cross_entropy(logits, labels)
    loss_i2t = F.cross_entropy(logits.t(), labels)

    return (loss_t2i + loss_i2t) / 2  

def cosine_score(T, V, logit_scale_local, tau=0.07):
    B, N, D = T.shape
    M = V.shape[1]

    # normalize tokens
    T = F.normalize(T, dim=-1)
    V = F.normalize(V, dim=-1)

    logits = torch.zeros(B, B, device=T.device)
    # S_all = torch.einsum('itd,jpd->ijtp', T, V) / tau
    S_all = torch.einsum('itd,jpd->ijtp', T, V) * logit_scale_local.exp()
    S_flat = S_all.reshape(B * B, N, M)
    scores = S_flat.sum(dim=(-1, -2))
    logits = scores.view(B, B)
    return logits

def cosine_infonce_loss(T, V, logit_scale_local, tau=0.07):
    # local, on full features
    B, N, D = T.shape
    
    logits = cosine_score(T, V, logit_scale_local, tau=tau)
    labels = torch.arange(B, device=T.device)
    loss_t2i = F.cross_entropy(logits, labels)
    loss_i2t = F.cross_entropy(logits.t(), labels)

    return (loss_t2i + loss_i2t) / 2      

def clip_info_nce_loss(image_feats, audio_feats, logit_scale_global, temperature=0.07, neg_batch=False):
    B = len(image_feats)
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    audio_feats = audio_feats / audio_feats.norm(dim=-1, keepdim=True)
    
    # logits = image_feats @ audio_feats.t() / temperature
    logits = (image_feats @ audio_feats.t()) * logit_scale_global.exp()
    labels = torch.arange(B, device=image_feats.device)
    
    if neg_batch:
        assert B % 2 == 0
        true_batch_size = int(B/2)
        loss_i2t = torch.nn.CrossEntropyLoss()(logits[:true_batch_size, :], labels[:true_batch_size])
        loss_t2i = torch.nn.CrossEntropyLoss()(logits[:, :true_batch_size].t(), labels[:true_batch_size])
    else:    
        loss_i2t = torch.nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = torch.nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2    

def clip_info_nce_loss_distributed(image_feats, audio_feats, logit_scale_global, temperature=0.07):
    image_feats = F.normalize(image_feats, dim=-1)
    audio_feats = F.normalize(audio_feats, dim=-1)
    
    # logits = image_feats @ audio_feats.t() / temperature
    logits = (image_feats @ audio_feats.t()) * logit_scale_global.exp()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    batch_size_per_gpu = len(image_feats) // world_size
    
    start_idx = rank * batch_size_per_gpu
    end_idx = start_idx + batch_size_per_gpu
    
    labels = torch.arange(start_idx, end_idx, device=image_feats.device)
    
    loss_i2t = torch.nn.CrossEntropyLoss()(logits[start_idx:end_idx], labels)
    loss_t2i = torch.nn.CrossEntropyLoss()(logits.t()[start_idx:end_idx], labels)
    
    return (loss_i2t + loss_t2i) / 2    

def compute_loss(image_tokens, audio_tokens, global_image_tokens, global_audio_tokens, method, logit_scale_local, logit_scale_global, alpha=0.5, n_iters=20, tau=0.07, pad_to_square=True, detach=False, temperature=0.07, neg_batch=False, regularization=False, reg_lambda=0):
    """
        image_tokens: B x N x D
        audio_tokens: B x M x D
        globa_image_tokens: B x D
        globa_audio_tokens: B x D
    """
    if "global_local_baseline" in method:
        global_loss = clip_info_nce_loss(global_image_tokens, global_audio_tokens, logit_scale_global, temperature=temperature, neg_batch=neg_batch)
        local_loss = cosine_infonce_loss(image_tokens, audio_tokens, logit_scale_local)
        return alpha * local_loss + (1 - alpha) * global_loss
    elif "local_baseline" in method:
        return cosine_infonce_loss(image_tokens, audio_tokens, logit_scale_local)
    elif "baseline" in method:
        return clip_info_nce_loss(global_image_tokens, global_audio_tokens, logit_scale_global, temperature=temperature, neg_batch=neg_batch)
    elif "sinkhorn_global" in method:
        sinkhorn_loss = sinkhorn_infonce_loss_chunked(image_tokens, audio_tokens, logit_scale_local, n_iters=n_iters, tau=tau, pad_to_square=pad_to_square, detach=detach, neg_batch=neg_batch, regularization=regularization, reg_lambda=reg_lambda)
        global_loss = clip_info_nce_loss(global_image_tokens, global_audio_tokens, logit_scale_global, temperature=temperature, neg_batch=neg_batch)
        return alpha * sinkhorn_loss + (1 - alpha) * global_loss       
    elif "sinkhorn" in method:
        return sinkhorn_infonce_loss_chunked(image_tokens, audio_tokens, logit_scale_local, n_iters=n_iters, tau=tau, pad_to_square=pad_to_square, detach=detach, neg_batch=neg_batch, regularization=regularization, reg_lambda=reg_lambda)
    # elif "softmax_global" in method:
    #     softmax_loss = softmax_infonce_loss(image_tokens, audio_tokens, logit_scale_local, tau=tau)
    #     global_loss = clip_info_nce_loss(global_image_tokens, global_audio_tokens, logit_scale_global, temperature=temperature, neg_batch=neg_batch)
    #     return alpha * softmax_loss + (1 - alpha) * global_loss


def sinkhorn_regularization(A, num_cols=7):
    # A is A_flat from Sinkhorn
    device = A.device
    K, N, M = A.shape
    assert N == 49
    rows = torch.arange(N, device=device) // num_cols
    cols = torch.arange(N, device=device) % num_cols
    
    rows = rows.view(1, N, 1)
    cols = cols.view(1, N, 1)
    
    A_t = A[:, :, :-1]  # (K, N, M-1)
    A_tp1 = A[:, :, 1:]  # (K, N, M-1)
    
    # column differences between all pairs
    col_i = cols  # (1, N, 1)
    col_j = cols  # (1, N, 1)
    d_col = col_j.view(1, N, 1, 1) - col_i.view(1, 1, N, 1)  # (1, N, N, 1)
    
    # row differences
    row_i = rows
    row_j = rows
    d_row = row_j.view(1, N, 1, 1) - row_i.view(1, 1, N, 1)  # (1, N, N, 1)
    
    # compute weighted distance
    A_i = A_t.view(K, N, 1, M-1) # prob at time t
    A_j = A_tp1.view(K, 1, N, M-1) # prob at time t+1
    
    # mask allowed wrap (move to next staff): col_i=6 → col_j=0 and row_j > row_i
    wrap_mask = ((col_i.view(1, N, 1, 1) == 6) & (col_j.view(1, 1, N, 1) == 0) & (d_row > 0))
    mask_float = (~wrap_mask).float()
    
    # Ai * Aj --> probability that we transitioned from index i to j
    # backward jumps (left)
    penalty_back = F.relu(-d_col) * A_i * A_j * mask_float
    
    # forward jumps >1
    penalty_forward = F.relu(d_col - 1) * A_i * A_j * mask_float
    
    # big jumps down (rows)
    penalty_down = F.relu(d_row - 1) * A_i * A_j * mask_float
    penalty_up = F.relu(-d_row) * A_i * A_j * mask_float # new

    # new
    # column mass concentration, high entropy means mass in more columns, low means concentrated in one column
    col_mass = A.sum(dim=1)
    col_prob = col_mass / (col_mass.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(col_prob * torch.log(col_prob + 1e-8)).sum(dim=1)
    loss_col = entropy.mean() 
    
    # loss = (penalty_back + penalty_forward + penalty_down + penalty_up + loss_col).mean()
    pairwise_loss = (penalty_back + penalty_forward + penalty_down + penalty_up).mean()
    loss = pairwise_loss + loss_col

    return loss