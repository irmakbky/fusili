import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import wandb
import argparse
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.nn.functional import all_gather
from hyperparams import *
from dataset import *
from utils import *
from model import AudioImageModel

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.autograd.set_detect_anomaly(True)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def main():

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    model = AudioImageModel(
        method=METHOD,
        # device=device,
        gradient_ckpt=GRADIENT_CKPT,
        clip_ckpt_path=CLIP_CKPT_PATH,
        clap_ckpt_path=CLAP_CKPT_PATH
    )
    if MODEL_CKPT is not None:
        checkpoint = torch.load(MODEL_CKPT, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if FREEZE_ENCODERS:
        model.freeze_image_encoder()
        model.freeze_audio_encoder()

    model = model.to(device)
    model = model.to(torch.float32)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True  # safe for conditional branches
        # find_unused_parameters=False
    )
    model._set_static_graph()

    # data
    train_loader, val_loader, train_sampler, val_sampler = get_train_val_data_hard_batch(CSV_PATH, model.module.preprocess, BATCH_SIZE, unwrapped=UNWRAPPED, trim_audio=trim_audio, image_size=IMAGE_SIZE, max_audio_len=MAX_AUDIO_LEN, split=SPLIT, return_test=False, ddp=ENABLE_DDP, neg_batch=NEG_BATCH, separate_by_chunk=SEPARATE_BY_CHUNK, random_batch=RANDOM_BATCH)

    optimizer = model.module.get_optimizer(LR, LR_FACTOR)
    if MODEL_CKPT is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    scaler = None
    # scaler = torch.cuda.amp.GradScaler()
    
    if LOG_WANDB and is_main:
        init_wandb(project_name, wandb_run_name, NUM_EPOCHS, LR, train_loader, METHOD, alpha, TAU, n_iters, model.module.image_model, model.module.audio_model, track_grads=True)
    
    scheduler = None
    warmup_active = False
    warmup_step = 0

    for epoch in range(START_EPOCH, NUM_EPOCHS):

        train_sampler.set_epoch(epoch)

        # ---- Unfreeze schedule ----
        if FREEZE_ENCODERS and epoch == UNFREEZE_EPOCH:
            if is_main:
                print(f"--- UNFREEZING AT EPOCH {epoch} ---")

            model.module.unfreeze_image_encoder()
            model.module.unfreeze_audio_encoder()

            optimizer = model.module.get_optimizer(LR, LR_FACTOR)
                    
            warmup_steps = len(train_loader) * WARMUP_EPOCHS
            if warmup_steps > 0:
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,   # start at 10% of current LR
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                warmup_active = True
                warmup_step = 0

        model.train()
        total_loss = 0.0

        for imgs, audio_data, idxs in tqdm(train_loader, disable=not is_main):
            optimizer.zero_grad()

            idxs = idxs.to(device)
            imgs = imgs.to(device, non_blocking=True)
            audio_data = audio_data.to(device, non_blocking=True)
            image_chunks = imgs.reshape(-1, 3, 224, 224) # B*4, 3, 224, 224
            audio_chunks = audio_data.reshape(-1, 480000) # B*2, 480000

            with torch.autocast("cuda", enabled=False):
                image_local, image_global, audio_local, audio_global, logit_scale_local, logit_scale_global = model(
                    image_chunks,
                    audio_chunks
                )

                all_indices = torch.cat(all_gather(idxs), dim=0)
                if NEG_BATCH:
                    all_indices = all_indices[::2]
                
                _, sort_indices = torch.sort(all_indices)

                image_global_all = None
                audio_global_all = None
                if image_global is not None:
                    if NEG_BATCH:
                        image_global_all = torch.cat(all_gather(image_global), dim=0)
                        image_global_all = image_global_all.reshape(-1, 2, 512) # 1 negative per orig
                        image_global_all = image_global_all[sort_indices]
                        pos_examples = image_global_all[:, 0, :]
                        neg_examples = image_global_all[:, 1:, :]
                        neg_examples = neg_examples.reshape(-1, 512)
                        image_global_all = torch.cat([pos_examples, neg_examples], dim=0)
                        
                        audio_global_all = torch.cat(all_gather(audio_global), dim=0)
                        # audio_global_all = audio_global_all.reshape(-1, 5, 512) # 4 negatives
                        audio_global_all = audio_global_all.reshape(-1, 2, 512) # 1 negative per orig
                        audio_global_all = audio_global_all[sort_indices]
                        pos_examples = audio_global_all[:, 0, :]
                        neg_examples = audio_global_all[:, 1:, :]
                        neg_examples = neg_examples.reshape(-1, 512)
                        audio_global_all = torch.cat([pos_examples, neg_examples], dim=0)
                    else:
                        image_global_all = torch.cat(all_gather(image_global), dim=0)[sort_indices]
                        audio_global_all = torch.cat(all_gather(audio_global), dim=0)[sort_indices]

                image_local_all = None
                audio_local_all = None
                if image_local is not None:
                    if NEG_BATCH:
                        image_local_all = torch.cat(all_gather(image_local), dim=0)
                        image_local_all = image_local_all.reshape(-1, 2, 50, 512)
                        image_local_all = image_local_all[sort_indices]
                        pos_examples = image_local_all[:, 0, :, :]
                        neg_examples = image_local_all[:, 1:, :, :] # 64, 49, 512
                        neg_examples = neg_examples.reshape(-1, 50, 512)
                        image_local_all = torch.cat([pos_examples, neg_examples], dim=0)
                        
                        audio_local_all = torch.cat(all_gather(audio_local), dim=0)
                        # audio_local_all = audio_local_all.reshape(-1, 5, 256, 512)
                        audio_local_all = audio_local_all.reshape(-1, 2, 256, 512)
                        audio_local_all = audio_local_all[sort_indices]
                        pos_examples = audio_local_all[:, 0, :, :]
                        neg_examples = audio_local_all[:, 1:, :, :]
                        neg_examples = neg_examples.reshape(-1, 256, 512) # 64, 256, 512
                        audio_local_all = torch.cat([pos_examples, neg_examples], dim=0)
                    else:
                        image_local_all = torch.cat(all_gather(image_local), dim=0)[sort_indices]
                        audio_local_all = torch.cat(all_gather(audio_local), dim=0)[sort_indices]

                loss = compute_loss(
                    image_local_all,
                    audio_local_all,
                    image_global_all,
                    audio_global_all,
                    METHOD,
                    logit_scale_local,
                    logit_scale_global, 
                    alpha=alpha,
                    n_iters=n_iters,
                    tau=TAU,
                    pad_to_square=PAD_TO_SQUARE,
                    detach=DETACH,
                    temperature=TEMPERATURE,
                    neg_batch=NEG_BATCH,
                    regularization=REGULARIZATION,
                    reg_lambda=REG_LAMBDA
                )
                            
            loss.backward()
            optimizer.step()
            # Logit scale param clamping
            with torch.no_grad():
                if logit_scale_global is not None:
                    logit_scale_global.clamp_(0, 4.6052)
                if logit_scale_local is not None:
                    logit_scale_local.clamp_(0, 4.6052)

            if warmup_active:
                scheduler.step()
                warmup_step += 1
                if warmup_step >= warmup_steps:
                    warmup_active = False

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.nn.functional.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item()

        if is_main:
            print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, audio_data, idxs in tqdm(val_loader, disable=not is_main):
                idxs = idxs.to(device)
                imgs = imgs.to(device, non_blocking=True)
                audio_data = audio_data.to(device, non_blocking=True)
                image_chunks = imgs.reshape(-1, 3, 224, 224) # B*4, 3, 224, 224
                audio_chunks = audio_data.reshape(-1, 480000) # B*2, 480000
    
                with torch.autocast("cuda", enabled=False):
                    image_local, image_global, audio_local, audio_global, logit_scale_local, logit_scale_global = model(
                        image_chunks,
                        audio_chunks
                    )
    
                    all_indices = torch.cat(all_gather(idxs), dim=0)
                    _, sort_indices = torch.sort(all_indices)
    
                    # No negative batch setup for validation (validation is without mutations)
                    image_global_all = None
                    audio_global_all = None
                    if image_global is not None:
                        image_global_all = torch.cat(all_gather(image_global), dim=0)[sort_indices]
                        audio_global_all = torch.cat(all_gather(audio_global), dim=0)[sort_indices]
    
                    image_local_all = None
                    audio_local_all = None
                    if image_local is not None:
                        image_local_all = torch.cat(all_gather(image_local), dim=0)[sort_indices]
                        audio_local_all = torch.cat(all_gather(audio_local), dim=0)[sort_indices]
                        
                    loss = compute_loss(
                        image_local_all,
                        audio_local_all,
                        image_global_all,
                        audio_global_all,
                        METHOD,
                        logit_scale_local,
                        logit_scale_global, 
                        alpha=alpha,
                        n_iters=n_iters,
                        tau=TAU,
                        pad_to_square=PAD_TO_SQUARE,
                        detach=DETACH,
                        temperature=TEMPERATURE,
                        # neg_batch=NEG_BATCH,
                        neg_batch=False,
                        regularization=REGULARIZATION,
                        reg_lambda=REG_LAMBDA
                    )

                val_loss += loss.item()


        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_tensor = torch.tensor(avg_val_loss, device=device)
        dist.nn.functional.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = avg_val_loss_tensor.item()

        if is_main and LOG_WANDB:
            wandb.log({"epoch": epoch, "train_loss": avg_loss, "val_loss": avg_val_loss, 
            "temp_global": 1.0 / logit_scale_global.exp().item() if logit_scale_global is not None else 0,
            "temp_local": 1.0 / logit_scale_local.exp().item() if logit_scale_local is not None else 0}, 
            step=epoch)

        if is_main:
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")
            if best_val_loss is None or avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                    'val_loss': avg_val_loss,
                }, os.path.join(CHECKPOINT_DIR, f"best_val.ckpt"))
                if LOG_WANDB:
                    wandb.log({"best_epoch": epoch, "best_val_loss": best_val_loss}, step=epoch)
                
    cleanup_ddp()
    if is_main:
        if LOG_WANDB:
            wandb.finish()

if __name__ == "__main__":
    main()
