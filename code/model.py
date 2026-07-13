import os
import math
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
from hyperparams import *


class AudioImageModel(nn.Module):
    def __init__(
        self,
        method,
        # device,
        gradient_ckpt=True,
        encoder_image_dim=768,
        encoder_audio_dim=1024,
        proj_dim=512,
        clip_ckpt_path=None,
        clap_ckpt_path=None,
    ):
        super().__init__()

        self.method = method
        # self.device = device

        # Image encoder (CLIP)
        if clip_ckpt_path is not None:
            # clip_ckpt_path = ["ViT-B-32", "laion2b_s34b_b79k"]
            self.image_model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_ckpt_path[0],
                pretrained=clip_ckpt_path[1] # none or pretrained ckpt
            )
        else:
            raise Exception("No image model specified")

        # Freeze non-visual parts
        if clip_ckpt_path[1] is not None:
            for name, param in self.image_model.named_parameters():
                if "visual.ln_pre" in name:
                    param.requires_grad = False
                if "visual.conv1" in name:
                    param.requires_grad = False
                if "visual.positional_embedding" in name:
                    param.requires_grad = False
                if "visual." not in name:
                    param.requires_grad = False
        else:
            for name, param in self.image_model.named_parameters():
                if "visual." in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            

        if gradient_ckpt:
            self.image_model.set_grad_checkpointing(True)

        # Audio encoder (CLAP)
        if clap_ckpt_path is not None:
            self.audio_model = laion_clap.CLAP_Module(
                enable_fusion=False,
                amodel=clap_ckpt_path[0]
            )
            if clap_ckpt_path[1] is not None:
                self.audio_model.load_ckpt(clap_ckpt_path[1])
        else:
            raise Exception("No audio model specified")

        self.audio_model.enable_fusion = False

        # Freeze non-audio parts
        for name, param in self.audio_model.named_parameters():
            if "model.audio_branch" not in name and "model.audio_projection" not in name:
                param.requires_grad = False

        if gradient_ckpt:
            self.audio_model.model.audio_branch.use_checkpoint = True

        # Projection heads and logit scale
        self.proj_img = None
        self.proj_audio = None
        self.logit_scale_global = None
        self.logit_scale_local = None

        if "sinkhorn" in self.method or "local_baseline" in self.method:
            self.proj_img = nn.Linear(encoder_image_dim, proj_dim)
            self.proj_audio = nn.Linear(encoder_audio_dim, proj_dim)
            
            # self.logit_scale_local = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
            self.logit_scale_local = nn.Parameter(torch.zeros([])) if clap_ckpt_path[1] is None else nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        if "global" in self.method or ("baseline" in self.method and "local" not in self.method):
            self.logit_scale_global = nn.Parameter(torch.zeros([])) if clap_ckpt_path[1] is None else nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def get_image_embeddings(self, imgs):
        image_local, image_global = None, None

        if ("baseline" in self.method and "local" not in self.method) or "global" in self.method:
            image_global = self.image_model.encode_image(imgs, full_rep=False)

        if "sinkhorn" in self.method or "softmax" in self.method or "local_baseline" in self.method: # or local in method
            image_local = self.image_model.encode_image(imgs, full_rep=True)
            image_local = self.proj_img(image_local)

        return image_local, image_global


    def get_audio_embeddings(self, audio_data, subsample_factor=8):
        audio_local, audio_global = None, None

        if ("baseline" in self.method and "local" not in self.method)or "global" in self.method:
            audio_global = self.audio_model.get_audio_embedding_from_data(
                x=audio_data,
                use_tensor=True,
                data_filling="pad",
                full_rep=False,
            )

        if "sinkhorn" in self.method or "softmax" in self.method or "local_baseline" in self.method:
            audio_local = self.audio_model.get_audio_embedding_from_data(
                x=audio_data,
                use_tensor=True,
                data_filling="pad",
                full_rep=True,
            )
            audio_local = self.proj_audio(audio_local)

            # pooling
            audio_local = audio_local.permute(0, 2, 1)
            audio_local = F.max_pool1d(audio_local, kernel_size=subsample_factor, stride=subsample_factor)
            audio_local = audio_local.permute(0, 2, 1)

        return audio_local, audio_global

    
    def freeze_image_encoder(self):
        for param in self.image_model.parameters():
            param.requires_grad = False


    def unfreeze_image_encoder(self):
        for name, param in self.image_model.named_parameters():
            if "visual." in name:
                param.requires_grad = True


    def freeze_audio_encoder(self):
        for param in self.audio_model.parameters():
            param.requires_grad = False


    def unfreeze_audio_encoder(self):
        for name, param in self.audio_model.named_parameters():
            if "model.audio_branch" in name or \
               "model.audio_projection" in name:
                param.requires_grad = True

    
    def forward(self, images, audio_data, subsample_factor=8, image_size=IMAGE_SIZE):
        image_local, image_global = None, None
        if images is not None:
            image_local, image_global = self.get_image_embeddings(images)
            if image_local is not None:
                x = image_local.reshape(-1, 4, 50, 512) if image_size == 448 else image_local.reshape(-1, 1, 50, 512)
                cls_tokens = x[:, :, 0:1, :]
                patch_tokens = x[:, :, 1:, :]
                avg_cls = cls_tokens.mean(dim=1)
                flat_patches = patch_tokens.reshape(-1, 4*49, 512) if image_size == 448 else patch_tokens.reshape(-1, 1*49, 512)
                image_local = torch.cat([avg_cls, flat_patches], dim=1)
            if image_global is not None:
                image_global = image_global.reshape(-1, 4, 512).mean(dim=1) if image_size == 448 else image_global.reshape(-1, 1, 512).mean(dim=1)
                image_global = F.normalize(image_global, dim=-1) # B, 512 

        audio_local, audio_global = None, None
        if audio_data is not None:
            audio_local, audio_global = self.get_audio_embeddings(audio_data, subsample_factor=subsample_factor)
            if audio_local is not None:
                audio_local = audio_local.reshape(-1, 2, 128, 512).reshape(-1, 2*128, 512) # B*2, 128, 512 --> B, 256, 512 (representing 20 second audio)
            if audio_global is not None:
                audio_global = audio_global.reshape(-1, 2, 512).mean(dim=1)
                audio_global = F.normalize(audio_global, dim=-1) # B, 512

        return image_local, image_global, audio_local, audio_global, self.logit_scale_local, self.logit_scale_global

        
    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
        

    def get_optimizer(self, lr, lr_factor):
        head_params = []
        backbone_params = []
    
        head_keywords = [
            "proj_img",            
            "proj_audio",          
            "visual.proj",         
            "visual.ln_post",      
            "audio_projection",   
            "visual.class_embedding",
            "logit_scale_global",
            "logit_scale_local",
        ]
    
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(key in name for key in head_keywords):
                head_params.append(param)
            else:
                backbone_params.append(param)
    
        param_groups = [{"params": head_params, "lr": lr, "weight_decay": 0.01}]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr * lr_factor, "weight_decay": 0.01})
        
        return torch.optim.AdamW(param_groups)