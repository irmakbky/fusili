import os
import re
import ast
import random
import torch
import pandas as pd
import librosa
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision

from torch.utils.data import Dataset, DataLoader, random_split, Subset, Sampler
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
from PIL import Image
from hyperparams import NEG_BATCH_DIR, TRAIN_CSV

def resize_unwrapped(img, target_size=(448, 224)):
    return img.resize(target_size, Image.Resampling.LANCZOS)

def resize_and_pad(image, target_size=448, fill_color=(255, 255, 255)):
    w, h = image.size
    if w > h:
        new_w = target_size
        new_h = int(target_size * h / w)
    else:
        new_h = target_size
        new_w = int(target_size * w / h)

    image_resized = image.resize((new_w, new_h), Image.LANCZOS)

    new_image = Image.new("RGB", (target_size, target_size), fill_color)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    new_image.paste(image_resized, (left, top))

    return new_image

def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')


class ImageAudioDataset(Dataset):
    def __init__(self, csv_path, transform=None, unwrapped=False, trim_audio=False, image_size=448, max_audio_len=20, neg_batch=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.unwrapped = unwrapped
        self.trim_audio = trim_audio
        self.image_size = image_size
        self.max_audio_len = max_audio_len
        self.neg_batch = neg_batch

    def __len__(self):
        return len(self.df)

    def _chunk_image(self, img):
        w, h = img.size
        if w == 224:
            chunks = [img]
        else: # 448
            mid_w, mid_h = w // 2, h // 2
            # 4 Regions: Top-Left, Top-Right, Bottom-Left, Bottom-Right
            chunks = [
                F_vision.crop(img, 0, 0, mid_h, mid_w),
                F_vision.crop(img, 0, mid_w, mid_h, mid_w),
                F_vision.crop(img, mid_h, 0, mid_h, mid_w),
                F_vision.crop(img, mid_h, mid_w, mid_h, mid_w)
            ]

        if self.transform: 
            chunks = torch.stack([self.transform(c) for c in chunks])
        
        return chunks

    def _chunk_audio(self, audio_data):
        # audio_data shape: [1, L]
        chunk_size = 10 * 48000 # 2, 10 second chunks
        temp_chunks = [
            audio_data[:, :chunk_size],
            audio_data[:, chunk_size:2 * chunk_size],
        ]
        chunks = torch.stack([F.pad(audio_data,(0, 480000 - audio_data.shape[1]), mode="constant", value=0)
                    for audio_data in temp_chunks])
        return chunks

    def _concat_images(self, images, fill_color=(255, 255, 255)):
        widths = [img.width for img in images]
        total_width = sum(widths) # 448 or 896
        height = images[0].height # 224
    
        min_width = 448 * 2
        final_width = max(total_width, min_width) # should always be 896
        assert final_width == 896
    
        canvas = Image.new(images[0].mode, (final_width, height), fill_color)
    
        x = 0
        for img in images:
            canvas.paste(img, (x, 0))
            x += img.width
    
        return canvas

    def _chunk_image_unwrapped(self, image):
        chunks = []
        for x in range(0, image.width, 224):
            box = (x, 0, x + 224, 224)
            chunks.append(image.crop(box))
        if self.transform: 
            chunks = torch.stack([self.transform(c) for c in chunks])
        return chunks

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_filename']
        audio_path = row['audio_filename']

        if self.unwrapped:
            img_path = ast.literal_eval(img_path)
            images = []
            for path in img_path:
                image = Image.open(path).convert("RGB")
                image = resize_unwrapped(image)
                images.append(image)
            concat_image = self._concat_images(images)
            image_chunks = self._chunk_image_unwrapped(concat_image)
        else:
            img_path = f"{img_path}"
            audio_path = random.choice(audio_path) if isinstance(audio_path, list) else audio_path
            image = Image.open(img_path).convert("RGB")
            image = resize_and_pad(image, target_size=self.image_size)
            image_chunks = self._chunk_image(image)

        if self.neg_batch:
            negative_audio_paths = []
            # for neg_idx in [0,1,2,3]:
            # for neg_idx in [0]:
            neg_path = NEG_BATCH_DIR + sub_path # filepath for the corresponding negative example
            negative_audio_paths.append(neg_path)
            
        
        audio_data, _ = librosa.load(audio_path, sr=48000)
        if self.trim_audio:
            audio_data = audio_data[:self.max_audio_len*48000]
        audio_data = audio_data.reshape(1, -1)
        audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

        audio_chunks = self._chunk_audio(audio_data)

        # NEG BATCH
        if self.neg_batch:
            all_audio_chunks = [audio_chunks]
            for neg_path in negative_audio_paths:
                audio_data, _ = librosa.load(audio_path, sr=48000)
                if self.trim_audio:
                    audio_data = audio_data[:self.max_audio_len*48000]
                audio_data = audio_data.reshape(1, -1)
                audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
                neg_audio_chunks = self._chunk_audio(audio_data)
                all_audio_chunks.append(neg_audio_chunks)
            all_audio_chunks = torch.cat(all_audio_chunks, dim=0)
            audio_chunks = all_audio_chunks

        return image_chunks, audio_chunks, idx

def baseline_local_embeds(model, audio_chunks, image_chunks):
    with torch.autocast("cuda", enabled=False):
        i_local = model.image_model.encode_image(image_chunks, full_rep=True)
        pooled, tokens = model.image_model.visual._pool(i_local)
        image_local_embeddings = (tokens @ model.image_model.visual.proj) # B, 49, 512

        a_local = model.audio_model.get_audio_embedding_from_data(x=audio_chunks, use_tensor=True, data_filling='pad', full_rep=True)
        a_local = model.audio_model.model.audio_projection(a_local)
        a_local = a_local.permute(0, 2, 1)
        a_local = torch.nn.MaxPool1d(kernel_size=8, stride=8)(a_local)
        a_local = a_local.permute(0, 2, 1) # B*2, 128, 512
        audio_local_embeddings = a_local.reshape(-1, 2, 128, 512).reshape(-1, 2*128, 512)
    return image_local_embeddings, audio_local_embeddings


def image_audio_for_validation(img_path, audio_path, model_preprocess, target_size=224, max_audio_len=20):
    image = Image.open(img_path).convert("RGB")
    image = resize_and_pad(image, target_size=target_size)
    image_chunks = torch.stack([model_preprocess(c) for c in [image]])

    audio_data, _ = librosa.load(audio_path, sr=48000)
    audio_data = audio_data[:max_audio_len*48000]
    audio_data = audio_data.reshape(1, -1)
    audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
    chunk_size = 10 * 48000
    temp_chunks = [audio_data[:, :chunk_size], audio_data[:, chunk_size:2 * chunk_size]]
    audio_chunks = torch.stack([F.pad(audio_data,(0, 480000 - audio_data.shape[1]), mode="constant", value=0) for audio_data in temp_chunks])
    
    return image_chunks, audio_chunks

def time_to_token(time):
    return int(time*256/20)

def find_keys_for_token(ranges, audio_token):
    in_ranges = [key for key, start, end in ranges if start <= audio_token < end]
    if in_ranges:
        if len(in_ranges) > 0:
            return in_ranges

    # No ranges contain the token, find closest key
    closest_key, min_dist = None, float('inf')
    for key, start, end in ranges:
        dist = min(abs(audio_token - start), abs(audio_token - end))
        if dist < min_dist:
            min_dist = dist
            closest_key = key
    return [closest_key]    
    
def top_k_acc(sims, audio_slice, all_pieces_patch_start_end_times, top_k):
    splits_list = audio_slice.split("/")
    wav_path = "/".join(splits_list[:-3]).replace("audio_slices", "msmd_wav")+"/"+splits_list[-4]+".wav"
    match = re.search(r"page_(.*?)_system_(.*?)(/|$)", audio_slice)
    page, system_idx = match.group(1), int(match.group(2))
    
    acc = []
    all_times = np.array(list(all_pieces_patch_start_end_times[wav_path][page][system_idx].values())).flatten()
    offset = min(all_times)
    max_audio_token = time_to_token(max(all_times)-offset)

    if max_audio_token == 0 or max_audio_token > 256:
        print("max_audio_token not in range for ", audio_slice, wav_path, page, system_idx)
        return [] # skip

    for audio_token in range(max_audio_token):
        values, indices = torch.topk(sims[:, audio_token], top_k)
        ranges = [
            (k, time_to_token(start_time - offset), time_to_token(end_time - offset))
            for k, (start_time, end_time) in sorted(all_pieces_patch_start_end_times[wav_path][page][system_idx].items())
        ]
        gt_patches = find_keys_for_token(ranges, audio_token)
        found_gt = False
        for pred_idx in indices:
            if pred_idx in gt_patches:
                acc.append(1)
                found_gt = True
                break
        if not found_gt:
            acc.append(0)
    return acc
    

def get_train_val_data(csv_path, preprocess, batch_size, unwrapped=False, trim_audio=False, image_size=448, max_audio_len=20, split="80/10/10", return_test=False, ddp=False, neg_batch=False):
    train_perc, val_perc, test_perc = split.split("/")
    train_perc, val_perc, test_perc = float(train_perc)/100, float(val_perc)/100, float(test_perc)/100
    
    dataset = ImageAudioDataset(csv_path, transform=preprocess, unwrapped=unwrapped, trim_audio=trim_audio, image_size=image_size, max_audio_len=max_audio_len, neg_batch=neg_batch)
    
    train_size = int(float(train_perc) * len(dataset))
    val_size = int(float(val_perc) * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=g)
    if ddp:
        if torch.distributed.get_rank() == 0:
            print("train_size, val_size, test_size", train_size, val_size, test_size)
    else:
        print("train_size, val_size, test_size", train_size, val_size, test_size)
    
    if ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True, seed=42)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True, drop_last=False)
    else:
        train_sampler, val_sampler = None, None
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, drop_last=False)

    if return_test:
        if ddp:
            test_sampler = DistributedSampler(test_set, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=4)
        else:
            test_sampler = None
            test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
        return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler, test_set, dataset
    else:
        if ddp:
            return train_loader, val_loader, train_sampler, val_sampler
        else:
            return train_loader, val_loader          


class SubfolderBatchSampler(Sampler):
    def __init__(self, dataset_subset, batch_size, ddp=False, shuffle=True, drop_last=True, separate_by_chunk=True, random_batch=False):
        self.dataset = dataset_subset.dataset
        self.indices = dataset_subset.indices
        self.batch_size = batch_size
        self.ddp = ddp
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.separate_by_chunk = separate_by_chunk

        if ddp:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.global_batch_size = self.batch_size * self.world_size
        self.random_batch = random_batch
        self.separate_by_chunk = separate_by_chunk

        self.groups = defaultdict(lambda: defaultdict(list))
        for local_idx, global_idx in enumerate(self.indices):
            path = self.dataset.df.iloc[global_idx]['image_filename']
            path_str = ast.literal_eval(path)[0] if (isinstance(path, str) and path.startswith('[')) else path
            
            piece_folder = "/".join(path_str.split("/")[-4:-1])
            cid = "_".join(path_str[:-4].split(piece_folder)[-1].split("_")[:2]) if self.separate_by_chunk else "all"
        
            self.groups[cid][piece_folder].extend([local_idx])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        index_stream = []

        if self.random_batch:
            if self.separate_by_chunk:
                chunk_ids = list(self.groups.keys())
                if self.shuffle:
                    random.Random(self.epoch).shuffle(chunk_ids)

                for cid in chunk_ids:
                    piece_dict = self.groups[cid]
                    piece_names = list(piece_dict.keys())
                    if self.shuffle:
                        random.Random(self.epoch).shuffle(piece_names)
                    for piece in piece_names:
                        indices = piece_dict[piece] # All pages/systems for P1/Ch115
                        if self.shuffle:
                            random.Random(self.epoch).shuffle(indices)
                        
                        index_stream.extend(indices)
            else:
                all_indices = []
                for cid in self.groups:
                    for piece in self.groups[cid]:
                        all_indices.extend(self.groups[cid][piece])
                
                if self.shuffle:
                    random.Random(self.epoch).shuffle(all_indices)
                index_stream.extend(all_indices)

        else:
            all_pieces = set()
            for cid in self.groups:
                all_pieces.update(self.groups[cid].keys())
            piece_list = list(all_pieces)
            
            if self.shuffle:
                random.Random(self.epoch).shuffle(piece_list)

            for piece in piece_list:
                chunk_ids = sorted(self.groups.keys()) 
                for cid in chunk_ids:
                    if piece in self.groups[cid]:
                        index_stream.extend(self.groups[cid][piece])

        global_batches = []
        for i in range(0, len(index_stream), self.global_batch_size):
            batch = index_stream[i : i + self.global_batch_size]
            
            if len(batch) == self.global_batch_size:
                global_batches.append(batch)
            elif not self.drop_last:
                remainder = len(batch) % self.world_size
                if len(batch) - remainder >= self.world_size:
                    global_batches.append(batch[: len(batch) - remainder])

        for global_batch in global_batches:
            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield global_batch[start:end]

    def __len__(self):
        total_indices = sum(len(piece_list) for cid in self.groups for piece_list in self.groups[cid].values())
        return total_indices // self.global_batch_size    

def get_train_val_data_hard_batch(csv_path, preprocess, batch_size, unwrapped=False, trim_audio=False, image_size=448, max_audio_len=20, split="80/10/10", return_test=False, ddp=True, neg_batch=False, separate_by_chunk=False, random_batch=False):
    train_perc, val_perc, test_perc = split.split("/")
    train_perc, val_perc, test_perc = float(train_perc)/100, float(val_perc)/100, float(test_perc)/100
    
    full_dataset = ImageAudioDataset(csv_path, transform=preprocess, unwrapped=unwrapped, trim_audio=trim_audio, image_size=image_size, max_audio_len=max_audio_len, neg_batch=neg_batch)

    def get_piece_id(p):
        path_str = ast.literal_eval(p)[0] if (isinstance(p, str) and p.startswith('[')) else p
        return "/".join(path_str.split("/")[-4:-1]) # Always the piece folder

    df = full_dataset.df.copy()
    df['sub'] = df['image_filename'].apply(get_piece_id)
    unique_subs = df['sub'].unique()
    random.Random(42).shuffle(unique_subs)
    
    n_train, n_val = int(train_perc* len(unique_subs)), int(val_perc* len(unique_subs))
    train_subs, val_subs, test_subs = unique_subs[:n_train], unique_subs[n_train:n_train+n_val], unique_subs[n_train+n_val:]
    
    train_idx = df[df['sub'].isin(train_subs)].index.tolist()
    val_idx = df[df['sub'].isin(val_subs)].index.tolist()
    test_idx = df[df['sub'].isin(test_subs)].index.tolist()

    # if random_batch: # shuffle (not from same piece) if random batch
    #     random.Random(42).shuffle(train_idx)
    random.Random(42).shuffle(val_idx)
    random.Random(42).shuffle(test_idx)

    if separate_by_chunk or neg_batch:
        train_dataset = ImageAudioDataset(TRAIN_CSV, transform=preprocess, unwrapped=unwrapped, trim_audio=trim_audio, image_size=image_size, max_audio_len=max_audio_len, neg_batch=neg_batch)
        train_df = train_dataset.df.copy()
        train_df['sub'] = train_df['image_filename'].apply(get_piece_id)
        unique_subs = train_df['sub'].unique()
        random.Random(42).shuffle(unique_subs)
        train_subs = unique_subs
        train_idx = train_df[train_df['sub'].isin(train_subs)].index.tolist()
        train_set = Subset(train_dataset, train_idx)
    else:
        train_set = Subset(full_dataset, train_idx)
    
    val_set, test_set = Subset(full_dataset, val_idx), Subset(full_dataset, test_idx)

    t_sampler = SubfolderBatchSampler(train_set, batch_size, ddp=ddp, shuffle=True, separate_by_chunk=separate_by_chunk, random_batch=random_batch)
    train_loader = DataLoader(train_set, batch_sampler=t_sampler, num_workers=4, pin_memory=True)
    
    v_sampler = DistributedSampler(val_set, shuffle=False) if ddp else None
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=v_sampler, shuffle=(v_sampler is None), num_workers=4)
        

    if return_test:
        ts_sampler = DistributedSampler(test_set, shuffle=False) if ddp else None
        test_loader = DataLoader(test_set, batch_size=batch_size, sampler=ts_sampler, num_workers=4)
        return train_loader, val_loader, test_loader, t_sampler, v_sampler, ts_sampler, test_set, full_dataset
    
    return (train_loader, val_loader, t_sampler, v_sampler) if ddp else (train_loader, val_loader)
