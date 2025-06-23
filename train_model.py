# https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25/notebook를 기반으로 함
import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import timm
import joblib 

import optuna
from optuna.trial import TrialState
logging.getLogger("optuna").setLevel(logging.WARNING) 

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    seed = 42
    debug = False # 디버그용
    apex = False
    print_freq = 100
    num_workers = 4
    
    OUTPUT_DIR = '/home/egus05/models/' # 모델 저장 경로
    
    train_datadir = '/home/egus05/birdclef-2025/train_audio'
    train_csv = '/home/egus05/birdclef-2025/train.csv'
    test_soundscapes = '/home/egus05/birdclef-2025/test_soundscapes'
    submission_csv = '/home/egus05/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/home/egus05/birdclef-2025/taxonomy.csv'

    # 미리 계산된 멜 스펙트로그램 데이터의 메타데이터 CSV 파일 경로.
    spectrogram_metadata_csv = '/home/egus05/data_melspec_overlapping/processed_melspec_metadata_overlapping.csv'
    
    # 모델 아키텍처 설정
    model_name = 'tf_efficientnet_b1' 
    pretrained = True 
    in_channels = 1

    LOAD_DATA_FROM_NPY = True 
    FS = 32000 # 샘플링 레이트
    TARGET_DURATION = 5.0 # 각 오디오 세그먼트의 길이
    TARGET_SHAPE = (256, 256)
    
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    epochs = 20 # 총 학습 에포크 수
    batch_size = 32
    criterion = 'BCEWithLogitsLoss' # 다중 레이블 분류를 위한 손실 함수

    n_fold = 3
    selected_folds = [0, 1, 2]

    optimizer = 'AdamW' # 최적화 함수
    lr = 1e-4 # 학습률 (OneCycleLR의 최대 학습률로 사용)
    weight_decay = 1e-5 # 가중치 감소
    
    scheduler = 'CosineAnnealingLR' 
    min_lr = 1e-6 # CosineAnnealingLR 사용 시 최소 학습률.

    # 데이터 증강 파라미터 (업데이트됨)
    random_crop_audio_prob = 0.8 # 오디오 파일 로드 시 무작위 5초 구간 추출 확률 (길이가 긴 경우)
    gaussian_noise_prob = 0.4 # 가우시안 노이즈 추가 확률
    gaussian_noise_std_range = (0.005, 0.02) # 가우시안 노이즈 표준편차 범위
    spec_masking_prob = 0.7 # 스펙트로그램 마스킹 적용 확률
    mask_time_ratio_range = (0.02, 0.1) # 시간 마스크 폭 비율 (스펙트로그램 너비의 비율)
    mask_freq_ratio_range = (0.02, 0.1) # 주파수 마스크 높이 비율 (스펙트로그램 높이의 비율)
    num_mask_range = (1, 3) # 각 유형(시간/주파수) 마스크 수
    spec_brightness_contrast_prob = 0.5 # 밝기/대비 조절 확률
    brightness_range = (-0.1, 0.1) # 밝기 조절 범위 (덧셈)
    contrast_range = (0.8, 1.2) # 대비 조절 범위 (곱셈)
    cutmix_prob = 0.3 # CutMix 적용 확률 (Mixup은 비활성화)
    mix_alpha = 0.5 # CutMix의 알파 파라미터 (베타 분포)
    mixup_prob = 0.0 # Mixup 비활성화
    audio_time_shift_prob = 0.3 # 오디오 시간 시프트 적용 확률
    audio_pitch_shift_prob = 0.3 # 오디오 피치 시프트 적용 확률
    audio_time_stretch_prob = 0.3 # 오디오 시간 스트레치 적용 확률
    pitch_shift_n_steps_range = (-2, 2) # 피치 시프트 단계 (반음)
    time_stretch_rate_range = (0.8, 1.2) # 시간 스트레치 비율

    # 조기 종료(Early Stopping) 파라미터
    early_stopping_patience = 5 # 검증 성능 향상 없는 에포크 수
    early_stopping_min_delta = 0.001 # 성능 개선으로 간주할 최소 변화량

    # Optuna 관련 파라미터
    use_optuna = True # Optuna 최적화 활성화 여부
    optuna_n_trials = 5 # Optuna 트라이얼(시도) 수
    optuna_study_name = "birdclef2025_hp_tuning_overlapping_data" # 스터디 이름 변경
    optuna_storage = "sqlite:///db.sqlite3" # Optuna 결과 저장 (선택 사항)

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]
            self.train_df_debug_limit = 100 # 디버그 모드에서 샘플 수 줄이기
            self.num_workers = 0 # 디버그 시 num_workers = 0으로 설정하여 디버깅 용이
            self.use_optuna = False # 디버그 모드에서는 Optuna 비활성화

cfg = CFG()

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
def audio2melspec(audio_data, cfg):
    if np.isnan(audio_data).any():
        audio_data = np.nan_to_num(audio_data, nan=0.0) 

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_file(audio_path, cfg, is_train_mode=True):
    """
    단일 오디오 파일을 처리하여 멜 스펙트로그램을 얻습니다.
    (이 함수는 LOAD_DATA_FROM_NPY가 False일 때만 호출됩니다.)
    """
    try:
        audio_data, sr = librosa.load(audio_path, sr=cfg.FS, mono=True)

        if is_train_mode:
            if random.random() < cfg.audio_time_shift_prob:
                shift_max = int(sr * 0.5) 
                shift = random.randint(-shift_max, shift_max)
                audio_data = np.roll(audio_data, shift) 

            if random.random() < cfg.audio_pitch_shift_prob:
                n_steps = random.uniform(cfg.pitch_shift_n_steps_range[0], cfg.pitch_shift_n_steps_range[1])
                audio_data = librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=n_steps)

            if random.random() < cfg.audio_time_stretch_prob:
                rate = random.uniform(cfg.time_stretch_rate_range[0], cfg.time_stretch_rate_range[1])
                stretched_audio = librosa.effects.time_stretch(y=audio_data, rate=rate)
                if len(stretched_audio) > len(audio_data):
                    audio_data = stretched_audio[:len(audio_data)]
                else:
                    audio_data = np.pad(stretched_audio, (0, len(audio_data) - len(stretched_audio)), mode='constant')

        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
        elif len(audio_data) > target_samples:
            if is_train_mode and random.random() < cfg.random_crop_audio_prob:
                start_idx = random.randint(0, len(audio_data) - target_samples)
            else: 
                start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
            audio_data = audio_data[start_idx:start_idx + target_samples]

        mel_spec = audio2melspec(audio_data, cfg)
        
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, (cfg.TARGET_SHAPE[1], cfg.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)
        
    except Exception as e:
        return None
    

class BirdCLEFDatasetFromNPY(Dataset): 
    def __init__(self, df, cfg, mode="train"): # spectrograms 인자 제거
        self.df = df
        self.cfg = cfg
        self.mode = mode

        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        # debug 모드에서 데이터프레임 샘플링은 run_training에서 처리
        if cfg.debug:
            print(f"디버그 모드: {mode} 데이터셋이 {len(self.df)}개 샘플로 제한됨.")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        spec = None
        if self.cfg.LOAD_DATA_FROM_NPY: # .npy 파일에서 로드
            try:
                # 'melspec_filepath' 컬럼에서 .npy 파일 경로를 가져와 로드
                spec = np.load(row['melspec_filepath'])
            except Exception as e:
                # print(f"경고: {row['melspec_filepath']} 로드 오류: {e}. 0 배열을 사용합니다.")
                spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
        else: # 오디오 파일에서 즉석 생성
            # row['original_filename']이 없거나 잘못된 경우를 대비하여 기본값 설정
            original_filename = row.get('original_filename', row.get('filename')) 
            spec = process_audio_file(os.path.join(self.cfg.train_datadir, original_filename), self.cfg, is_train_mode=(self.mode == "train"))

        if spec is None: # process_audio_file에서 오류 발생 시
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train" or self.mode == "valid": 
                print(f"경고: {original_filename}에 대한 스펙트로그램을 찾거나 생성할 수 없습니다. 0 배열을 사용합니다.")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        if self.mode == "train":
            # 데이터 증강 (가우시안 노이즈)
            if random.random() < self.cfg.gaussian_noise_prob:
                spec = self.add_gaussian_noise(spec, std_range=self.cfg.gaussian_noise_std_range)
            
            # 스펙트로그램 마스킹 (XY 마스킹)
            if random.random() < self.cfg.spec_masking_prob:
                spec = self.apply_spec_masking(spec)
            
            # 무작위 밝기/대비 
            if random.random() < self.cfg.spec_brightness_contrast_prob:
                spec = self.apply_brightness_contrast(spec)
            
        target = self.encode_label(row['primary_label'])
        
        # 보조 레이블 처리
        # secondary_labels가 문자열 '[...]' 형태로 저장되어 있을 수 있으므로 eval 사용
        if 'secondary_labels' in row and pd.notna(row['secondary_labels']):
            secondary_labels_str = str(row['secondary_labels'])
            if secondary_labels_str not in ['[]', '']: # 비어있지 않은 경우에만 eval 시도
                try:
                    secondary_labels = eval(secondary_labels_str)
                    if isinstance(secondary_labels, list): # eval 결과가 리스트인지 확인
                        for label in secondary_labels:
                            if label in self.label_to_idx:
                                target[self.label_to_idx[label]] = 1.0
                except:
                    pass # eval 실패 시 무시
        
        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['original_filename'] # 원본 파일명 유지 (디버깅용)
        }
    
    def add_gaussian_noise(self, spec_tensor, mean=0.0, std_range=(0.005, 0.02)):
        std = random.uniform(std_range[0], std_range[1])
        noise = torch.randn(spec_tensor.size(), device=spec_tensor.device) * std + mean
        noisy_spec = spec_tensor + noise
        return torch.clamp(noisy_spec, 0, 1)

    def apply_spec_masking(self, spec): 
        # 시간 마스킹 (가로 줄무늬)
        if random.random() < 0.5: 
            num_masks = random.randint(self.cfg.num_mask_range[0], self.cfg.num_mask_range[1])
            for _ in range(num_masks):
                width = random.randint(int(spec.shape[2] * self.cfg.mask_time_ratio_range[0]), 
                                       int(spec.shape[2] * self.cfg.mask_time_ratio_range[1]))
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0 
        
        # 주파수 마스킹 (세로 줄무늬)
        if random.random() < 0.5: 
            num_masks = random.randint(self.cfg.num_mask_range[0], self.cfg.num_mask_range[1])
            for _ in range(num_masks):
                height = random.randint(int(spec.shape[1] * self.cfg.mask_freq_ratio_range[0]), 
                                        int(spec.shape[1] * self.cfg.mask_freq_ratio_range[1]))
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0 
                
        return spec
    
    def apply_brightness_contrast(self, spec): 
        gain = random.uniform(self.cfg.contrast_range[0], self.cfg.contrast_range[1]) 
        bias = random.uniform(self.cfg.brightness_range[0], self.cfg.brightness_range[1]) 
        spec = spec * gain + bias
        return torch.clamp(spec, 0, 1) 
    
    def encode_label(self, label):
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target
    
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0, is_horizontal_only=False):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    target_a = y
    target_b = y[rand_index]
    
    if is_horizontal_only:
        bbx1, bby1, bbx2, bby2 = rand_bbox_horizontal_only(x.size(), lam)
    else:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (x.size()[-1] * x.size()[-2])
    return x_mixed, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def rand_bbox_horizontal_only(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    bbx1 = np.clip(random.randint(0, W - cut_w), 0, W)
    bby1 = 0 
    bbx2 = np.clip(bbx1 + cut_w, 0, W)
    bby2 = H 
    return bbx1, bby1, bbx2, bby2

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def collate_fn(batch):
    """서로 다른 크기의 스펙트로그램을 처리하기 위한 사용자 정의 collate 함수"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    result = {key: [] for key in batch[0].keys()}
    
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    
    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1: 
                result[key] = torch.stack(result[key])
            else:
                raise ValueError(f"Batch contains spectrograms of different shapes: {shapes}. Consider using drop_last=True in DataLoader.")
        
    return result

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2, 
            drop_path_rate=0.2 
        )
        
        if hasattr(self.backbone, 'classifier'):
            backbone_out_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            try:
                backbone_out_features = self.backbone.get_classifier().in_features
                self.backbone.reset_classifier(0, '')
            except AttributeError:
                raise ValueError(f"Unknown backbone classifier type for model: {cfg.model_name}")

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(backbone_out_features, cfg.num_classes)
            
    def forward(self, x, targets=None):
        is_training_and_targets_provided = self.training and targets is not None
        
        if is_training_and_targets_provided:
            apply_mixup = random.random() < self.cfg.mixup_prob
            apply_cutmix = random.random() < self.cfg.cutmix_prob

            if apply_mixup and not apply_cutmix:
                x, targets_a, targets_b, lam = mixup_data(x, targets, self.cfg.mix_alpha, x.device)
            elif apply_cutmix and not apply_mixup:
                x, targets_a, targets_b, lam = cutmix_data(x, targets, self.cfg.mix_alpha, is_horizontal_only=True)
            elif apply_mixup and apply_cutmix:
                if random.random() < 0.5: 
                    x, targets_a, targets_b, lam = cutmix_data(x, targets, self.cfg.mix_alpha, is_horizontal_only=True)
                else:
                    x, targets_a, targets_b, lam = mixup_data(x, targets, self.cfg.mix_alpha, x.device)
            else: 
                targets_a, targets_b, lam = None, None, None
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)
        
        if isinstance(features, dict) and 'features' in features:
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if self.training and is_training_and_targets_provided and (targets_a is not None):
            loss = mixup_criterion(F.binary_cross_entropy_with_logits,
                                   logits, targets_a, targets_b, lam)
            return logits, loss
            
        return logits 
            
def get_optimizer(model, cfg):
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"옵티마이저 {cfg.optimizer}는 구현되지 않았습니다.")
        
    return optimizer

def get_scheduler(optimizer, cfg, steps_per_epoch=None):
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        if steps_per_epoch is None:
            raise ValueError("OneCycleLR 스케줄러를 사용하려면 steps_per_epoch를 제공해야 합니다.")
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.epochs,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=10000
        )
    else:
        scheduler = None
        
    return scheduler

def get_criterion(cfg):
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"손실 함수 {cfg.criterion}는 구현되지 않았습니다.")
        
    return criterion

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, epoch_idx=0):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch_idx+1} Training")
    
    for step, batch in pbar:
        inputs = batch['melspec'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        outputs_with_loss = model(inputs, targets)

        if isinstance(outputs_with_loss, tuple):
            outputs, loss = outputs_with_loss
        else:
            outputs = outputs_with_loss
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        
        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss.item())
        
        pbar.set_postfix({
            'train_loss': np.mean(losses[-cfg.print_freq:]) if len(losses) >= cfg.print_freq else np.mean(losses),
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            
            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss.item())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    aucs = []
    
    probs = 1 / (1 + np.exp(-outputs))
    
    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0 and len(np.unique(targets[:, i])) > 1:
            try:
                class_auc = roc_auc_score(targets[:, i], probs[:, i])
                aucs.append(class_auc)
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0

def run_training(df, cfg, trial=None):
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)
    
    if trial:
        cfg.lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        cfg.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        cfg.gaussian_noise_prob = trial.suggest_float('gaussian_noise_prob', 0.2, 0.6)
        cfg.spec_masking_prob = trial.suggest_float('spec_masking_prob', 0.5, 0.9)
        cfg.spec_brightness_contrast_prob = trial.suggest_float('spec_brightness_contrast_prob', 0.3, 0.7)
        cfg.cutmix_prob = trial.suggest_float('cutmix_prob', 0.1, 0.5)
        scheduler_name = trial.suggest_categorical('scheduler', ['OneCycleLR', 'CosineAnnealingLR'])
        cfg.scheduler = scheduler_name
        if scheduler_name == 'CosineAnnealingLR':
            cfg.min_lr = trial.suggest_loguniform('min_lr', 1e-7, 1e-5)

        print(f"\n--- Optuna Trial {trial.number} ---")
        print(f"Suggested LR: {cfg.lr:.6f}, Batch Size: {cfg.batch_size}")
        print(f"Suggested GN_Prob: {cfg.gaussian_noise_prob:.2f}, Mask_Prob: {cfg.spec_masking_prob:.2f}")
        print(f"Suggested Bright_Cont_Prob: {cfg.spec_brightness_contrast_prob:.2f}, CutMix_Prob: {cfg.cutmix_prob:.2f}")
        print(f"Suggested Scheduler: {cfg.scheduler}")
        if cfg.scheduler == 'CosineAnnealingLR':
            print(f"Suggested Min_LR: {cfg.min_lr:.6f}")

    if cfg.debug:
        cfg.update_debug_settings()

    # 미리 계산된 멜 스펙트로그램 메타데이터 CSV 로드
    if cfg.LOAD_DATA_FROM_NPY:
        print(f"미리 계산된 멜 스펙트로그램 메타데이터를 CSV 파일({cfg.spectrogram_metadata_csv})에서 로드 중...")
        try:
            full_data_df = pd.read_csv(cfg.spectrogram_metadata_csv)
            print(f"미리 계산된 멜 스펙트로그램 메타데이터 {len(full_data_df)}개 로드 완료.")
            
            unique_files_df = full_data_df[['original_filename', 'primary_label']].drop_duplicates().reset_index(drop=True)
            
            df_for_kfold_split = unique_files_df # KFold 분할의 'X'와 'y'에 전달할 DataFrame
            print(f"총 {len(df_for_kfold_split)}개의 원본 파일 기준으로 폴드 분할을 준비합니다.")

        except Exception as e:
            print(f"미리 계산된 스펙트로그램 메타데이터 로드 오류: {e}")
            print("대신 오디오 파일에서 즉석에서 스펙트로그램을 생성합니다. (LOAD_DATA_FROM_NPY=False)")
            cfg.LOAD_DATA_FROM_NPY = False # 로드 실패 시 즉석 생성으로 전환
            df_for_kfold_split = df.copy() # 즉석 생성 모드에서는 원본 df 사용
            # original_filename이 없으므로 filename을 사용
            df_for_kfold_split = df_for_kfold_split.rename(columns={'filename': 'original_filename'})
            df_for_kfold_split['samplename'] = df_for_kfold_split.original_filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    else:
        # LOAD_DATA_FROM_NPY가 False일 때, 원본 train_df를 사용
        df_for_kfold_split = df[['filename', 'primary_label']].copy()
        df_for_kfold_split = df_for_kfold_split.rename(columns={'filename': 'original_filename'})
        df_for_kfold_split['samplename'] = df_for_kfold_split.original_filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        print(f"오디오 파일에서 즉석 생성 모드: {len(df_for_kfold_split)}개의 원본 오디오 파일로 학습 준비.")


    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    
    overall_best_aucs = []
    
    folds_to_run = cfg.selected_folds if not cfg.debug else [0]

    # df_for_kfold_split를 사용하여 분할 (X = 파일명, y = primary_label)
    for fold in folds_to_run:
        # unique_files_df의 인덱스를 사용하여 train_idx와 val_idx를 얻습니다.
        train_file_idx, val_file_idx = list(skf.split(df_for_kfold_split['original_filename'], df_for_kfold_split['primary_label']))[fold]
        
        # 이제 실제 데이터프레임 (full_data_df 또는 원본 df)에서 해당 파일에 속하는 모든 세그먼트를 선택해야 합니다.
        train_files = df_for_kfold_split.iloc[train_file_idx]['original_filename'].tolist()
        val_files = df_for_kfold_split.iloc[val_file_idx]['original_filename'].tolist()
        
        # full_data_df (전체 세그먼트 데이터)에서 해당 파일에 속하는 세그먼트만 필터링합니다.
        # 즉석 생성 모드일 경우 df (원본 train_df)를 사용해야 합니다.
        source_df = full_data_df if cfg.LOAD_DATA_FROM_NPY else df
        
        train_df = source_df[source_df['original_filename'].isin(train_files)].reset_index(drop=True)
        val_df = source_df[source_df['original_filename'].isin(val_files)].reset_index(drop=True)
        
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        # 디버그 모드에서 데이터셋 샘플링은 여기서 수행 (필터링된 train_df, val_df에 적용)
        if cfg.debug:
            train_df = train_df.sample(min(cfg.train_df_debug_limit, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
            val_df = val_df.sample(min(cfg.train_df_debug_limit // 4, len(val_df)), random_state=cfg.seed).reset_index(drop=True)
            print(f"디버그 모드: 학습 세트 {len(train_df)}개 세그먼트, 검증 세트 {len(val_df)}개 세그먼트로 샘플링됨.")

        print(f'학습 세트: {len(train_df)} 세그먼트 샘플')
        print(f'검증 세트: {len(val_df)} 세그먼트 샘플')
        
        # BirdCLEFDatasetFromNPY 클래스 인스턴스화
        train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, mode='valid')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        
        if cfg.scheduler == 'OneCycleLR':
            scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
        elif cfg.scheduler == 'CosineAnnealingLR':
            scheduler = get_scheduler(optimizer, cfg)
        else:
            scheduler = get_scheduler(optimizer, cfg)
        
        best_auc = 0.0
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(cfg.epochs):
            print(f"\n에포크 {epoch+1}/{cfg.epochs}")
            
            train_loss, train_auc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cfg.device,
                scheduler=scheduler,
                epoch_idx=epoch
            )
            
            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

            if scheduler is not None and isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
                scheduler.step()
            elif scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"학습 손실: {train_loss:.4f}, 학습 AUC: {train_auc:.4f}")
            print(f"검증 손실: {val_loss:.4f}, 검증 AUC: {val_auc:.4f}")
            
            if val_auc > best_auc + cfg.early_stopping_min_delta:
                best_auc = val_auc
                best_epoch = epoch + 1
                epochs_no_improve = 0
                print(f"새로운 최고 AUC: {best_auc:.4f} (에포크 {best_epoch})")

                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg
                }, os.path.join(cfg.OUTPUT_DIR, f"{cfg.model_name}_fold{fold}_best.pth"))
            else:
                epochs_no_improve += 1
                print(f"검증 AUC 개선 없음. 현재 최고 AUC: {best_auc:.4f}. 개선 없는 에포크 수: {epochs_no_improve}/{cfg.early_stopping_patience}")
                if epochs_no_improve >= cfg.early_stopping_patience:
                    print(f"조기 종료: {cfg.early_stopping_patience} 에포크 동안 검증 AUC 개선이 없어 학습을 중단합니다.")
                    break

        overall_best_aucs.append(best_auc)
        print(f"\n폴드 {fold}의 최고 AUC: {best_auc:.4f} (에포크 {best_epoch})")
        
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("교차 검증 결과:")
    for fold_idx, score in enumerate(overall_best_aucs):
        print(f"폴드 {fold_idx}: {score:.4f}")
    print(f"평균 AUC: {np.mean(overall_best_aucs):.4f}")
    print("="*60)
    
    return np.mean(overall_best_aucs)

# --- Optuna 최적화 함수 ---
def objective(trial: optuna.Trial, df: pd.DataFrame, base_cfg: CFG):
    # Optuna에 제안할 하이퍼파라미터 정의 (여기에 필요한 파라미터 추가)
    cfg_tuned = base_cfg # 기본 cfg 복사 (파라미터 변경을 위해)
    
    # 1. 학습률 튜닝 (1e-5 ~ 1e-3 로그 스케일)
    cfg_tuned.lr = trial.suggest_loguniform('lr', 1e-5, 1e-3) 
    
    # 2. 배치 크기 튜닝 (범주형)
    cfg_tuned.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # 3. 증강 확률 튜닝
    cfg_tuned.gaussian_noise_prob = trial.suggest_float('gaussian_noise_prob', 0.2, 0.6)
    cfg_tuned.spec_masking_prob = trial.suggest_float('spec_masking_prob', 0.5, 0.9)
    cfg_tuned.spec_brightness_contrast_prob = trial.suggest_float('spec_brightness_contrast_prob', 0.3, 0.7)
    cfg_tuned.cutmix_prob = trial.suggest_float('cutmix_prob', 0.1, 0.5)

    # 4. 스케줄러 튜닝
    scheduler_name = trial.suggest_categorical('scheduler', ['OneCycleLR', 'CosineAnnealingLR'])
    cfg_tuned.scheduler = scheduler_name
    if scheduler_name == 'CosineAnnealingLR':
        cfg_tuned.min_lr = trial.suggest_loguniform('min_lr', 1e-7, 1e-5)

    print(f"\n--- Optuna Trial {trial.number} ---")
    print(f"Suggested LR: {cfg_tuned.lr:.6f}, Batch Size: {cfg_tuned.batch_size}")
    print(f"Suggested GN_Prob: {cfg_tuned.gaussian_noise_prob:.2f}, Mask_Prob: {cfg_tuned.spec_masking_prob:.2f}")
    print(f"Suggested Bright_Cont_Prob: {cfg_tuned.spec_brightness_contrast_prob:.2f}, CutMix_Prob: {cfg_tuned.cutmix_prob:.2f}")
    print(f"Suggested Scheduler: {cfg_tuned.scheduler}")
    if cfg_tuned.scheduler == 'CosineAnnealingLR':
        print(f"Suggested Min_LR: {cfg_tuned.min_lr:.6f}")

    # run_training 함수를 호출하여 학습 및 검증 AUC 반환
    mean_auc = run_training(df, cfg_tuned, trial) 

    return mean_auc

# --- 파이프라인 실행 ---
if __name__ == "__main__":
    set_seed(cfg.seed)
    
    print("\n학습 데이터 로드 중...")
    # 원본 train.csv 로드 (이 정보는 KFold 분할 및 기타 메타데이터 추출에 사용됩니다)
    train_df = pd.read_csv(cfg.train_csv) 
    
    # OUTPUT_DIR 생성 확인
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Optuna 활성화 여부에 따라 학습 방식 결정
    if cfg.use_optuna:
        print("\n--- Optuna 하이퍼파라미터 최적화 시작 ---")
        # Optuna 스터디 
        study = optuna.create_study(
            direction="maximize",
            study_name=cfg.optuna_study_name,
            storage=cfg.optuna_storage,
            load_if_exists=True # 기존 스터디가 있으면 로드하여 이어서 진행
        )
        study.optimize(lambda trial: objective(trial, train_df, cfg), n_trials=cfg.optuna_n_trials)

        print("\n--- Optuna 최적화 결과 ---")
        print(f"최적의 하이퍼파라미터: {study.best_params}")
        print(f"최고 AUC: {study.best_value:.4f}")
        print(f"최적의 트라이얼: {study.best_trial.number}")
        print("==========================================")
    else:
        print("\n--- 단일 학습 실행 시작 ---")
        final_mean_auc = run_training(train_df, cfg)
        print(f"\n단일 학습 실행 완료! 평균 AUC: {final_mean_auc:.4f}")