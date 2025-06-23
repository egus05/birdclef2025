# https://www.kaggle.com/code/kadircandrisolu/transforming-audio-to-mel-spec-birdclef-25 코드를 기반으로 함
import os
import cv2
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import warnings
warnings.filterwarnings("ignore")

import joblib

class Config:
    DEBUG_MODE = False
    

    OUTPUT_DIR = '/home/egus05/data_melspec_overlapping' # 아웃풋 경로
    DATA_ROOT = '/home/egus05/birdclef-2025' # 원본 BirdCLEF 데이터 경로
    FS = 32000 # 샘플링 레이트
    
    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 500
    N_MELS = 128
    FMIN = 40
    FMAX = 15000
    
    TARGET_DURATION = 5.0 #오디오 길이
    OVERLAP_DURATION = 2.5 # 겹치는 시간
    TARGET_SHAPE = (256, 256) # 멜 스펙트로그램 이미지 크기
    
    N_MAX = 50 if DEBUG_MODE else None # 디버그 모드에서 처리할 최대 파일 수
    
    # Spleeter 설정 제거

# 인스턴스 생성
config = Config()

# 출력 디렉토리 생성
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print(f"Debug mode: {'ON' if config.DEBUG_MODE else 'OFF'}")
print(f"Max files to process in debug mode: {config.N_MAX if config.N_MAX is not None else 'ALL'}")
print(f"Segment duration: {config.TARGET_DURATION}s, Overlap duration: {config.OVERLAP_DURATION}s")
print(f"Output directory for mel spectrograms: {config.OUTPUT_DIR}")

print("Loading taxonomy data...")
taxonomy_df = pd.read_csv(f'{config.DATA_ROOT}/taxonomy.csv')
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))

print("Loading training metadata...")
train_df = pd.read_csv(f'{config.DATA_ROOT}/train.csv')

label_list = sorted(train_df['primary_label'].unique())
label_id_list = list(range(len(label_list)))
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))

print(f'Found {len(label_list)} unique species')
working_df = train_df[['primary_label', 'rating', 'filename']].copy()
working_df['target'] = working_df.primary_label.map(label2id)
full_audio_root = os.path.join(config.DATA_ROOT, 'train_audio')
working_df['filepath'] = working_df['filename'].apply(
    lambda x: os.path.join(full_audio_root, x)
)
working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))

total_files_to_process = min(len(working_df), config.N_MAX or len(working_df))
print(f'Total files to process: {total_files_to_process} out of {len(working_df)} available')
print(f'Samples by class:')
print(working_df['class'].value_counts())


def audio2melspec(audio_data):
    """오디오 데이터를 정규화된 멜 스펙트로그램으로 변환합니다."""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=config.FS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_segments(audio_data, sr, target_duration, overlap_duration):
    """
    오디오 데이터를 겹치는 세그먼트로 분할합니다.
    오디오가 타겟 길이보다 짧으면 순환 패딩하여 하나의 세그먼트를 반환합니다.
    """
    segments = []
    segment_length_samples = int(target_duration * sr)
    hop_length_samples = int((target_duration - overlap_duration) * sr)

    # 오디오 데이터가 타겟 세그먼트 길이보다 짧을 경우 순환 패딩
    if len(audio_data) < segment_length_samples:
        padded_audio = np.pad(audio_data, 
                              (0, segment_length_samples - len(audio_data)), 
                              mode='wrap') # 'wrap'은 'circular'와 동일
        segments.append(padded_audio)
        return segments

    # 슬라이딩 윈도우로 세그먼트 추출
    for start_sample in range(0, len(audio_data) - segment_length_samples + 1, hop_length_samples):
        end_sample = start_sample + segment_length_samples
        segment = audio_data[start_sample:end_sample]
        segments.append(segment)
        
    return segments


print("Starting audio processing...")
print(f"{'DEBUG MODE - Processing only 50 files' if config.DEBUG_MODE else 'FULL MODE - Processing all files'}")
start_time = time.time()

processed_segments_metadata = []
errors = []
total_segments_processed = 0


for i, row in tqdm(working_df.iterrows(), total=total_files_to_process):
    if config.N_MAX is not None and i >= config.N_MAX:
        break
    
    filepath = row.filepath
    
    try:
        # 오디오 로드 및 설정된 FS로 리샘플링
        audio_data, sr_original = librosa.load(filepath, sr=None, mono=True)
        if sr_original != config.FS:
            audio_data = librosa.resample(y=audio_data, orig_sr=sr_original, target_sr=config.FS)
            
        # 오디오 파일에서 여러 겹치는 세그먼트 추출
        audio_segments_list = process_audio_segments(audio_data, 
                                                     config.FS, 
                                                     config.TARGET_DURATION, 
                                                     config.OVERLAP_DURATION)
        
        # 각 세그먼트 처리 및 파일로 저장
        for seg_idx, segment in enumerate(audio_segments_list):
            mel_spec = audio2melspec(segment)
            if mel_spec.shape != config.TARGET_SHAPE:
                mel_spec = cv2.resize(mel_spec, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            
            output_sub_dir = os.path.join(config.OUTPUT_DIR, row.primary_label)
            os.makedirs(output_sub_dir, exist_ok=True)
            
            segment_filename = f"{row.samplename}_{seg_idx}.npy"
            melspec_filepath = os.path.join(output_sub_dir, segment_filename)
            
            np.save(melspec_filepath, mel_spec.astype(np.float32))
            
            # 처리된 세그먼트의 메타데이터를 리스트에 추가
            processed_segments_metadata.append({
                'original_filename': row.filename,
                'samplename': row.samplename,
                'segment_idx': seg_idx,
                'primary_label': row.primary_label,
                'target': row.target,
                'rating': row.rating,
                'class': row['class'],
                'melspec_filepath': melspec_filepath # 저장된 멜 스펙트로그램 파일 경로
            })
            total_segments_processed += 1
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        errors.append((filepath, str(e)))

end_time = time.time()
print(f"Processing completed in {end_time - start_time:.2f} seconds")
print(f"Successfully processed {total_segments_processed} segments from {total_files_to_process} files")
print(f"Failed to process {len(errors)} files")

# 처리된 세그먼트 메타데이터를 DataFrame으로 변환 및 저장
processed_df = pd.DataFrame(processed_segments_metadata)
processed_df_path = os.path.join(config.OUTPUT_DIR, 'processed_melspec_metadata_overlapping.csv')
processed_df.to_csv(processed_df_path, index=False)
print(f"Processed mel spectrograms metadata saved to {processed_df_path}")

#시각화
import matplotlib.pyplot as plt

samples_to_display = []
displayed_original_files = set() # 원본 파일 기준으로 중복 없이 선택

max_files_to_display = min(4, len(processed_df['original_filename'].unique())) 

# 시각화를 위해 처리된 데이터에서 샘플 선택
# 각 원본 파일에서 첫 번째 세그먼트만 대표로 시각화
for idx, row in processed_df.iterrows():
    if len(samples_to_display) >= max_files_to_display:
        break
    
    if row['original_filename'] not in displayed_original_files:
        # 해당 원본 파일의 첫 번째 세그먼트 데이터
        samples_to_display.append({
            'melspec_filepath': row['melspec_filepath'],
            'class_name': id2label[row['target']],
            'species': row['primary_label'],
            'segment_idx': row['segment_idx'] # 시각화할 세그먼트 인덱스
        })
        displayed_original_files.add(row['original_filename'])

if samples_to_display:
    plt.figure(figsize=(16, 12))
    
    for i, sample_info in enumerate(samples_to_display):
        plt.subplot(2, 2, i+1)
        # 딕셔너리 키를 사용하여 멜 스펙트로그램 데이터 가져오기
        try:
            melspec_data = np.load(sample_info['melspec_filepath'])
            plt.imshow(melspec_data, aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"{sample_info['class_name']}: {sample_info['species']} (Seg {sample_info['segment_idx']})")
            plt.colorbar(format='%+2.1f') 
        except Exception as e:
            print(f"Error loading {sample_info['melspec_filepath']} for plot: {e}")
            plt.title(f"Error loading {sample_info['species']}")
    
    plt.tight_layout()
    debug_note = "debug_" if config.DEBUG_MODE else ""
    plt.savefig(f'{config.OUTPUT_DIR}/{debug_note}melspec_overlapping_examples.png') # 저장 경로 수정
    plt.show()