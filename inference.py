#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from tqdm import tqdm
import glob

def inference(model, test_loader, device):
    """
    모델을 사용해 추론을 수행하는 함수.
    
    Args:
        model: 추론할 모델.
        test_loader: 테스트 데이터를 로드할 DataLoader.
        device: 모델과 데이터를 로드할 장치 (cuda 또는 cpu).
    
    Returns:
        predictions: 모델이 예측한 결과 리스트.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def get_latest_checkpoint(checkpoint_dir):
    """
    주어진 디렉토리에서 가장 최신의 체크포인트 파일을 가져오는 함수.
    
    Args:
        checkpoint_dir: 체크포인트 파일이 저장된 디렉토리 경로.
    
    Returns:
        latest_checkpoint: 가장 최근의 체크포인트 파일 경로.
    """
    # 모든 .pth 파일을 검색
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'model_*.pth'))
    if not checkpoints:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    # 가장 최근에 수정된 파일을 선택
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint

