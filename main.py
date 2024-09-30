#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import wandb
from dataset import CustomDataset
from model import ConvNextModel
from train import train
from inference import inference, get_latest_checkpoint
from seed import seed_everything  # 시드 설정 함수 불러오기
from augmentation import SketchAutoAugment

def get_arguments():
    parser = argparse.ArgumentParser(description="ConvNext Training and Inference Script")
    
    # 모델 및 데이터 설정
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the CSV file for training data')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test data directory')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the CSV file for test data')
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./training_logs', help='Directory to save training logs')
    
    # 학습 설정
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and inference')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs for training')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the latest checkpoint')
    
    # 이미지 크기 설정
    parser.add_argument('--resize_height', type=int, default=448, help='Height to resize images to')
    parser.add_argument('--resize_width', type=int, default=448, help='Width to resize images to')
    
    # 시드 설정
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    
    # 시드 설정
    seed_everything(args.seed)
    
    wandb.init(project="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 경로 설정
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 데이터 로드
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    num_classes = len(train_df['target'].unique())

    # Train/Validation split
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=args.seed, stratify=train_df['target'])

    # 데이터 증강 및 변환 설정
    train_transform = A.Compose([
        SketchAutoAugment(p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=255, p=0.3),
        A.ISONoise(p=0.3),
        A.InvertImg(p=0.2),
        A.Resize(args.resize_height, args.resize_width),  # argparse 인자로 받은 이미지 크기 적용
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(args.resize_height, args.resize_width),  # argparse 인자로 받은 이미지 크기 적용
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 데이터셋 및 데이터 로더 생성
    train_dataset = CustomDataset(args.train_dir, train_df, train_transform)
    val_dataset = CustomDataset(args.train_dir, val_df, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델, 손실 함수, 옵티마이저, 스케줄러 설정
    model = ConvNextModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # 체크포인트 로드 (옵션에 따라)
    start_epoch = 0
    if args.resume_training:
        latest_checkpoint = get_latest_checkpoint(args.save_dir)
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    # 학습 실행
    best_val_loss, best_val_acc, best_model_path = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                                                         max_epochs=args.max_epochs, save_dir=args.save_dir, log_dir=args.log_dir, 
                                                         start_epoch=start_epoch, patience=args.patience, accumulation_steps=args.accumulation_steps)

    print(f"\nTraining completed. Best Validation Loss: {best_val_loss:.4f}, Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")

    # 추론
    test_dataset = CustomDataset(args.test_dir, test_df, val_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ConvNextModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    predictions = inference(model, test_loader, device)

    submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
    submission.to_csv('submission.csv', index=False)
    print("Inference completed. Submission file saved.")

    wandb.finish()

