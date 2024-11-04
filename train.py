#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from tqdm import tqdm
import wandb

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, max_epochs, save_dir, log_dir, start_epoch=0, patience=5, accumulation_steps=8):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stopping_counter = 0
    log_file = os.path.join(log_dir, 'training_log.txt')
    
    if start_epoch == 0:
        with open(log_file, 'w') as f:
            f.write("Training log\n")
            f.write("Epoch,Train Loss,Validation Loss,Validation Accuracy\n")
    else:
        with open(log_file, 'a') as f:
            f.write(f"\nResuming training from epoch {start_epoch}\n")

    for epoch in range(start_epoch, max_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} - Training")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Normalize the loss
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
        
        train_loss /= len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        scheduler.step(val_loss)
        
        # Log the results
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f}\n")
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save model after each epoch
        model_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}_loss_{val_loss:.4f}_acc_{val_acc:.4f}.pth')
        torch.save(model_state, save_path)
        print(f"Model saved at: {save_path}")
        
        # Update best model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            early_stopping_counter = 0
            best_model_path = save_path
            print(f"New best model with validation loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}, Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")
    
    return best_val_loss, best_val_acc, best_model_path

