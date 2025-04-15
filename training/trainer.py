import torch
import time
import copy
from tqdm import tqdm
import numpy as np

# Add mixup
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs and targets'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_one_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.2):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Apply mixup
        if mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss with mixup if applied
        if mixup_alpha > 0:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimize
        optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        
        # For accuracy calculation with mixup
        if mixup_alpha > 0:
            # For mixup, use weighted accuracy of both targets
            running_corrects += (lam * torch.sum(preds == labels_a.data).item() + 
                              (1 - lam) * torch.sum(preds == labels_b.data).item())
        else:
            running_corrects += torch.sum(preds == labels.data).item()
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on the given dataloader."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=100, device='cuda', mixup_alpha=0.2):
    """
    Train and evaluate a model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        device: Device to train on
        mixup_alpha: Alpha parameter for mixup augmentation (0 to disable)
        
    Returns:
        model: Best model based on validation accuracy
        history: Dictionary containing training and validation metrics
    """
    since = time.time()
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Initialize best model and metrics
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Train phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mixup_alpha
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Evaluation phase
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history