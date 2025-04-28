import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_cifar10_dataloaders(batch_size=128, num_workers=4, val_split=0.1, aug_params=None):
    """
    Load CIFAR-10 dataset with standard augmentations and create train/val/test dataloaders.
    
    Args:
        batch_size: Batch size for training and evaluation
        num_workers: Number of workers for data loading
        val_split: Fraction of training data to use for validation
        aug_params: Dictionary of augmentation parameters
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Set default augmentation parameters
    if aug_params is None:
        aug_params = {
            'random_crop_padding': 4,
            'random_erasing_prob': 0.2,
            'color_jitter_brightness': 0.1,
            'color_jitter_contrast': 0.1
        }
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=aug_params['random_crop_padding']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=aug_params['color_jitter_brightness'],
            contrast=aug_params['color_jitter_contrast']
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=aug_params['random_erasing_prob']),
    ])
    
    # Just normalization for validation and testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load full training set with augmentations
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=train_transform
    )
    
    # Split into training and validation sets
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override transform for validation set to use test transforms
    val_dataset.dataset.transform = test_transform
    
    # Load test set
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader