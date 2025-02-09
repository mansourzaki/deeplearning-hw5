# Standard library imports
from pathlib import Path

# Scientific/Image processing imports
import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb

# PyTorch imports
import torch
from torch.utils.data import Dataset
import torchvision as tv

# Data handling imports
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
imgs=[]
cracked_imgs=[]
inactive_imgs=[]

def train_test_split(data, test_size=0.2, random_state=42, shuffle=True):
    """
    Split the data into training and validation sets
    Args:
        data: pandas DataFrame to split
        test_size: proportion of the dataset to include in the validation split
        random_state: random seed for reproducibility
        shuffle: whether to shuffle the data before splitting
    Returns:
        train_data, val_data: the split DataFrames
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get total number of samples
    n_samples = len(data)
    
    # Calculate number of validation samples
    n_val = int(n_samples * test_size)
    
    # Create index array
    indices = np.arange(n_samples)
    
    # Shuffle indices if requested
    if shuffle:
        np.random.shuffle(indices)
    
    # Split indices
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Create train and validation sets
    train_data = data.iloc[train_indices].reset_index(drop=True)
    val_data = data.iloc[val_indices].reset_index(drop=True)
    
    return train_data, val_data

class ChallengeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mode: str):
        """
        Initialize the dataset
        Args:
            data: pandas dataframe containing image filenames and labels
            mode: 'train' or 'val'
        """
        self.data = data
        self.mode = mode
        
        # Get base path of the project
        self.base_path = Path(__file__).parent.parent
        
        # Create transform pipeline
        self.transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ])

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a sample from the dataset
        Args:
            index: index of the sample
        Returns:
            tuple: (image, label) where image is a transformed image and label is a tensor
        """
        # Get image path and labels
        img_path = self.base_path / self.data.iloc[index, 0]  # Convert to absolute path
        crack_label = self.data.iloc[index, 1]
        inactive_label = self.data.iloc[index, 2]
        
        # Read image and convert to RGB
        img = imread(str(img_path))  # Convert Path to string
        img = gray2rgb(img)
        
        # Apply transforms
        img = self.transform(img)
        
        # Create label tensor
        label = torch.tensor([crack_label, inactive_label], dtype=torch.float32)
        
        return img, label

def imshow(img_path, crack_label, inactive_label):
    """
    Display an image with its crack and inactive status in the caption
    Args:
        img_path: path to the image
        crack_label: boolean indicating if the cell is cracked
        inactive_label: boolean indicating if the cell is inactive
    """
    # Import matplotlib here to avoid potential circular imports
    import matplotlib.pyplot as plt
    
    # Read and display the image
    img = imread(img_path)
    
    # Create figure and display image
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    
    # Create caption text
    caption = []
    if crack_label:
        caption.append("Cracked")
    if inactive_label:
        caption.append("Inactive")
    if not caption:
        caption.append("Normal")
    
    plt.title(" & ".join(caption))
    plt.axis('off')
    plt.show()