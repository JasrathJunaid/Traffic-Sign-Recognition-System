import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class TrafficSignDataset(Dataset):
    def __init__(self, data_dir, transform=None, samples_per_class=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d)) and d != '__pycache__'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.data = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if samples_per_class:
                images = images[:samples_per_class]
            
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.data.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir, n_way=5, k_shot=5, query_size=5, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = TrafficSignDataset(data_dir, transform=transform)
    
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=0.3, random_state=42, 
        stratify=full_dataset.labels
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, full_dataset.classes

def create_episode(dataset, n_way=5, k_shot=5, query_size=5):
    classes = np.random.choice(len(dataset.classes), n_way, replace=False)
    
    support_data = []
    support_labels = []
    query_data = []
    query_labels = []
    
    for class_idx in classes:
        class_samples = [i for i, (_, label) in enumerate(dataset) 
                        if label == class_idx]
        
        if len(class_samples) < k_shot + query_size:
            continue
            
        selected_samples = np.random.choice(class_samples, k_shot + query_size, replace=False)
        
        support_samples = selected_samples[:k_shot]
        query_samples = selected_samples[k_shot:]
        
        for sample_idx in support_samples:
            img, _ = dataset[sample_idx]
            support_data.append(img)
            support_labels.append(np.where(classes == class_idx)[0][0])
        
        for sample_idx in query_samples:
            img, _ = dataset[sample_idx]
            query_data.append(img)
            query_labels.append(np.where(classes == class_idx)[0][0])
    
    if len(support_data) == 0:
        return create_episode(dataset, n_way, k_shot, query_size)
    
    support_data = torch.stack(support_data)
    support_labels = torch.tensor(support_labels)
    query_data = torch.stack(query_data)
    query_labels = torch.tensor(query_labels)
    
    return (support_data, support_labels, query_data, query_labels, classes)
