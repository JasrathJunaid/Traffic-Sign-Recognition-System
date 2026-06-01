import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=64):
        super(PrototypicalNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2*5*5, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
    
    def compute_prototypes(self, support_features, support_labels, n_classes):
        prototypes = torch.zeros(n_classes, support_features.size(1))
        
        for i in range(n_classes):
            mask = (support_labels == i).float()
            if mask.sum() > 0:
                prototypes[i] = (support_features * mask.unsqueeze(1)).sum(0) / mask.sum()
        
        return prototypes
    
    def compute_distances(self, query_features, prototypes):
        query_features = query_features.unsqueeze(1)
        distances = torch.sum((query_features - prototypes.unsqueeze(0)) ** 2, dim=2)
        return distances

def prototypical_loss(distances, target_labels):
    log_p_y = F.log_softmax(-distances, dim=1)
    loss = F.nll_loss(log_p_y, target_labels)
    return loss

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
