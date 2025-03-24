import torch
import torch.nn as nn

class ClassifierEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # if input tensor is 10000 * 2 * 20, then resize it as 10000 * 40
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        # add more statistic features
        stats = torch.cat([
            x.mean(dim=1, keepdim=True),
            x.std(dim=1, keepdim=True),
            x.max(dim=1,keepdim=True).values,
            x.min(dim=1, keepdim=True).values,
        ], axis=1)
        
        return torch.cat([x, stats], dim=1)
        # return x
    
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        return x + self.block(x)

class ClassifierModel(nn.Module):
    def __init__(self, feature_num = 1, model_num = 20):
        super().__init__()

        total_dim = model_num * feature_num + 4
        self.embedding = ClassifierEmbedding()

        self.net = nn.Sequential(
            nn.Linear(total_dim, 512), 
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
            ResidualBlock(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            ResidualBlock(64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        return self.net(x)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        p_t = torch.exp(-ce_loss)
        return ((1 - p_t) ** self.gamma * ce_loss).mean()