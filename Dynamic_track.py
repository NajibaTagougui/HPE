# -------------------------------
# 1. Imports
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

# -------------------------------
# 2. Helper Functions
# -------------------------------

def get_positional_embeddings(video_shape, embed_dim):
    """
    Generate positional embeddings for video patches.
    
    Args:
        video_shape (tuple): (T, H, W) - Temporal, Height, Width
        embed_dim (int): Dimension of patch embedding
    Returns:
        pos_emb (torch.Tensor): (T*H*W, embed_dim)
    """
    T, H, W = video_shape
    # Learnable positional embeddings
    pos_emb = nn.Parameter(torch.randn(T * H * W, embed_dim))
    return pos_emb

# -------------------------------
# 3. TimeSformer Model
# -------------------------------

class TimeSformer(nn.Module):
    def __init__(self, 
                 num_classes=10,  # Simplified for demo
                 num_frames=8,
                 img_size=64,     # Reduced for faster training
                 patch_size=16,
                 embed_dim=128,
                 depth=4,
                 num_heads=4):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        # Positional Embeddings
        self.pos_emb = get_positional_embeddings(
            video_shape=(num_frames, img_size // patch_size, img_size // patch_size),
            embed_dim=embed_dim
        )
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation="gelu"
            ),
            num_layers=depth
        )
        
        # Classification Head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # ✅ Corrected from earlier errors
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W)
        Returns:
            logits (torch.Tensor): (B, num_classes)
        """
        B, C, T, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T, H_patch, W_patch)
        
        # Flatten spatial and temporal dimensions
        x = rearrange(x, 'b c t h w -> b (t h w) c')  # (B, T*H*W, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_emb.unsqueeze(0).repeat(B, 1, 1)  # (B, T*H*W, embed_dim)
        
        # Transformer
        x = self.transformer(x)  # (B, T*H*W, embed_dim)
        
        # Global average pooling over tokens
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Classification head
        logits = self.cls_head(x)  # (B, num_classes)
        
        return logits

# -------------------------------
# 4. Dummy Dataset (No External Dependencies)
# -------------------------------

class DummyVideoDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=10):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random video: 3 channels, 8 frames, 64x64 resolution
        video = torch.randn(3, 8, 64, 64)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return video, label

# -------------------------------
# 5. Training Setup
# -------------------------------

# Hyperparameters
batch_size = 4
num_epochs = 5
lr = 1e-3
num_classes = 10

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSformer(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Data Loader
dataset = DummyVideoDataset(num_classes=num_classes)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# 6. Training Loop
# -------------------------------

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in loader:
        videos, labels = batch
        videos, labels = videos.to(device), labels.to(device)
        
        # Forward pass
        logits = model(videos)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

print("Training complete!")