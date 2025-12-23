import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class FunkSVD(nn.Module):
    """
    Funk SVD (Simon Funk's matrix factorization model)
    
    Decomposes a ratings matrix R (n_users x n_items) into:
    R ≈ U @ V.T where U is (n_users x n_factors) and V is (n_items x n_factors)
    """
    def __init__(self, n_users, n_items, n_factors=20, init_std=0.1):
        super(FunkSVD, self).__init__()
        
        # User embeddings (latent factors)
        self.user_factors = nn.Embedding(n_users, n_factors)
        # Item embeddings (latent factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Initialize with small random values (important for gradient descent)
        nn.init.normal_(self.user_factors.weight, std=init_std)
        nn.init.normal_(self.item_factors.weight, std=init_std)
        
    def forward(self, user_ids, item_ids):
        """
        Predict ratings for given user-item pairs
        
        Args:
            user_ids: Tensor of user indices [batch_size]
            item_ids: Tensor of item indices [batch_size]
            
        Returns:
            predictions: Tensor of predicted ratings [batch_size]
        """
        # Get user and item factor vectors
        user_embedding = self.user_factors(user_ids)  # [batch_size, n_factors]
        item_embedding = self.item_factors(item_ids)  # [batch_size, n_factors]
        
        # Dot product between user and item factors
        predictions = (user_embedding * item_embedding).sum(dim=1)
        
        return predictions
    
    def predict_all(self):
        """Predict all ratings in the matrix"""
        user_factors = self.user_factors.weight  # [n_users, n_factors]
        item_factors = self.item_factors.weight  # [n_items, n_factors]
        return user_factors @ item_factors.T  # [n_users, n_items]


class FunkSVDWithBias(nn.Module):
    """
    Enhanced Funk SVD with user and item biases + global mean
    
    prediction = global_mean + user_bias + item_bias + dot(user_factors, item_factors)
    
    This often performs better in practice as it captures baseline preferences.
    """
    def __init__(self, n_users, n_items, n_factors=20, init_std=0.1):
        super(FunkSVDWithBias, self).__init__()
        
        # Latent factors
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Biases
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize
        nn.init.normal_(self.user_factors.weight, std=init_std)
        nn.init.normal_(self.item_factors.weight, std=init_std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_embedding = self.user_factors(user_ids)
        item_embedding = self.item_factors(item_ids)
        
        # Get biases
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        # Compute prediction
        dot_product = (user_embedding * item_embedding).sum(dim=1)
        predictions = self.global_bias + user_b + item_b + dot_product
        
        return predictions
    
    def predict_all(self):
        """Predict all ratings in the matrix"""
        user_factors = self.user_factors.weight  # [n_users, n_factors]
        item_factors = self.item_factors.weight  # [n_items, n_factors]
        user_biases = self.user_bias.weight.squeeze()  # [n_users]
        item_biases = self.item_bias.weight.squeeze()  # [n_items]
        
        # Compute dot products: [n_users, n_items]
        dot_products = user_factors @ item_factors.T
        
        # Add biases: broadcasting user_biases as column, item_biases as row
        predictions = self.global_bias + user_biases.unsqueeze(1) + item_biases.unsqueeze(0) + dot_products
        
        return predictions


class RatingsDataset(Dataset):
    """Dataset for user-item ratings"""
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def create_sample_data():
    """Create sample sparse rating matrix"""
    # 5 users x 5 items
    ratings_matrix = np.array([
        [5, 3, np.nan, 1, np.nan],
        [4, np.nan, np.nan, 1, np.nan],
        [1, 1, np.nan, 5, np.nan],
        [1, np.nan, np.nan, 4, np.nan],
        [np.nan, 1, 5, 4, np.nan]
    ])
    
    # Extract known ratings
    users, items = np.where(~np.isnan(ratings_matrix))
    ratings = ratings_matrix[users, items]
    
    return ratings_matrix, users, items, ratings


def create_large_sparse_data(n_users=100000, n_items=50000, n_ratings=5000000, 
                              rating_scale=(1, 5), sparsity=0.999):
    """
    Create large sparse rating matrix (simulating real-world recommender systems)
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_ratings: Number of observed ratings (controls sparsity)
        rating_scale: (min, max) rating values
        sparsity: Fraction of missing values (0.999 = 99.9% sparse, like Netflix)
    
    Returns:
        users, items, ratings arrays (NOT full matrix to save memory)
    """
    print(f"Creating large sparse dataset:")
    print(f"  Users: {n_users:,}, Items: {n_items:,}")
    print(f"  Target ratings: {n_ratings:,}")
    print(f"  Sparsity: {sparsity*100:.2f}% (density: {(1-sparsity)*100:.4f}%)")
    print(f"  Full matrix size would be: {n_users*n_items:,} entries")
    print(f"  Memory if dense: {n_users*n_items*4/(1024**3):.2f} GB (float32)")
    
    # Generate random user-item pairs
    np.random.seed(42)
    users = np.random.randint(0, n_users, size=n_ratings)
    items = np.random.randint(0, n_items, size=n_ratings)
    
    # Remove duplicates
    unique_pairs = np.unique(np.column_stack([users, items]), axis=0)
    users = unique_pairs[:, 0]
    items = unique_pairs[:, 1]
    n_unique = len(users)
    
    print(f"  Actual unique ratings: {n_unique:,}")
    actual_sparsity = 1 - (n_unique / (n_users * n_items))
    print(f"  Actual sparsity: {actual_sparsity*100:.4f}%")
    
    # Generate synthetic ratings with some structure
    # Model: rating = user_bias + item_bias + noise
    user_biases = np.random.normal(0, 0.5, n_users)
    item_biases = np.random.normal(0, 0.5, n_items)
    global_mean = (rating_scale[0] + rating_scale[1]) / 2
    
    ratings = (global_mean + 
               user_biases[users] + 
               item_biases[items] + 
               np.random.normal(0, 0.3, n_unique))
    
    # Clip to rating scale
    ratings = np.clip(ratings, rating_scale[0], rating_scale[1])
    
    return users, items, ratings.astype(np.float32), n_users, n_items


def train_funk_svd(model, train_loader, n_epochs=100, lr=0.01, weight_decay=0.02, 
                   device='cpu', verbose=True):
    """
    Train Funk SVD model
    
    Args:
        model: FunkSVD model
        train_loader: DataLoader with training data
        n_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength (this is the λ parameter)
        device: 'cpu' or 'cuda'
        verbose: Print training progress
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    # Print GPU usage info at start
    if device == 'cuda' and verbose:
        print(f"\nGPU Memory before training:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    import time
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        n_batches = 0
        epoch_start = time.time()
        
        # Progress bar for batches
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", 
                         leave=False, unit="batch")
        
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(batch_pbar):
            # Move data to device (non-blocking for faster transfer)
            user_ids = user_ids.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)
            
            # Forward pass
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/n_batches:.4f}'
            })
            
            # Print progress for first epoch to verify GPU usage
            if epoch == 0 and batch_idx == 0 and device == 'cuda' and verbose:
                print(f"\nFirst batch processed on GPU:")
                print(f"  Batch size: {len(user_ids)}")
                print(f"  GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / n_batches
        rmse = np.sqrt(avg_loss)
        loss_history.append(avg_loss)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'RMSE': f'{rmse:.4f}',
            'time': f'{epoch_time:.1f}s',
            'GPU_mem': f'{torch.cuda.memory_allocated()/1e9:.2f}GB' if device == 'cuda' else 'N/A'
        })
    
    return loss_history


def main():
    """Example usage"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Choose dataset size
    USE_LARGE_DATA = True  # Set to False for small demo
    
    if USE_LARGE_DATA:
        print("="*70)
        print("LARGE SCALE EXPERIMENT")
        print("="*70)
        
        # Create large sparse dataset
        # Netflix Prize scale: 480K users, 17K movies, 100M ratings (~98.8% sparse)
        # Let's do something substantial but not quite that big
        users, items, ratings, n_users, n_items = create_large_sparse_data(
            n_users=100000,      # 100K users
            n_items=50000,       # 50K items
            n_ratings=10000000,  # 10M ratings (~99.998% sparse)
            rating_scale=(1, 5)
        )
        
        # Split into train/test
        n_train = int(0.9 * len(ratings))
        indices = np.random.permutation(len(ratings))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        print(f"\nTrain size: {len(train_idx):,}, Test size: {len(test_idx):,}")
        
        # Create datasets
        train_dataset = RatingsDataset(users[train_idx], items[train_idx], ratings[train_idx])
        test_dataset = RatingsDataset(users[test_idx], items[test_idx], ratings[test_idx])
        
        # Larger batches for GPU efficiency
        train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=8192, shuffle=False, num_workers=4)
        
        n_factors = 64  # More factors for large dataset
        n_epochs = 20
        
    else:
        print("Creating sample rating matrix...")
        ratings_matrix, users, items, ratings = create_sample_data()
        n_users, n_items = ratings_matrix.shape
        print(f"Rating matrix shape: {ratings_matrix.shape}")
        print(f"Number of known ratings: {len(ratings)}")
        print("\nOriginal matrix (NaN = missing):")
        print(ratings_matrix)
        
        dataset = RatingsDataset(users, items, ratings)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        test_loader = None
        
        n_factors = 5
        n_epochs = 200
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Verify PyTorch can see CUDA
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("WARNING: CUDA not available! Running on CPU.")
        print("Check: 1) NVIDIA drivers, 2) CUDA toolkit, 3) PyTorch CUDA build")
        print(f"PyTorch version: {torch.__version__}")
        import sys
        print(f"Python: {sys.version}")
        response = input("Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Initialize model
    print(f"\nTraining Funk SVD with {n_factors} latent factors...")
    model = FunkSVDWithBias(n_users, n_items, n_factors=n_factors)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1e6:.2f} MB (float32)")
    
    # Train with timing
    import time
    start_time = time.time()
    
    loss_history = train_funk_svd(
        model, train_loader, 
        n_epochs=n_epochs, 
        lr=0.005,  # Lower LR for large dataset
        weight_decay=0.01,
        device=device,
        verbose=True
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    print(f"Time per epoch: {train_time/n_epochs:.2f} seconds")
    
    # Evaluate on test set if available
    if test_loader is not None:
        model.eval()
        test_loss = 0
        n_batches = 0
        
        print("\nEvaluating on test set...")
        with torch.no_grad():
            for user_ids, item_ids, ratings_batch in tqdm(test_loader, desc="Testing", unit="batch"):
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings_batch = ratings_batch.to(device)
                
                predictions = model(user_ids, item_ids)
                loss = ((predictions - ratings_batch) ** 2).mean()
                test_loss += loss.item()
                n_batches += 1
        
        test_rmse = np.sqrt(test_loss / n_batches)
        print(f"\nTest RMSE: {test_rmse:.4f}")
        
        # Sample some predictions
        print("\nSample predictions (User ID, Item ID, True Rating, Predicted):")
        sample_idx = np.random.choice(len(test_dataset), size=10, replace=False)
        with torch.no_grad():
            for idx in sample_idx:
                u, i, r = test_dataset[idx]
                pred = model(u.unsqueeze(0).to(device), i.unsqueeze(0).to(device))
                print(f"  User {u.item():6d}, Item {i.item():6d}: True={r.item():.2f}, Pred={pred.item():.2f}")
    
    else:
        # Small dataset - show full predictions
        model.eval()
        with torch.no_grad():
            predicted_matrix = model.predict_all().cpu().numpy()
        
        print("\nPredicted matrix:")
        print(np.round(predicted_matrix, 2))
        
        known_mask = ~np.isnan(ratings_matrix)
        original_known = ratings_matrix[known_mask]
        predicted_known = predicted_matrix[known_mask]
        rmse = np.sqrt(np.mean((original_known - predicted_known) ** 2))
        print(f"\nFinal RMSE on known ratings: {rmse:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Funk SVD Training Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('funk_svd_loss.png', dpi=150, bbox_inches='tight')
    print("\nLoss plot saved to 'funk_svd_loss.png'")
    plt.show()


if __name__ == "__main__":
    main()