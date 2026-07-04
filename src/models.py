import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================================
# 1. PyTorch Dataset Helper
# ==========================================
class EmergencyDataset(Dataset):
    """Custom PyTorch Dataset for loading preprocessed ambulance dispatch features."""
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==========================================
# 2. Linear/Logistic Regression Baseline Wrapper
# ==========================================
class LogisticRegressionBaseline:
    """Wrapper around scikit-learn's LogisticRegression for binary classification."""
    def __init__(self, C=0.1, max_iter=1000, random_state=42):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        self.model_type = "linear"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        probas = self.predict_proba(X)
        preds = (probas >= 0.5).astype(int)
        
        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, probas)
        except ValueError:
            auc = 0.5
            
        return {"accuracy": acc, "auc_roc": auc}


# ==========================================
# 3. FCNN with Integer Encoding
# ==========================================
class FCNNIntegerNet(nn.Module):
    """
    A 3-layer fully connected neural network which treats integer-encoded categorical 
    features as continuous numerical values alongside the standardized numerical inputs.
    """
    def __init__(self, input_dim=13, hidden_dim1=64, hidden_dim2=32, dropout=0.2):
        super(FCNNIntegerNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim2, 1) # Outputs logits
        )
        
    def forward(self, x):
        return self.network(x).squeeze(-1)


# ==========================================
# 4. FCNN with Embedding Layers (Best DL Approach)
# ==========================================
class FCNNEmbeddingNet(nn.Module):
    """
    An advanced deep neural network that uses PyTorch Embedding layers (nn.Embedding)
    to learn low-dimensional, dense representations of categorical variables, 
    and concatenates them with continuous numerical features.
    """
    def __init__(self, 
                 categorical_dims: dict, 
                 num_continuous_features: int = 6,
                 hidden_dim1: int = 64, 
                 hidden_dim2: int = 32, 
                 dropout: float = 0.2):
        super(FCNNEmbeddingNet, self).__init__()
        
        self.embeddings = nn.ModuleList()
        self.cat_cols_order = list(categorical_dims.keys())
        
        total_embed_dim = 0
        for col_name in self.cat_cols_order:
            vocab_size = categorical_dims[col_name]
            # Standard heuristic for embedding dimension size: min(50, (vocab_size + 1) // 2)
            embed_dim = max(2, min(50, (vocab_size + 1) // 2))
            
            self.embeddings.append(nn.Embedding(vocab_size, embed_dim))
            total_embed_dim += embed_dim
            
        # Overall input dimensions = embedded categorical dims + numerical features count
        combined_input_dim = total_embed_dim + num_continuous_features
        
        self.fc_network = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim2, 1) # Outputs logits
        )
        
    def forward(self, x):
        # Continuous numerical variables are the first 6 columns of x
        continuous_x = x[:, :6]
        
        # Categorical variables are the remaining columns
        categorical_x = x[:, 6:].long()
        
        embedded_list = []
        for i, embed_layer in enumerate(self.embeddings):
            # Take index column, apply embedding layer
            embedded_list.append(embed_layer(categorical_x[:, i]))
            
        # Concatenate all embeddings and continuous inputs
        combined_x = torch.cat([continuous_x] + embedded_list, dim=1)
        
        return self.fc_network(combined_x).squeeze(-1)


# ==========================================
# 5. Trainer Wrapper for PyTorch Models
# ==========================================
class PyTorchModelTrainer:
    """Helper wrapper to handle the training, evaluation, saving and loading of PyTorch models."""
    def __init__(self, model: nn.Module, model_type: str, lr=0.001, weight_decay=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_type = model_type
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def fit_loader(self, 
                   train_loader: DataLoader, 
                   val_loader: DataLoader, 
                   epochs=50, 
                   early_stopping_patience=7) -> dict:
        """
        Trains the neural network using train and validation DataLoader.
        Implements early stopping on validation loss.
        """
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        for epoch in range(1, epochs + 1):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_train += (preds == y_batch).sum().item()
                total_train += X_batch.size(0)
                
            epoch_train_loss = train_loss / total_train
            epoch_train_acc = correct_train / total_train
            
            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    
                    val_loss += loss.item() * X_batch.size(0)
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    correct_val += (preds == y_batch).sum().item()
                    total_val += X_batch.size(0)
                    
            epoch_val_loss = val_loss / total_val
            epoch_val_acc = correct_val / total_val
            
            # Record metrics
            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_acc"].append(epoch_val_acc)
            
            # Early Stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                # print(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Restore best weights
        if best_model_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_weights.items()})
            
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs probabilities of delay risk."""
        self.model.eval()
        dataset = EmergencyDataset(X)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        all_probas = []
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probas = torch.sigmoid(outputs)
                all_probas.extend(probas.cpu().numpy())
                
        return np.array(all_probas)

    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        probas = self.predict_proba(X)
        preds = (probas >= 0.5).astype(int)
        
        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, probas)
        except ValueError:
            auc = 0.5
            
        return {"accuracy": acc, "auc_roc": auc}

    def save_model(self, path: str):
        """Saves model state dictionary and type details."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "hyperparameters": {
                # Save essential layers details
            }
        }
        # If FCNNEmbeddingNet, we can also store categorical_dims on model class
        if hasattr(self.model, "categorical_dims"):
            checkpoint["categorical_dims"] = self.model.categorical_dims
        if hasattr(self.model, "cat_cols_order"):
            checkpoint["cat_cols_order"] = self.model.cat_cols_order
            
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        """Loads model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return self


def load_pytorch_model_checkpoint(path: str, categorical_dims: dict = None) -> PyTorchModelTrainer:
    """Factory helper to load and reconstruct PyTorch models from their checkpoint file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint["model_type"]
    
    if model_type == "integer_fcnn":
        model = FCNNIntegerNet(input_dim=13)
    elif model_type == "embedding_fcnn":
        dims = checkpoint.get("categorical_dims", categorical_dims)
        if dims is None:
            raise ValueError("Categorical dimensions are required to load FCNN Embedding Net.")
        model = FCNNEmbeddingNet(categorical_dims=dims)
        model.categorical_dims = dims
    else:
        raise ValueError(f"Unknown PyTorch model type in checkpoint: {model_type}")
        
    model.load_state_dict(checkpoint["model_state_dict"])
    trainer = PyTorchModelTrainer(model, model_type)
    return trainer


if __name__ == '__main__':
    print("Testing Models Module...")
    # 1. Test Linear Baseline
    X_fake = np.random.randn(100, 26)
    y_fake = np.random.randint(0, 2, 100)
    lr_model = LogisticRegressionBaseline()
    lr_model.fit(X_fake, y_fake)
    res_lr = lr_model.evaluate(X_fake, y_fake)
    print("Logistic Regression Eval:", res_lr)
    
    # 2. Test Integer FCNN
    X_fake_int = np.random.randn(100, 13)
    int_net = FCNNIntegerNet()
    trainer_int = PyTorchModelTrainer(int_net, "integer_fcnn")
    
    dataset = EmergencyDataset(X_fake_int, y_fake)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    history = trainer_int.fit_loader(loader, loader, epochs=3)
    print("Integer FCNN trained successfully! Loss hist:", history["train_loss"])
    
    # 3. Test Embedding FCNN
    cat_dims = {'priority_level': 5, 'incident_type': 7, 'time_of_day': 5, 
                'weather_conditions': 5, 'day_of_week': 3, 'traffic_density': 4, 'dispatch_zone': 6}
    X_fake_embed = np.random.randn(100, 13)
    # Ensure categorical parts of fake dataset are valid positive integers
    X_fake_embed[:, 6:] = np.random.randint(1, 3, size=(100, 7))
    
    embed_net = FCNNEmbeddingNet(categorical_dims=cat_dims)
    embed_net.categorical_dims = cat_dims
    trainer_embed = PyTorchModelTrainer(embed_net, "embedding_fcnn")
    
    dataset_embed = EmergencyDataset(X_fake_embed, y_fake)
    loader_embed = DataLoader(dataset_embed, batch_size=16, shuffle=True)
    
    history_embed = trainer_embed.fit_loader(loader_embed, loader_embed, epochs=3)
    print("Embedding FCNN trained successfully! Loss hist:", history_embed["train_loss"])
    
    print("All model code loaded and verified successfully!")
