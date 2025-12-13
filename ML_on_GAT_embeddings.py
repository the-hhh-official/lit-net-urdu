import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ======================
# CONFIG
# ======================

FINGERPRINT_CSV = "gat_fingerprints.csv"   # Your good GAT output

# Autoencoder hyperparameters
AE_HIDDEN_DIM = 64
AE_LATENT_DIM = 16
AE_EPOCHS = 200
AE_BATCH_SIZE = 16
AE_LR = 1e-3
AE_NOISE_STD = 0.10      # noise added in latent space
N_AUG_PER_REAL = 5       # synthetic samples per real training book

RANDOM_STATE_SPLIT = 35


# ======================
# 1. LOAD FINGERPRINTS
# ======================

def load_fingerprints(path):
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    X = df[emb_cols].values.astype(np.float32)
    authors = df["author"].astype(str).tolist()

    print(f"[LOAD] Loaded {df.shape[0]} books, feature dim = {X.shape[1]}")
    print(f"[LOAD] Authors:", sorted(set(authors)))
    return df, X, authors


# ======================
# 2. AUTHOR-AWARE SPLIT
# ======================

def make_fixed_author_split(authors, random_state=35):
    rng = np.random.RandomState(random_state)

    author_to_indices = defaultdict(list)
    for idx, a in enumerate(authors):
        author_to_indices[a].append(idx)

    test_idx = []
    train_idx = []

    print("\n[SPLIT] Building fixed author-aware split:")
    print("        Exactly 1 held-out test book per author.")

    for a, idxs in author_to_indices.items():
        idxs = np.array(idxs)
        if len(idxs) < 2:
            raise ValueError(f"Author '{a}' has < 2 books, cannot split.")

        rng.shuffle(idxs)
        this_test = idxs[0]
        this_train = idxs[1:]

        test_idx.append(this_test)
        train_idx.extend(this_train.tolist())

        print(f"        Author '{a}': test=1 book, train={len(this_train)} books")

    return np.array(train_idx), np.array(test_idx)


# ======================
# 3. AUTOENCODER
# ======================

class EmbeddingAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_ae(X_train, device):
    """
    Train AE on real training fingerprints only.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train_t), batch_size=AE_BATCH_SIZE, shuffle=True)

    model = EmbeddingAE(X_train.shape[1], AE_HIDDEN_DIM, AE_LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\n[AE] Training Autoencoder...")
    model.train()
    for epoch in range(1, AE_EPOCHS + 1):
        total_loss = 0
        n = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            n += batch_x.size(0)

        if epoch % 20 == 0 or epoch == 1 or epoch == AE_EPOCHS:
            print(f"[AE] Epoch {epoch}/{AE_EPOCHS} - Recon Loss: {total_loss/n:.6f}")

    return model


def generate_synthetic(X_train, y_train, ae_model, device):
    """
    Create synthetic embeddings by perturbing the AE latent space.
    """
    ae_model.eval()
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)

    X_syn, y_syn = [], []

    print("\n[AE] Generating synthetic embeddings...")

    with torch.no_grad():
        for i in range(X_train.shape[0]):
            x = X_train_t[i:i+1]

            for _ in range(N_AUG_PER_REAL):
                _, z = ae_model(x)
                z_noisy = z + AE_NOISE_STD * torch.randn_like(z)
                x_hat = ae_model.decoder(z_noisy)

                X_syn.append(x_hat.cpu().numpy()[0])
                y_syn.append(y_train[i])

    X_syn = np.array(X_syn, dtype=np.float32)
    y_syn = np.array(y_syn)

    print(f"[AE] Synthetic samples generated: {len(X_syn)}")
    return X_syn, y_syn


# ======================
# 4. LOGISTIC REGRESSION ONLY (NO RF)
# ======================

def supervised_gat_ae_lr(X, authors):
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(authors)
    class_names = le.classes_

    # Split
    train_idx, test_idx = make_fixed_author_split(authors, RANDOM_STATE_SPLIT)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("\n[DATA] Train:", X_train.shape, " Test:", X_test.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train AE
    ae_model = train_ae(X_train, device)

    # Generate synthetic embeddings
    X_syn, y_syn = generate_synthetic(X_train, y_train, ae_model, device)

    # Combine real + synthetic
    X_aug = np.concatenate([X_train, X_syn], axis=0)
    y_aug = np.concatenate([y_train, y_syn], axis=0)

    print(f"[DATA] Augmented train: {X_aug.shape}")

    # Scale
    scaler = StandardScaler()
    X_aug_s = scaler.fit_transform(X_aug)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression
    print("\n=== Logistic Regression (AE-Augmented GAT Fingerprints) ===")
    logreg = LogisticRegression(
        multi_class="multinomial",
        max_iter=600,
        solver="lbfgs"
    )
    logreg.fit(X_aug_s, y_aug)

    # Evaluate
    y_pred = logreg.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"[RESULT] Test Accuracy = {test_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       index=class_names, columns=class_names))


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    print("=== BEST PIPELINE: GAT Fingerprints + AE Augmentation + Logistic Regression ===")
    df, X, authors = load_fingerprints(FINGERPRINT_CSV)
    supervised_gat_ae_lr(X, authors)
