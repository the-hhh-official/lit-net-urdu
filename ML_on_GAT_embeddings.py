import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================
# CONFIG
# ======================

FINGERPRINT_CSV = "gat_fingerprints.csv"   # output of pure GAT script
RANDOM_STATE_SPLIT = 42

# Autoencoder hyperparams
AE_HIDDEN_DIM = 64
AE_LATENT_DIM = 16
AE_EPOCHS = 200
AE_BATCH_SIZE = 16
AE_LR = 1e-3
AE_NOISE_STD = 0.10      # noise added in latent space
N_AUG_PER_REAL = 5       # synthetic samples per real training book


# ======================
# 1. LOAD FINGERPRINTS
# ======================

def load_fingerprints(path):
    """
    Expect a CSV with columns:
        book_id, author, emb_0, emb_1, ..., emb_{d-1}
    """
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    X = df[emb_cols].values.astype(np.float32)
    authors = df["author"].astype(str).tolist()
    book_ids = df["book_id"].tolist()

    print(f"[LOAD] Loaded {df.shape[0]} books, feature dim = {X.shape[1]}")
    print(f"[LOAD] Authors: {sorted(set(authors))}")
    return df, X, authors, book_ids


# ======================
# 2. AUTHOR-AWARE SPLIT
# ======================

def make_fixed_author_split(authors, random_state=42, verbose=True):
    """
    Fixed author-aware split:

      - For each author:
          * Randomly choose exactly 1 book as TEST.
          * All remaining books form TRAIN.
    """
    rng = np.random.RandomState(random_state)

    author_to_indices = defaultdict(list)
    for idx, a in enumerate(authors):
        author_to_indices[a].append(idx)

    test_idx = []
    train_idx = []

    if verbose:
        print("\n[SPLIT] Building fixed author-aware split:")
        print("        Exactly 1 held-out test book per author.")

    for a, idxs in author_to_indices.items():
        idxs = np.array(idxs)
        if len(idxs) < 2:
            raise ValueError(
                f"Author '{a}' has only {len(idxs)} book(s); "
                f"need at least 2 to hold out 1 for test and still have training data."
            )

        rng.shuffle(idxs)
        this_test = idxs[0]       # single held-out test book
        this_train = idxs[1:]     # remaining books

        test_idx.append(this_test)
        train_idx.extend(this_train.tolist())

        if verbose:
            print(f"        Author '{a}': test=1 book, train={len(this_train)} books")

    test_idx = np.array(sorted(test_idx))
    train_idx = np.array(sorted(train_idx))

    if verbose:
        print(f"[SPLIT] Total train books: {len(train_idx)}")
        print(f"[SPLIT] Total test books:  {len(test_idx)}\n")

    return train_idx, test_idx


# ======================
# 3. AUTOENCODER ON FINGERPRINTS
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


def train_ae_on_embeddings(X_train, input_dim, device):
    """
    Train AE on train fingerprints (no test data used).
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_train_t)
    loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    model = EmbeddingAE(input_dim, hidden_dim=AE_HIDDEN_DIM, latent_dim=AE_LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\n[AE] Training autoencoder on train fingerprints...")
    model.train()
    for epoch in range(1, AE_EPOCHS + 1):
        total_loss = 0.0
        total_samples = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        if epoch % 20 == 0 or epoch == 1 or epoch == AE_EPOCHS:
            print(f"[AE] Epoch {epoch:03d}/{AE_EPOCHS}  Recon Loss: {avg_loss:.6f}")

    print("[AE] Done training autoencoder.")
    return model


def generate_augmented_embeddings(X_train, y_train,
                                  ae_model,
                                  device,
                                  n_aug_per_real=N_AUG_PER_REAL,
                                  noise_std=AE_NOISE_STD):
    """
    Use AE to generate synthetic embeddings around each real train embedding.
    """
    ae_model.eval()
    X_syn_list = []
    y_syn_list = []

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)

    print(f"\n[AE] Generating synthetic embeddings: "
          f"{n_aug_per_real} per real train sample...")

    with torch.no_grad():
        for i in range(X_train.shape[0]):
            x = X_train_t[i:i+1]  # shape (1, d)

            for _ in range(n_aug_per_real):
                # encode
                _, z = ae_model(x)      # (1, latent_dim)
                # add noise
                z_noisy = z + noise_std * torch.randn_like(z)
                # decode
                x_tilde = ae_model.decoder(z_noisy)  # (1, d)

                X_syn_list.append(x_tilde.cpu().numpy()[0])
                y_syn_list.append(y_train[i])

    X_syn = np.stack(X_syn_list, axis=0)
    y_syn = np.array(y_syn_list)

    print(f"[AE] Generated {X_syn.shape[0]} synthetic embeddings.")
    return X_syn, y_syn


# ======================
# 4. SUPERVISED TRAINING WITH AE-AUGMENTED TRAINING SET
# ======================

def run_supervised_with_ae(X, authors, book_ids):
    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(authors)
    class_names = le.classes_

    # Split
    train_idx, test_idx = make_fixed_author_split(authors, random_state=RANDOM_STATE_SPLIT)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("[DATA] Train shape:", X_train.shape, " Test shape:", X_test.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Train AE on train fingerprints only ---
    ae_model = train_ae_on_embeddings(X_train, input_dim=X_train.shape[1], device=device)

    # --- Generate synthetic train embeddings ---
    X_syn, y_syn = generate_augmented_embeddings(
        X_train, y_train, ae_model, device,
        n_aug_per_real=N_AUG_PER_REAL, noise_std=AE_NOISE_STD
    )

    # Combine real + synthetic for training
    X_train_aug = np.concatenate([X_train, X_syn], axis=0)
    y_train_aug = np.concatenate([y_train, y_syn], axis=0)

    print(f"[AE] Augmented train shape: {X_train_aug.shape} "
          f"(real={X_train.shape[0]}, synthetic={X_syn.shape[0]})")

    # ---------------------------
    # 4.1 Logistic Regression
    # ---------------------------
    print("\n=== Logistic Regression with AE-augmented training ===")

    scaler = StandardScaler()
    X_train_aug_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        multi_class="multinomial",
        max_iter=500,
        solver="lbfgs",
        random_state=0
    )
    logreg.fit(X_train_aug_scaled, y_train_aug)

    y_train_pred = logreg.predict(X_train_aug_scaled)
    y_test_pred = logreg.predict(X_test_scaled)

    print(f"[LOGREG] Train accuracy: {accuracy_score(y_train_aug, y_train_pred):.3f}")
    print(f"[LOGREG] Test  accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print("\n[LOGREG] Classification report (test):")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    cm_logreg = confusion_matrix(y_test, y_test_pred)
    print("[LOGREG] Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm_logreg, index=class_names, columns=class_names))

    # ---------------------------
    # 4.2 Random Forest
    # ---------------------------
    print("\n=== Random Forest with AE-augmented training ===")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=0,
        n_jobs=-1
    )
    rf.fit(X_train_aug, y_train_aug)

    y_train_pred_rf = rf.predict(X_train_aug)
    y_test_pred_rf = rf.predict(X_test)

    print(f"[RF] Train accuracy: {accuracy_score(y_train_aug, y_train_pred_rf):.3f}")
    print(f"[RF] Test  accuracy: {accuracy_score(y_test, y_test_pred_rf):.3f}")
    print("\n[RF] Classification report (test):")
    print(classification_report(y_test, y_test_pred_rf, target_names=class_names))

    cm_rf = confusion_matrix(y_test, y_test_pred_rf)
    print("[RF] Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm_rf, index=class_names, columns=class_names))


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    print("=== Supervised ML on GAT fingerprints with AE-augmented training ===")
    df, X, authors, book_ids = load_fingerprints(FINGERPRINT_CSV)
    run_supervised_with_ae(X, authors, book_ids)
