import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# ======================
# CONSTANTS & SETTINGS
# ======================

META_EDGES_PATH = "books_metadata - w0.csv"  # edges + author
META_NODE_PATH  = "features_metadata.csv"    # node features + author

# columns in w0
META0_COL_BOOK_ID = "book_id"
META0_COL_EDGES   = "edges_path"
META0_COL_AUTHOR  = "author"

# columns in node-features metadata
META5_COL_BOOK_ID    = "book_id"
META5_COL_AUTHOR     = "author"
META5_COL_NODE_FEAT  = "node_features_path"

NODE_COL_CHAR   = "Character"
NODE_COL_GENDER = "Gender"
NODE_COL_ROLE   = "Role"

# One-hot categories
GENDER_CATS = ["Male", "Female", "Unknown"]
ROLE_CATS   = ["Protagonist", "Antagonist", "Support",
               "Major-Support", "Narrator", "Minor", "Inactive"]

GENDER_MAP = {g: i for i, g in enumerate(GENDER_CATS)}
ROLE_MAP   = {r: i for i, r in enumerate(ROLE_CATS)}

# ======================
# AE (Autoencoder) SETTINGS
# ======================

AE_HIDDEN_DIM = 64      # hidden layer size of AE
AE_LATENT_DIM = 16      # bottleneck size
AE_EPOCHS = 50          # AE training epochs
AE_BATCH_SIZE = 64      # batch size for AE training
AE_LR = 1e-3            # learning rate for AE
AE_NOISE_STD = 0.10     # std of Gaussian noise in latent space
N_AUG_PER_GRAPH = 4     # how many synthetic graphs per real train graph


# ======================
# ONE-HOT ENCODING
# ======================

def encode_one_hot(gender: str, role: str):
    g_idx = GENDER_MAP.get(gender, GENDER_MAP["Unknown"])
    r_idx = ROLE_MAP.get(role, ROLE_MAP["Minor"])

    g_vec = np.zeros(len(GENDER_CATS), dtype=np.float32)
    r_vec = np.zeros(len(ROLE_CATS), dtype=np.float32)

    g_vec[g_idx] = 1
    r_vec[r_idx] = 1

    return np.concatenate([g_vec, r_vec], axis=0)


# ======================
# LOAD METADATA
# ======================

def load_metadata():
    print("[META] Loading metadata...")
    meta_edges = pd.read_csv(META_EDGES_PATH)
    meta_nodes = pd.read_csv(META_NODE_PATH)

    merged = pd.merge(
        meta_edges[[META0_COL_BOOK_ID, META0_COL_EDGES, META0_COL_AUTHOR]],
        meta_nodes[[META5_COL_BOOK_ID, META5_COL_NODE_FEAT, META5_COL_AUTHOR]],
        left_on=[META0_COL_BOOK_ID, META0_COL_AUTHOR],
        right_on=[META5_COL_BOOK_ID, META5_COL_AUTHOR],
        how="inner"
    )

    print(f"[META] Edges metadata rows: {len(meta_edges)}")
    print(f"[META] Node-features metadata rows: {len(meta_nodes)}")
    print(f"[META] Merged rows (books used): {len(merged)}")

    book_ids   = merged[META0_COL_BOOK_ID].tolist()
    edge_paths = merged[META0_COL_EDGES].tolist()
    node_paths = merged[META5_COL_NODE_FEAT].tolist()
    authors    = merged[META0_COL_AUTHOR].tolist()

    return merged, book_ids, edge_paths, node_paths, authors


# ======================
# BUILD GRAPHS
# ======================

def load_node_feature_dict(path):
    df = pd.read_csv(path)
    g, r = {}, {}
    for _, row in df.iterrows():
        ch = str(row[NODE_COL_CHAR])
        g[ch] = str(row[NODE_COL_GENDER])
        r[ch] = str(row[NODE_COL_ROLE])
    return g, r


def build_nx_graph(edge_path, node_feat_path):
    df_edges = pd.read_csv(edge_path)
    gender_dict, role_dict = load_node_feature_dict(node_feat_path)
    G = nx.Graph()

    for _, row in df_edges.iterrows():
        u, v = str(row["Character_A"]), str(row["Character_B"])
        w = float(row["Weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    # ensure all characters from node CSV exist as nodes
    for ch in gender_dict:
        if not G.has_node(ch):
            G.add_node(ch)

    # attach node features
    for node in G.nodes():
        g = gender_dict.get(node, "Unknown")
        r = role_dict.get(node, "Minor")
        G.nodes[node]["x"] = encode_one_hot(g, r)

    return G


def build_all_graphs(edge_paths, node_paths):
    print("[GRAPH] Building NetworkX graphs...")
    graphs = []
    total = len(edge_paths)
    for i, (e, n) in enumerate(zip(edge_paths, node_paths)):
        print(f"[GRAPH] Building graph {i+1}/{total}...", end="\r")
        G = build_nx_graph(e, n)
        graphs.append(G)
    print(f"\n[GRAPH] Done. Built {len(graphs)} graphs.")
    return graphs


# ======================
# AUTHOR STATS & SPLIT
# ======================

def print_author_stats(authors):
    print("\n[STATS] Author book counts:")
    author_to_indices = defaultdict(list)
    for idx, a in enumerate(authors):
        author_to_indices[a].append(idx)

    counts = {a: len(idxs) for a, idxs in author_to_indices.items()}
    for a, c in counts.items():
        print(f"    {a}: {c} books")

    min_books = min(counts.values())
    print(f"[STATS] Minimum #books for any author = {min_books}")
    return counts, min_books


def make_fixed_author_split(authors, random_state=42, verbose=False):
    """
    Fixed author-aware split:

      - For each author:
          * Randomly choose exactly 1 book as TEST.
          * All remaining books form that author's TRAIN POOL.
      - The chosen test book is never used in training (no leakage).
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
        this_train = idxs[1:]     # remaining books = train pool

        test_idx.append(this_test)
        train_idx.extend(this_train.tolist())

        if verbose:
            print(f"        Author '{a}': test=1 book, train={len(this_train)} books")

    test_idx = np.array(sorted(test_idx))
    train_idx = np.array(sorted(train_idx))

    if verbose:
        print(f"[SPLIT] Total train graphs: {len(train_idx)}")
        print(f"[SPLIT] Total test graphs:  {len(test_idx)}\n")

    return train_idx, test_idx, author_to_indices


# ======================
# FEATURE AUTOENCODER
# ======================

class FeatureAE(nn.Module):
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


def train_feature_ae(train_graphs, in_dim, device):
    """
    Train a simple MLP autoencoder on node features x from all training graphs.
    Returns the trained AE model.
    """
    # collect all node features from train graphs
    xs = []
    for data in train_graphs:
        xs.append(data.x)
    X_all = torch.cat(xs, dim=0)   # shape: (total_nodes_across_train_graphs, in_dim)

    dataset = TensorDataset(X_all)
    loader = TorchDataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    model = FeatureAE(in_dim, hidden_dim=AE_HIDDEN_DIM, latent_dim=AE_LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\n[AE] Training feature autoencoder on train node features...")
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
        if epoch % 10 == 0 or epoch == 1 or epoch == AE_EPOCHS:
            print(f"[AE] Epoch {epoch:03d}/{AE_EPOCHS}  Recon Loss: {avg_loss:.6f}")

    print("[AE] Done training autoencoder.")
    return model


def generate_augmented_graphs(train_graphs, ae_model, device,
                              n_aug_per_graph=N_AUG_PER_GRAPH,
                              noise_std=AE_NOISE_STD):
    """
    For each real training graph, generate n_aug_per_graph synthetic graphs
    by encoding node features, adding noise in latent space, decoding back.

    Structure (edge_index) and labels (y) are preserved.
    Only node features x are perturbed in a learned way.
    """
    from torch_geometric.data import Data

    ae_model.eval()
    aug_graphs = []

    print(f"\n[AE] Generating synthetic graphs: {n_aug_per_graph} per real train graph...")
    with torch.no_grad():
        for idx, data in enumerate(train_graphs):
            x = data.x.to(device)  # (num_nodes, in_dim)

            for _ in range(n_aug_per_graph):
                # encode
                _, z = ae_model(x)  # (num_nodes, latent_dim)
                # add noise
                z_noisy = z + noise_std * torch.randn_like(z)
                # decode
                x_tilde = ae_model.decoder(z_noisy)  # (num_nodes, in_dim)

                # build synthetic Data object
                new_data = Data(
                    x=x_tilde.cpu(),
                    edge_index=data.edge_index.clone(),
                    y=data.y.clone()
                )
                aug_graphs.append(new_data)

    print(f"[AE] Generated {len(aug_graphs)} synthetic graphs.")
    return aug_graphs


# ======================
# GAT FINGERPRINT PIPELINE
# ======================

def gat_build_fingerprints(out_np="gat_fingerprints.npy",
                           out_csv="gat_fingerprints.csv"):
    """
    Build book-level fingerprints with GAT, save them,
    and return (Z, y, book_ids, authors, label_encoder).

    METHOD:

      - For each author, pick exactly ONE fixed test book (never used in training).
      - All other books per author are REAL training books.
      - Train an autoencoder on node features of REAL training graphs.
      - Generate synthetic training graphs by perturbing AE latent space.
      - Train GAT on REAL + SYNTHETIC training graphs.
      - Evaluate ONLY on REAL held-out test books.
    """
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
    from sklearn.preprocessing import LabelEncoder

    # reproducibility-ish
    torch.manual_seed(42)
    np.random.seed(42)

    # 1) Load metadata and graphs
    meta, book_ids, edge_paths, node_paths, authors = load_metadata()
    counts, min_books = print_author_stats(authors)

    graphs = build_all_graphs(edge_paths, node_paths)

    # 2) Convert to PyG Data objects
    print("\n[PYG] Converting NetworkX graphs to PyG Data objects...")

    def nx_to_pyg(G, label_idx):
        node_list = list(G.nodes())
        idx_map = {n: i for i, n in enumerate(node_list)}

        # edges (undirected → both directions)
        edges = []
        for u, v in G.edges():
            iu = idx_map[u]
            iv = idx_map[v]
            edges.append([iu, iv])
            edges.append([iv, iu])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # node features
        X = [G.nodes[n]["x"] for n in node_list]
        x = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)

        y = torch.tensor([label_idx], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(authors)

    pyg_data_list = []
    total_graphs = len(graphs)
    for i, G in enumerate(graphs):
        print(f"[PYG] Converting graph {i+1}/{total_graphs}...", end="\r")
        pyg_data_list.append(nx_to_pyg(G, y[i]))
    print(f"\n[PYG] Done. Built {len(pyg_data_list)} PyG graphs.")

    in_dim = pyg_data_list[0].x.shape[1]
    num_classes = len(label_encoder.classes_)

    # 3) GAT model with dropout
    class GATGraphClassifier(nn.Module):
        def __init__(self, in_dim, hidden_dim=32, out_dim=64,
                     heads=4, num_classes=10, dropout=0.6):
            super().__init__()
            self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
            self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=True)
            self.classifier = nn.Linear(out_dim, num_classes)
            self.dropout = dropout

        def forward(self, x, edge_index, batch):
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.gat2(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x_graph = global_mean_pool(x, batch)  # fingerprint
            logits = self.classifier(x_graph)
            return logits, x_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATGraphClassifier(
        in_dim,
        hidden_dim=32,
        out_dim=64,
        heads=4,
        num_classes=num_classes,
        dropout=0.6
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 4) Fixed author-aware split (1 test book per author)
    train_idx, test_idx, _ = make_fixed_author_split(
        authors,
        random_state=42,
        verbose=True
    )

    real_train_graphs = [pyg_data_list[i] for i in train_idx]
    test_dataset      = [pyg_data_list[i] for i in test_idx]

    # 5) Train autoencoder on node features of REAL training graphs
    ae_model = train_feature_ae(real_train_graphs, in_dim=in_dim, device=device)

    # 6) Generate synthetic training graphs via AE
    aug_graphs = generate_augmented_graphs(real_train_graphs, ae_model, device)

    # 7) Final training dataset = real + synthetic
    train_dataset = real_train_graphs + aug_graphs

    train_loader = GeoDataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader  = GeoDataLoader(test_dataset,  batch_size=8, shuffle=False)

    # 8) Train GAT
    EPOCHS = 100
    print("\n[TRAIN] Starting GAT training with AE-augmented training set "
          "and fixed real test set...")

    for epoch in range(1, EPOCHS + 1):
        # --- training ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pred = logits.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.num_graphs

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc  = correct / total if total > 0 else 0.0

        # --- evaluation on fixed real test set ---
        model.eval()
        correct_t = 0
        total_t = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index, batch.batch)
                pred = logits.argmax(dim=1)
                correct_t += int((pred == batch.y).sum())
                total_t   += batch.num_graphs
        test_acc = correct_t / total_t if total_t > 0 else 0.0

        print(f"[TRAIN] Epoch {epoch:03d}/{EPOCHS}  "
              f"Train Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.3f}  Test Acc: {test_acc:.3f}")

    # 9) Extract fingerprints for ALL graphs
    print("\n[EMB] Extracting GAT fingerprints for all graphs...")
    all_loader = GeoDataLoader(pyg_data_list, batch_size=8, shuffle=False)

    model.eval()
    Z_list = []
    y_list = []
    for i, batch in enumerate(all_loader):
        batch = batch.to(device)
        with torch.no_grad():
            logits, z_graph = model(batch.x, batch.edge_index, batch.batch)
        Z_list.append(z_graph.cpu().numpy())
        y_list.append(batch.y.cpu().numpy())
        print(f"[EMB] Processed batch {i+1}", end="\r")
    print("\n[EMB] Done extracting fingerprints.")

    Z = np.concatenate(Z_list, axis=0)
    y_arr = np.concatenate(y_list, axis=0)

    # 10) Save fingerprints
    np.save(out_np, Z)
    df_out = pd.DataFrame(Z, columns=[f"emb_{i}" for i in range(Z.shape[1])])
    df_out.insert(0, "book_id", book_ids)
    df_out.insert(1, "author", authors)
    df_out.to_csv(out_csv, index=False)
    print(f"[SAVE] Saved fingerprints to: {out_np}")
    print(f"[SAVE] Saved CSV to:          {out_csv}")

    return Z, y_arr, book_ids, authors, label_encoder


if __name__ == "__main__":
    print("Running GAT fingerprint generation with AE-augmented training...")
    gat_build_fingerprints()
