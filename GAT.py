import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

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
    if min_books < 2:
        print("[STATS] WARNING: some author has < 2 books. "
              "The 'min_books - 1' train rule will fail.")
    return counts, min_books

def make_author_split(authors, random_state=42, verbose=False):
    """
    Build train/test indices such that:
      - For each author with k books:
          * (m - 1) books go to train, where m = min_k(k)
          * remaining books go to test
      - So each author has at least 1 test book.
    """
    rng = np.random.RandomState(random_state)

    author_to_indices = defaultdict(list)
    for idx, a in enumerate(authors):
        author_to_indices[a].append(idx)

    counts = {a: len(idxs) for a, idxs in author_to_indices.items()}
    min_books = min(counts.values())

    if min_books < 2:
        raise ValueError(
            "At least one author has < 2 books; "
            "the 'min_books - 1' rule requires >= 2 books per author."
        )

    train_per_author = min_books - 1

    if verbose:
        print(f"\n[SPLIT] Using author-aware split (random_state={random_state}):")
        print(f"        train_per_author = min_books - 1 = {train_per_author}")

    train_idx = []
    test_idx  = []

    for a, idxs in author_to_indices.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        this_train = idxs[:train_per_author]
        this_test  = idxs[train_per_author:]
        train_idx.extend(this_train)
        test_idx.extend(this_test)
        if verbose:
            print(f"        Author '{a}': "
                  f"train {len(this_train)} books, test {len(this_test)} books")

    train_idx = np.array(sorted(train_idx))
    test_idx  = np.array(sorted(test_idx))

    if verbose:
        print(f"[SPLIT] Total train graphs: {len(train_idx)}")
        print(f"[SPLIT] Total test  graphs: {len(test_idx)}\n")

    return train_idx, test_idx

# ======================
# GAT FINGERPRINT PIPELINE
# ======================

def gat_build_fingerprints(out_np="gat_fingerprints.npy",
                           out_csv="gat_fingerprints.csv"):
    """
    Build book-level fingerprints with GAT, save them,
    and return (Z, y, book_ids, authors, label_encoder).

    NOTE: During training, train/test split is resampled
    at every epoch (author-aware random split).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
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
        import torch
        from torch_geometric.data import Data

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

    import torch
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

    # 4) Train GAT with RANDOM SPLIT EACH EPOCH
    EPOCHS = 100
    print("\n[TRAIN] Starting GAT training with per-epoch random splits...")
    for epoch in range(1, EPOCHS + 1):
        # --- resample split (different random_state each epoch) ---
        train_idx, test_idx = make_author_split(
            authors,
            random_state=42 + epoch,      # different seed → different split
            verbose=(epoch == 1)          # show full details only for epoch 1
        )

        from torch_geometric.loader import DataLoader
        train_dataset = [pyg_data_list[i] for i in train_idx]
        test_dataset  = [pyg_data_list[i] for i in test_idx]

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

        # --- standard training step ---
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

        # quick test accuracy on *this epoch's* random test split
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

    # 5) Extract fingerprints for ALL graphs
    print("\n[EMB] Extracting GAT fingerprints for all graphs...")
    from torch_geometric.loader import DataLoader as DL_all
    all_loader = DL_all(pyg_data_list, batch_size=8, shuffle=False)

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

    # 6) Save fingerprints
    np.save(out_np, Z)
    df_out = pd.DataFrame(Z, columns=[f"emb_{i}" for i in range(Z.shape[1])])
    df_out.insert(0, "book_id", book_ids)
    df_out.insert(1, "author", authors)
    df_out.to_csv(out_csv, index=False)
    print(f"[SAVE] Saved fingerprints to: {out_np}")
    print(f"[SAVE] Saved CSV to:          {out_csv}")

    return Z, y_arr, book_ids, authors, label_encoder


if __name__ == "__main__":
    print("Running GAT fingerprint generation...")
    gat_build_fingerprints()
