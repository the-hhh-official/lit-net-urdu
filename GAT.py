import os
import numpy as np
import pandas as pd
import networkx as nx

# ======================
# CONSTANTS & SETTINGS
# ======================

META_EDGES_PATH = "books_metadata - w0.csv"  # edges + author
META_NODE_PATH  = "features_metadata.csv"    # node features + author

# columns in w0
META0_COL_BOOK_ID = "book_id"
META0_COL_EDGES   = "edges_path"
META0_COL_AUTHOR  = "author"

# columns in w5
META5_COL_BOOK_ID = "book_id"
META5_COL_AUTHOR  = "author"
META5_COL_NODE_FEAT = "node_features_path"

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
    meta_edges = pd.read_csv(META_EDGES_PATH)
    meta_nodes = pd.read_csv(META_NODE_PATH)

    merged = pd.merge(
        meta_edges[[META0_COL_BOOK_ID, META0_COL_EDGES, META0_COL_AUTHOR]],
        meta_nodes[[META5_COL_BOOK_ID, META5_COL_NODE_FEAT, META5_COL_AUTHOR]],
        left_on=[META0_COL_BOOK_ID, META0_COL_AUTHOR],
        right_on=[META5_COL_BOOK_ID, META5_COL_AUTHOR],
        how="inner"
    )

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

    for ch in gender_dict:
        if not G.has_node(ch):
            G.add_node(ch)

    for node in G.nodes():
        g = gender_dict.get(node, "Unknown")
        r = role_dict.get(node, "Minor")
        G.nodes[node]["x"] = encode_one_hot(g, r)

    return G

def build_all_graphs(edge_paths, node_paths):
    return [build_nx_graph(e, n) for e, n in zip(edge_paths, node_paths)]

# ======================
# GAT FINGERPRINT PIPELINE
# ======================

def gat_build_fingerprints(out_np="gat_fingerprints.npy",
                           out_csv="gat_fingerprints.csv"):
    """
    Build book-level fingerprints with GAT, save them,
    and return (Z, y, book_ids, authors).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # 1) Load metadata and graphs
    meta, book_ids, edge_paths, node_paths, authors = load_metadata()
    graphs = build_all_graphs(edge_paths, node_paths)

    # 2) Convert to PyG Data objects
    def nx_to_pyg(G, label_idx):
        import torch
        from torch_geometric.data import Data

        node_list = list(G.nodes())
        idx_map = {n: i for i, n in enumerate(node_list)}

        # edges (undirected → we add both directions)
        edges = []
        for u, v in G.edges():
            iu = idx_map[u]
            iv = idx_map[v]
            edges.append([iu, iv])
            edges.append([iv, iu])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # graph with isolated nodes
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
    for i, G in enumerate(graphs):
        pyg_data_list.append(nx_to_pyg(G, y[i]))

    # 3) Train / test split at graph level
    idx_all = np.arange(len(pyg_data_list))
    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.2, stratify=y, random_state=42
    )

    train_dataset = [pyg_data_list[i] for i in train_idx]
    test_dataset  = [pyg_data_list[i] for i in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

    in_dim = pyg_data_list[0].x.shape[1]
    num_classes = len(label_encoder.classes_)
    
    class GATGraphClassifier(nn.Module):
        def __init__(self, in_dim, hidden_dim=32, out_dim=32, heads=4, num_classes=10):
            super().__init__()
            self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
            self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=True)
            self.classifier = nn.Linear(out_dim, num_classes)

        def forward(self, x, edge_index, batch):
            x = F.elu(self.gat1(x, edge_index))
            x = F.elu(self.gat2(x, edge_index))
            x_graph = global_mean_pool(x, batch)  # fingerprint
            logits = self.classifier(x_graph)
            return logits, x_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATGraphClassifier(in_dim, hidden_dim=32, out_dim=64,
                               heads=4, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 4) Train GAT as graph classifier
    EPOCHS = 100
    for epoch in range(1, EPOCHS + 1):
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

        train_loss = total_loss / total
        train_acc  = correct / total

        # quick test accuracy
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
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"[GAT] Epoch {epoch}/{EPOCHS}  "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Train Acc: {train_acc:.3f}  Test Acc: {test_acc:.3f}")

    # 5) Extract fingerprints for ALL graphs
    model.eval()
    from torch_geometric.loader import DataLoader as DL_all
    all_loader = DL_all(pyg_data_list, batch_size=8, shuffle=False)

    Z_list = []
    y_list = []
    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(device)
            logits, z_graph = model(batch.x, batch.edge_index, batch.batch)
            Z_list.append(z_graph.cpu().numpy())
            y_list.append(batch.y.cpu().numpy())

    Z = np.concatenate(Z_list, axis=0)
    y_arr = np.concatenate(y_list, axis=0)

    # 6) Save fingerprints
    np.save(out_np, Z)
    df_out = pd.DataFrame(Z, columns=[f"emb_{i}" for i in range(Z.shape[1])])
    df_out.insert(0, "book_id", book_ids)
    df_out.insert(1, "author", authors)
    df_out.to_csv(out_csv, index=False)
    print(f"[GAT] Saved fingerprints to {out_np} and {out_csv}")

    return Z, y_arr, book_ids, authors, label_encoder


if __name__ == "__main__":
    print("Running GAT fingerprint generation...")
    gat_build_fingerprints()
