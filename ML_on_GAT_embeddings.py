import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ======================
# CONFIG
# ======================

FINGERPRINT_CSV   = "gat_fingerprints.csv"      # from your pure GAT script (64-d emb)
META_EDGES_PATH   = "books_metadata - w0.csv"   # edges + author + book_id
META_NODE_PATH    = "features_metadata.csv"     # node features + author + book_id

# Columns
META0_COL_BOOK_ID   = "book_id"
META0_COL_EDGES     = "edges_path"
META0_COL_AUTHOR    = "author"

META5_COL_BOOK_ID   = "book_id"
META5_COL_NODE_FEAT = "node_features_path"
META5_COL_AUTHOR    = "author"

NODE_COL_CHAR   = "Character"
NODE_COL_GENDER = "Gender"
NODE_COL_ROLE   = "Role"

# Categorical vocab (same as GAT)
GENDER_CATS = ["Male", "Female", "Unknown"]
ROLE_CATS   = ["Protagonist", "Antagonist", "Support",
               "Major-Support", "Narrator", "Minor", "Inactive"]

RANDOM_STATE_SPLIT = 42


# ======================
# 1. LOAD GAT FINGERPRINTS
# ======================

def load_fingerprints(path):
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    X_gat = df[emb_cols].values.astype(np.float32)
    authors = df["author"].astype(str).tolist()
    book_ids = df["book_id"].tolist()

    print(f"[GAT] Loaded {df.shape[0]} books, emb_dim = {X_gat.shape[1]}")
    return df, X_gat, authors, book_ids, emb_cols


# ======================
# 2. BUILD GRAPHS & POOLED FEATURES
# ======================

def load_metadata_edges_nodes():
    meta_edges = pd.read_csv(META_EDGES_PATH)
    meta_nodes = pd.read_csv(META_NODE_PATH)

    merged = pd.merge(
        meta_edges[[META0_COL_BOOK_ID, META0_COL_EDGES, META0_COL_AUTHOR]],
        meta_nodes[[META5_COL_BOOK_ID, META5_COL_NODE_FEAT, META5_COL_AUTHOR]],
        left_on=[META0_COL_BOOK_ID, META0_COL_AUTHOR],
        right_on=[META5_COL_BOOK_ID, META5_COL_AUTHOR],
        how="inner"
    )
    print(f"[META] Merged edges + nodes rows: {len(merged)}")
    return merged


def load_node_feature_dict(path):
    df = pd.read_csv(path)
    g, r = {}, {}
    for _, row in df.iterrows():
        ch = str(row[NODE_COL_CHAR])
        g[ch] = str(row[NODE_COL_GENDER])
        r[ch] = str(row[NODE_COL_ROLE])
    return g, r


def compute_semantic_features(gender_dict, role_dict):
    genders = list(gender_dict.values())
    roles   = list(role_dict.values())

    total_g = len(genders) if len(genders) > 0 else 1
    total_r = len(roles)   if len(roles)   > 0 else 1

    gender_props = [genders.count(g) / total_g for g in GENDER_CATS]
    role_props   = [roles.count(r)   / total_r for r in ROLE_CATS]

    return gender_props + role_props  # len = 3 + 7 = 10


def compute_structural_features(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return [0.0]*8

    degs = [d for _, d in G.degree()]
    if len(degs) == 0:
        degs = [0]

    density      = nx.density(G) if n > 1 else 0.0
    deg_mean     = float(np.mean(degs))
    deg_std      = float(np.std(degs))
    deg_max      = float(np.max(degs))
    clustering   = float(nx.average_clustering(G)) if m > 0 else 0.0
    transitivity = float(nx.transitivity(G))       if m > 0 else 0.0

    return [
        float(n),
        float(m),
        density,
        deg_mean,
        deg_std,
        deg_max,
        clustering,
        transitivity
    ]


def compute_pooled_features_for_all_books():
    meta = load_metadata_edges_nodes()

    sem_list = []
    struct_list = []
    book_ids = []
    authors = []

    total = len(meta)
    print("[FEAT] Computing pooled semantic + structural features...")
    for i, (_, row) in enumerate(meta.iterrows()):
        print(f"[FEAT] Book {i+1}/{total}", end="\r")

        edge_path = row[META0_COL_EDGES]
        node_path = row[META5_COL_NODE_FEAT]

        # build graph
        df_edges = pd.read_csv(edge_path)
        gender_dict, role_dict = load_node_feature_dict(node_path)

        G = nx.Graph()
        for _, erow in df_edges.iterrows():
            u = str(erow["Character_A"])
            v = str(erow["Character_B"])
            w = float(erow["Weight"])
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)

        # ensure isolated nodes
        for ch in gender_dict:
            if ch not in G:
                G.add_node(ch)

        sem_vec   = compute_semantic_features(gender_dict, role_dict)  # 10
        struct_vec = compute_structural_features(G)                    # 8

        sem_list.append(sem_vec)
        struct_list.append(struct_vec)
        book_ids.append(row[META0_COL_BOOK_ID])
        authors.append(row[META0_COL_AUTHOR])

    print("\n[FEAT] Done.")

    sem_arr   = np.array(sem_list, dtype=np.float32)
    struct_arr = np.array(struct_list, dtype=np.float32)

    # Build DataFrame
    sem_cols = (
        [f"gender_prop_{g}" for g in GENDER_CATS] +
        [f"role_prop_{r.replace('-', '_')}" for r in ROLE_CATS]
    )
    struct_cols = [
        "n_nodes", "n_edges", "density",
        "deg_mean", "deg_std", "deg_max",
        "clustering", "transitivity"
    ]

    df_feat = pd.DataFrame({
        "book_id": book_ids,
        "author": authors
    })
    for j, col in enumerate(sem_cols):
        df_feat[col] = sem_arr[:, j]
    for j, col in enumerate(struct_cols):
        df_feat[col] = struct_arr[:, j]

    return df_feat, sem_cols, struct_cols


# ======================
# 3. AUTHOR-AWARE SPLIT
# ======================

def make_fixed_author_split(authors, random_state=42, verbose=True):
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
        this_test = idxs[0]
        this_train = idxs[1:]

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
# 4. SUPERVISED TRAINING ON HYBRID FEATURES
# ======================

def run_supervised_hybrid():
    # --- Load GAT fingerprints ---
    df_gat, X_gat, authors_gat, book_ids_gat, emb_cols = load_fingerprints(FINGERPRINT_CSV)

    # --- Compute pooled semantic + structural ---
    df_feat, sem_cols, struct_cols = compute_pooled_features_for_all_books()

    # --- Merge on book_id + author ---
    merged = pd.merge(
        df_gat,
        df_feat,
        on=["book_id", "author"],
        how="inner"
    )
    print(f"[MERGE] After merge: {len(merged)} books")

    # Rebuild arrays after merge
    authors = merged["author"].astype(str).tolist()
    book_ids = merged["book_id"].tolist()

    # GAT part
    emb_cols_merged = [c for c in merged.columns if c.startswith("emb_")]
    X_gat_m = merged[emb_cols_merged].values.astype(np.float32)

    # pooled features
    feat_cols = sem_cols + struct_cols
    X_pooled = merged[feat_cols].values.astype(np.float32)

    # hybrid concat
    X_hybrid = np.concatenate([X_gat_m, X_pooled], axis=1)
    print(f"[MERGE] GAT dims: {X_gat_m.shape[1]}  pooled dims: {X_pooled.shape[1]}  hybrid: {X_hybrid.shape[1]}")

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(authors)
    class_names = le.classes_

    # Author-aware split
    train_idx, test_idx = make_fixed_author_split(authors, random_state=RANDOM_STATE_SPLIT)

    X_train, X_test = X_hybrid[train_idx], X_hybrid[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("[DATA] Train shape:", X_train.shape, " Test shape:", X_test.shape)

    # ---------------------------
    # 4.1 Logistic Regression
    # ---------------------------
    print("\n=== Logistic Regression on HYBRID (GAT + pooled semantic + structural) ===")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    logreg = LogisticRegression(
        multi_class="multinomial",
        max_iter=500,
        solver="lbfgs",
        random_state=0
    )
    logreg.fit(X_train_scaled, y_train)

    y_train_pred = logreg.predict(X_train_scaled)
    y_test_pred  = logreg.predict(X_test_scaled)

    print(f"[LOGREG] Train accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"[LOGREG] Test  accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print("\n[LOGREG] Classification report (test):")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    cm_logreg = confusion_matrix(y_test, y_test_pred)
    print("[LOGREG] Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm_logreg, index=class_names, columns=class_names))

    # ---------------------------
    # 4.2 Random Forest
    # ---------------------------
    print("\n=== Random Forest on HYBRID (GAT + pooled semantic + structural) ===")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=0,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf  = rf.predict(X_test)

    print(f"[RF] Train accuracy: {accuracy_score(y_train, y_train_pred_rf):.3f}")
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
    print("=== Hybrid Supervised ML: GAT fingerprints + pooled semantic + structural (no titles) ===")
    run_supervised_hybrid()
