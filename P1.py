import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix


META_PATH = "ML\\books_metadata - w5.csv"
meta = pd.read_csv(META_PATH)
book_ids   = meta["book_id"].tolist()
edge_paths = meta["edges_path"].tolist()
authors    = meta["author"].tolist()



def load_graph_from_edges(path):
    df_edges = pd.read_csv(path)
    src_col = "Character_A"
    dst_col = "Character_B"
    w_col   = "Weight"
    G = nx.Graph()

    for _, row in df_edges.iterrows():
        u = row[src_col]
        v = row[dst_col]
        w = float(row[w_col])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def compute_graph_features(G):
    features = {}

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    features["n_nodes"] = n_nodes
    features["n_edges"] = n_edges
    features["density"] = nx.density(G)

    if n_nodes > 0:
        degrees = np.array([d for _, d in G.degree()])
        features["deg_mean"] = float(degrees.mean())
        features["deg_max"]  = float(degrees.max())
        features["deg_std"]  = float(degrees.std())
    else:
        features["deg_mean"] = 0.0
        features["deg_max"]  = 0.0
        features["deg_std"]  = 0.0

    # Edge weight stats
    if n_edges > 0:
        weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
        features["w_mean"] = float(weights.mean())
        features["w_max"]  = float(weights.max())
        features["w_std"]  = float(weights.std())
    else:
        features["w_mean"] = 0.0
        features["w_max"]  = 0.0
        features["w_std"]  = 0.0

    # Weighted degree stats (sum of weights per node)
    if n_nodes > 0 and n_edges > 0:
        w_deg = []
        for node in G.nodes():
            w_sum = sum(d["weight"] for _, _, d in G.edges(node, data=True))
            w_deg.append(w_sum)
        w_deg = np.array(w_deg)
        features["wdeg_mean"] = float(w_deg.mean())
        features["wdeg_max"]  = float(w_deg.max())
        features["wdeg_std"]  = float(w_deg.std())
    else:
        features["wdeg_mean"] = 0.0
        features["wdeg_max"]  = 0.0
        features["wdeg_std"]  = 0.0

    # Clustering 
    if n_nodes > 0 and n_edges > 0:
        clustering = nx.clustering(G, weight="weight")
        clustering_vals = np.array(list(clustering.values()))
        features["clustering_mean"] = float(clustering_vals.mean())
    else:
        features["clustering_mean"] = 0.0

    # Connected components
    if n_nodes > 0:
        components = list(nx.connected_components(G))
        features["n_components"] = len(components)
        largest_cc_size = max(len(c) for c in components)
        features["largest_cc_size"] = largest_cc_size
        features["largest_cc_frac"] = largest_cc_size / n_nodes
    else:
        features["n_components"] = 0
        features["largest_cc_size"] = 0
        features["largest_cc_frac"] = 0.0

    return features



all_feature_dicts = []

for path in edge_paths:
    G = load_graph_from_edges(path)
    feats = compute_graph_features(G)
    all_feature_dicts.append(feats)

X_df = pd.DataFrame(all_feature_dicts)
print("Feature columns:", list(X_df.columns))
print("X_df shape:", X_df.shape)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(authors)

print("Author label mapping:")
for idx, name in enumerate(label_encoder.classes_):
    print(f"  {idx} -> {name}")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier



X = X_df.values 

log_reg_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=5000   
    ))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(log_reg_clf, X, y, cv=skf, scoring="accuracy")

print("\n[Logistic Regression] Cross-validated accuracy scores:", cv_scores_lr)
print("[Logistic Regression] Mean accuracy:", cv_scores_lr.mean())
print("[Logistic Regression] Std accuracy:", cv_scores_lr.std())


rf_clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

cv_scores_rf = cross_val_score(rf_clf, X, y, cv=skf, scoring="accuracy")

print("\n[Random Forest] Cross-validated accuracy scores:", cv_scores_rf)
print("[Random Forest] Mean accuracy:", cv_scores_rf.mean())
print("[Random Forest] Std accuracy:", cv_scores_rf.std())


best_clf = rf_clf

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, book_ids, test_size=0.2, stratify=y, random_state=42
)

best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print("\nClassification report (test split, Random Forest):")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_
))

print("Confusion matrix (rows: true, cols: predicted):")
print(confusion_matrix(y_test, y_pred))

