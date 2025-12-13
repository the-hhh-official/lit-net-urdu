import pandas as pd
import numpy as np
import networkx as nx

from karateclub import Graph2Vec  # pip install karateclub
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score



META_PATH = r"ML\books_metadata - w0.csv"  
meta = pd.read_csv(META_PATH)

book_ids   = meta["book_id"].tolist()
edge_paths = meta["edges_path"].tolist()
authors    = meta["author"].tolist()

print(f"Loaded metadata for {len(book_ids)} books.")


def load_graph_from_edges(path):
    df = pd.read_csv(path)
    src_col = "Character_A"
    dst_col = "Character_B"
    w_col   = "Weight"

    G = nx.Graph()
    for _, row in df.iterrows():
        u = row[src_col]
        v = row[dst_col]
        w = row[w_col]

        if pd.isna(u) or pd.isna(v):
            continue

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G



graphs = []
y = []
valid_book_ids = []


for book_id, edges_path, author in zip(book_ids, edge_paths, authors):
    print(f"\nProcessing book_id={book_id}")
    print(f"  Edges file: {edges_path}")
    try:
        G = load_graph_from_edges(edges_path)
        print(f"  Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        
        G_int = nx.convert_node_labels_to_integers(G)
        print(f"  Relabeled graph: {G_int.number_of_nodes()} nodes, {G_int.number_of_edges()} edges")

        if G_int.number_of_nodes() == 0:
            print("  WARNING: Empty graph, skipping this book.")
            continue

        graphs.append(G_int)
        y.append(author)
        valid_book_ids.append(book_id)

    except Exception as e:
        print(f"  ERROR processing {book_id}: {e}")
        continue

y = np.array(y)
print("\nTotal usable graphs:", len(graphs))
print("Labels shape:", y.shape)


g2v = Graph2Vec(
    dimensions=128,
    wl_iterations=2,
    attributed=False,
    workers=4,
    min_count=1,
    epochs=10
)

print("\nFitting Graph2Vec on all graphs...")
g2v.fit(graphs)

X = g2v.get_embedding()  
print("Graph2Vec embedding shape:", X.shape)




le = LabelEncoder()
y_encoded = le.fit_transform(y)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        solver="lbfgs"
    ))
])



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="accuracy")

print("\nCross-validation accuracy (Graph2Vec → Logistic Regression):")
for i, s in enumerate(scores, start=1):
    print(f"  Fold {i}: {s:.3f}")
print(f"  Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
