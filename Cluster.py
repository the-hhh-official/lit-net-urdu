import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

META_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\SNA Proj\out\books_metadata - w0.csv"   # adjust if needed
EMB_OUTPUT = "ML\\book_enhanced_embeddings.csv"


# ----------------------------------------------------------
#   REINDEX GRAPH (Convert node names to integers)
# ----------------------------------------------------------
def reindex_graph(G):
    mapping = {node: i for i, node in enumerate(G.nodes())}
    H = nx.relabel_nodes(G, mapping, copy=True)
    return H



def graph_features(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # --- BASIC FEATURES ---
    degrees = [deg for _, deg in G.degree()]
    avg_deg = np.mean(degrees) if degrees else 0
    max_deg = np.max(degrees) if degrees else 0
    density = nx.density(G)
    clustering = nx.average_clustering(G) if n > 1 else 0

    # --- CENTRALITY FEATURES ---
    if n > 2:
        betw = nx.betweenness_centrality(G, normalized=True)
        clos = nx.closeness_centrality(G)

        try:
            eig = nx.eigenvector_centrality(G, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            eig = {node: 0 for node in G.nodes()}  # fallback if fails

        avg_betw = np.mean(list(betw.values()))
        max_betw = np.max(list(betw.values()))

        avg_clos = np.mean(list(clos.values()))
        max_clos = np.max(list(clos.values()))

        avg_eig = np.mean(list(eig.values()))
        max_eig = np.max(list(eig.values()))
    else:
        avg_betw = max_betw = 0
        avg_clos = max_clos = 0
        avg_eig = max_eig = 0

    # --- CONNECTIVITY FEATURES ---
    comps = list(nx.connected_components(G))
    num_components = len(comps)

    if num_components > 0:
        largest = G.subgraph(max(comps, key=len)).copy()

        if largest.number_of_nodes() > 1:
            avg_path = nx.average_shortest_path_length(largest)

            try:
                diameter = nx.diameter(largest)
            except nx.NetworkXError:
                diameter = 0
        else:
            avg_path = 0
            diameter = 0
    else:
        avg_path = 0
        diameter = 0
    
    return np.array([
        n, m, avg_deg, max_deg, density, clustering,
        avg_betw, max_betw,
        avg_clos, max_clos,
        avg_eig, max_eig,
        num_components, avg_path, diameter
    ], dtype=float)



def build_simple_embeddings(meta_path=META_PATH, output_path=EMB_OUTPUT):

    graphs, meta_out = load_all_graphs(meta_path)

    feat_list = []
    for G in graphs:
        feat_list.append(graph_features(G))

    X = np.vstack(feat_list)

    # Correct column labels
    emb_cols = [
        "n_nodes", "n_edges", "avg_deg", "max_deg", "density", "clustering",
        "avg_betweenness", "max_betweenness",
        "avg_closeness", "max_closeness",
        "avg_eigenvector", "max_eigenvector",
        "num_components", "avg_shortest_path", "diameter"
    ]

    emb_df = pd.DataFrame(X, columns=emb_cols)

    final_df = pd.concat([meta_out.reset_index(drop=True), emb_df], axis=1)
    final_df.to_csv(output_path, index=False)
    print(f"Saved enhanced embeddings to {output_path}")


def load_graph_from_edges(path):
    df_edges = pd.read_csv(path)

    src_col = "Character_A"
    dst_col = "Character_B"
    w_col = "Weight"

    G = nx.Graph()
    for _, row in df_edges.iterrows():
        u = row[src_col]
        v = row[dst_col]
        w = row[w_col]

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G



def load_all_graphs(meta_path=META_PATH):
    meta = pd.read_csv(meta_path)

    has_title = "title" in meta.columns

    graphs = []
    book_ids = []
    authors = []
    titles = [] if has_title else None

    for _, row in meta.iterrows():
        book_id = row["book_id"]
        author = row["author"]
        edges_path = row["edges_path"]

        if not os.path.isfile(edges_path):
            print(f"[WARN] Missing edges file: {edges_path}")
            continue

        G = load_graph_from_edges(edges_path)
        graphs.append(reindex_graph(G))
        book_ids.append(book_id)
        authors.append(author)
        if has_title:
            titles.append(row["title"])

    meta_out = pd.DataFrame({
        "book_id": book_ids,
        "author": authors
    })
    if has_title:
        meta_out["title"] = titles

    return graphs, meta_out



# if __name__ == "__main__":
#     build_simple_embeddings()






df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Desktop\SNA Proj\out\book_enhanced_embeddings.csv")

feature_cols = [
    "n_nodes", "n_edges", "avg_deg", "max_deg", "density", "clustering",
    "avg_betweenness", "max_betweenness",
    "avg_closeness", "max_closeness",
    "avg_eigenvector", "max_eigenvector",
    "num_components", "avg_shortest_path", "diameter"
]
X = df[feature_cols].values


kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

import umap.umap_ as umap

reducer = umap.UMAP(random_state=42)
coords = reducer.fit_transform(X)   # X is your feature matrix

# Add UMAP coordinates to dataframe
df["u1"] = coords[:, 0]
df["u2"] = coords[:, 1]



# Get unique authors
authors = df["author"].unique()
num_authors = len(authors)

# Generate distinct colors for each author
colors = cm.get_cmap("tab20", num_authors)   # tab20 supports up to 20; can change to tab20b/tab20c if needed

# Create a mapping from author -> color index
author_to_color = {author: i for i, author in enumerate(authors)}

plt.figure(figsize=(14, 8))

# Plot each author group separately to create legend entries
for author in authors:
    subset = df[df["author"] == author]
    
    plt.scatter(
        subset["u1"],
        subset["u2"],
        c=[colors(author_to_color[author])],
        label=str(author),
        s=70
    )
    
    for i in range(len(subset)):
        plt.text(
            subset["u1"].iloc[i] + 0.02,
            subset["u2"].iloc[i] + 0.02,
            str(subset["book_id"].iloc[i]),
            fontsize=8
        )

plt.title("Book Graph UMAP — Colored by Author, Labeled by Book ID", fontsize=16)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

# Add legend
plt.legend(title="Authors", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()





