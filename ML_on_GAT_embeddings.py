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


# ======================
# CONFIG
# ======================

FINGERPRINT_CSV = "gat_fingerprints.csv"   # output of the GAT+AE script
RANDOM_STATE_SPLIT = 42                   # to match the idea of previous split


# ======================
# 1. LOAD FINGERPRINTS
# ======================

def load_fingerprints(path):
    """
    Expect a CSV with columns:
        book_id, author, emb_0, emb_1, ..., emb_{d-1}
    """
    df = pd.read_csv(path)
    # feature columns = all that start with "emb_"
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
      - The chosen test book is never used in training.
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
        this_train = idxs[1:]     # remaining books = train

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
# 3. SUPERVISED TRAINING
# ======================

def run_supervised_models(X, authors, book_ids):
    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(authors)
    class_names = le.classes_

    # author-aware split
    train_idx, test_idx = make_fixed_author_split(authors, random_state=RANDOM_STATE_SPLIT)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("[DATA] Train shape:", X_train.shape, " Test shape:", X_test.shape)

    # ---------------------------
    # 3.1 Logistic Regression
    # ---------------------------
    print("\n=== Logistic Regression (multinomial, with StandardScaler) ===")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        multi_class="multinomial",
        max_iter=500,
        solver="lbfgs",
        random_state=0
    )
    logreg.fit(X_train_scaled, y_train)

    y_train_pred = logreg.predict(X_train_scaled)
    y_test_pred = logreg.predict(X_test_scaled)

    print(f"[LOGREG] Train accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"[LOGREG] Test  accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print("\n[LOGREG] Classification report (test):")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    cm_logreg = confusion_matrix(y_test, y_test_pred)
    print("[LOGREG] Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm_logreg, index=class_names, columns=class_names))

    # ---------------------------
    # 3.2 Random Forest
    # ---------------------------
    print("\n=== Random Forest ===")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=0,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf = rf.predict(X_test)

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
    print("=== Supervised ML on GAT fingerprints ===")
    df, X, authors, book_ids = load_fingerprints(FINGERPRINT_CSV)
    run_supervised_models(X, authors, book_ids)
