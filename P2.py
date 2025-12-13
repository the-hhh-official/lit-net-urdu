import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix



META_PATH = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\SNA Proj\\ML\\features_metadata.csv"  



meta = pd.read_csv(META_PATH)

book_ids  = meta["book_id"].tolist()
authors   = meta["author"].tolist()
node_paths = meta["node_features_path"].tolist()

print(f"Loaded metadata for {len(book_ids)} books.")



all_genders = set()
all_roles   = set()
all_titles  = set()

node_tables = []  

for path in node_paths:
    df = pd.read_csv(path)
    node_tables.append(df)

    if "Gender" in df.columns:
        all_genders.update(df["Gender"].dropna().unique().tolist())
    if "Role" in df.columns:
        all_roles.update(df["Role"].dropna().unique().tolist())
    if "TitleOrClass" in df.columns:
        all_titles.update(df["TitleOrClass"].dropna().unique().tolist())

print("Unique genders:", all_genders)
print("Unique roles:", all_roles)
print("Unique titles/classes:", all_titles)

all_genders = sorted(list(all_genders))
all_roles   = sorted(list(all_roles))
all_titles  = sorted(list(all_titles))


def clean_name(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))



book_feature_dicts = []
for book_id, df_nodes in zip(book_ids, node_tables):
    feats = {}
    n_nodes = len(df_nodes)
    feats["n_nodes"] = n_nodes

    if n_nodes == 0:
        for g in all_genders:
            feats[f"gender_prop_{clean_name(g)}"] = 0.0
        for r in all_roles:
            feats[f"role_prop_{clean_name(r)}"] = 0.0
        for t in all_titles:
            feats[f"title_prop_{clean_name(t)}"] = 0.0
    else:
        if "Gender" in df_nodes.columns:
            for g in all_genders:
                count = (df_nodes["Gender"] == g).sum()
                feats[f"gender_prop_{clean_name(g)}"] = count / n_nodes
        else:
            for g in all_genders:
                feats[f"gender_prop_{clean_name(g)}"] = 0.0

        if "Role" in df_nodes.columns:
            for r in all_roles:
                count = (df_nodes["Role"] == r).sum()
                feats[f"role_prop_{clean_name(r)}"] = count / n_nodes
        else:
            for r in all_roles:
                feats[f"role_prop_{clean_name(r)}"] = 0.0

        if "TitleOrClass" in df_nodes.columns:
            for t in all_titles:
                count = (df_nodes["TitleOrClass"] == t).sum()
                feats[f"title_prop_{clean_name(t)}"] = count / n_nodes
        else:
            for t in all_titles:
                feats[f"title_prop_{clean_name(t)}"] = 0.0

    book_feature_dicts.append(feats)



X_df = pd.DataFrame(book_feature_dicts)
print("Book feature columns:", list(X_df.columns))
print("X_df shape:", X_df.shape)



label_encoder = LabelEncoder()
y = label_encoder.fit_transform(authors)

print("Author label mapping:")
for idx, name in enumerate(label_encoder.classes_):
    print(f"  {idx} -> {name}")



X = X_df.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_reg = LogisticRegression(
    solver="lbfgs",
    max_iter=5000
)

cv_scores_lr = cross_val_score(log_reg, X, y, cv=skf, scoring="accuracy")
print("\n[Logistic Regression] CV accuracy scores:", cv_scores_lr)
print("[Logistic Regression] Mean accuracy:", cv_scores_lr.mean())
print("[Logistic Regression] Std accuracy:", cv_scores_lr.std())

rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

cv_scores_rf = cross_val_score(rf_clf, X, y, cv=skf, scoring="accuracy")
print("\n[Random Forest] CV accuracy scores:", cv_scores_rf)
print("[Random Forest] Mean accuracy:", cv_scores_rf.mean())
print("[Random Forest] Std accuracy:", cv_scores_rf.std())



best_clf = rf_clf   

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, book_ids, test_size=0.2, stratify=y, random_state=42
)

best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print("\nClassification report (test split):")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_
))

print("Confusion matrix (rows: true, cols: predicted):")
print(confusion_matrix(y_test, y_pred))



out_df = X_df.copy()
out_df.insert(0, "book_id", book_ids)
out_df.insert(1, "author", authors)

OUT_PATH = "book_semantic_features_from_nodes.csv"
out_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved book-level semantic features to {OUT_PATH}")
