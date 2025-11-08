import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("dataset.csv")

feature_cols = [
    "popularity",
    "danceability",
    "energy",
    "key",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

df_feats = df.dropna(subset=feature_cols).reset_index(drop=True)

X = df_feats[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

nn_model = NearestNeighbors(
    n_neighbors=51,
    metric="minkowski",
    p=4
)
nn_model.fit(X_scaled)

def recommend_from_track_id_knn(track_id, df=df_feats, X=X_scaled, nn=nn_model,
                                n_recs=10):
    matches = df.index[df["track_id"] == track_id].tolist()
    if not matches:
        raise ValueError("track_id not found")
    seed_idx = matches[0]

    seed_vec = X[seed_idx].reshape(1, -1)
    dists, idxs = nn.kneighbors(seed_vec, n_neighbors=n_recs + 1)

    idxs = idxs[0]
    dists = dists[0]

    filtered = [(i, d) for i, d in zip(idxs, dists) if i != seed_idx][:n_recs]

    recs = df.iloc[[i for i, _ in filtered]][
        ["track_id", "artists", "track_name", "popularity"]
    ]
    return recs

recs_knn = recommend_from_track_id_knn("5SuOikwiRyPMVoIQDJUgSV")
print(recs_knn)

