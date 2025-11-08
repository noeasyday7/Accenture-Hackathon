import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import joblib

# ----------------------------
# 1️⃣ Load and preprocess dataset
# ----------------------------
DATA_PATH = "dataset.csv"

df = pd.read_csv(DATA_PATH)
df = df.dropna()
df['explicit'] = df['explicit'].astype(int)

# Numeric features
numeric_features = ['popularity','explicit','danceability','energy',
                    'loudness','mode','speechiness','acousticness',
                    'instrumentalness','liveness','tempo']
# removed features: 'key','valence','time_signature','duration_ms',

# ----------------------------
# Feature engineering
# ----------------------------
# Interaction features
df['dance_energy'] = df['danceability'] * df['energy']
df['loud_tempo'] = df['loudness'] * df['tempo']

# Polynomial features (square)
for f in ['danceability','energy','valence','tempo']:
    df[f + '_sq'] = df[f] ** 2

all_features = numeric_features + ['dance_energy','loud_tempo'] + \
               [f + '_sq' for f in ['danceability','energy','valence','tempo']]

# ----------------------------
# Encode track_genre as one-hot
# ----------------------------
encoder = OneHotEncoder(sparse_output=False)
genre_encoded = encoder.fit_transform(df[['track_genre']])
all_features_array = np.hstack([df[all_features].values, genre_encoded])

# ----------------------------
# Scale numeric features (leave one-hot as is)
# ----------------------------
scaler = StandardScaler()
all_features_array[:, :len(all_features)] = scaler.fit_transform(all_features_array[:, :len(all_features)])

# ----------------------------
# Apply feature weights
# ----------------------------
# Assign higher weight to key features: danceability, energy, valence, tempo
weights = np.ones(all_features_array.shape[1])
feature_indices = {f:i for i,f in enumerate(all_features)}
for key_f in ['danceability','energy','valence','tempo','dance_energy','loud_tempo','acousticness','instrumentalness']:
    if key_f in feature_indices:
        weights[feature_indices[key_f]] = 2  # weight important features 3x
all_features_weighted = all_features_array * weights

# ----------------------------
# Normalize rows for cosine similarity
# ----------------------------
features_norm = normalize(all_features_weighted, norm='l2')

# Track IDs
track_ids = df['track_id'].values

# ----------------------------
# 2️⃣ Build kNN model
# ----------------------------
if __name__ == "__main__":
    N_NEIGHBORS = 1000
    knn_model = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='euclidean', algorithm='auto')
    knn_model.fit(features_norm)

    # ----------------------------
    # 3️⃣ Save model and metadata
    # ----------------------------
    joblib.dump(knn_model, "model/music_knn_model_enhanced.joblib")
    joblib.dump(scaler, "model/music_scaler_enhanced.joblib")
    joblib.dump(encoder, "model/music_genre_encoder_enhanced.joblib")
    np.save("model/track_ids.npy", track_ids)

    print("Enhanced model trained and saved successfully!")
else:
    knn_model = joblib.load("model/music_knn_model_enhanced.joblib")
# ----------------------------
# 4️⃣ Recommendation function (multi-track aggregation)
# ----------------------------
def recommend_tracks(track_ids_input, n_recommend=10, allowed_artists=None, verbose=True):
    """
    Get recommendations for multiple input tracks using weighted mean aggregation.
    Optionally restrict recommendations to tracks created by artists in `allowed_artists`.
    Ensures no duplicate tracks even if they differ only by genre.
    """
    # Normalize input type
    if isinstance(track_ids_input, str):
        track_ids_input = [track_ids_input]

    # Map track IDs to indices
    valid_indices = []
    for track_id in track_ids_input:
        if track_id not in track_ids:
            if verbose:
                print(f"Track ID {track_id} not found, skipping.")
        else:
            valid_indices.append(np.where(track_ids == track_id)[0][0])

    if len(valid_indices) == 0:
        return []

    # Weighted mean vector of input tracks
    mean_vector = np.mean(features_norm[valid_indices], axis=0)

    # Query more neighbors than needed to handle duplicates + filtering
    distances, neighbors_idx = knn_model.kneighbors([mean_vector], n_neighbors=n_recommend * 5)

    # Prepare sorting (ascending distance = more similar)
    neighbors_distances = list(zip(neighbors_idx[0], distances[0]))
    neighbors_distances.sort(key=lambda x: x[1])

    # Prepare artist filter if provided
    allowed_artists_set = None
    if allowed_artists:
        # Lowercase for robust matching
        allowed_artists_set = {a.strip().lower() for a in allowed_artists}

    recommended_tracks = []
    seen_tracks = set()  # track_name||artists
    input_tracks_set = set(
        df.loc[np.array(valid_indices), 'track_name'] + "||" + df.loc[np.array(valid_indices), 'artists']
    )

    for idx, dist in neighbors_distances:
        row = df.iloc[idx]
        key = row['track_name'] + "||" + row['artists']

        # Skip if already seen or is one of the input tracks
        if key in seen_tracks or key in input_tracks_set:
            continue

        # If we have an artist hint, only accept tracks from those artists
        if allowed_artists_set is not None:
            artists_lower = str(row['artists']).lower()
            if not any(a in artists_lower for a in allowed_artists_set):
                continue

        recommended_tracks.append(row['track_id'])
        seen_tracks.add(key)

        if len(recommended_tracks) >= n_recommend:
            break

    if verbose:
        # Print input tracks info
        print("\n--- Input Tracks ---")
        input_info = df[df['track_id'].isin(track_ids_input)].drop_duplicates(subset=['track_name', 'artists'])
        for _, r in input_info.iterrows():
            print(
                f"{r['track_name']} | Artist: {r['artists']} | Genre: {r['track_genre']} "
                f"| Popularity: {r['popularity']} | Danceability: {r['danceability']} "
                f"| Energy: {r['energy']} | Tempo: {r['tempo']}"
            )

        # Print recommended tracks info
        print("\n--- Recommended Tracks ---")
        for tid in recommended_tracks:
            r = df[df['track_id'] == tid].iloc[0]
            print(
                f"{r['track_name']} | Artist: {r['artists']} | Genre: {r['track_genre']} "
                f"| Popularity: {r['popularity']} | Danceability: {r['danceability']} "
                f"| Energy: {r['energy']} | Tempo: {r['tempo']}"
            )

    return recommended_tracks





