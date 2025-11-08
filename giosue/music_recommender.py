import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import joblib

# ----------------------------
# 1️⃣ Load and preprocess dataset
# ----------------------------
DATA_PATH = "dataset_and_requirements/dataset.csv"

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
N_NEIGHBORS = 1000
knn_model = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine', algorithm='auto')
knn_model.fit(features_norm)

# ----------------------------
# 3️⃣ Save model and metadata
# ----------------------------
joblib.dump(knn_model, "giosue/model/music_knn_model_enhanced.joblib")
joblib.dump(scaler, "giosue/model/music_scaler_enhanced.joblib")
joblib.dump(encoder, "giosue/model/music_genre_encoder_enhanced.joblib")
np.save("giosue/model/track_ids.npy", track_ids)

print("Enhanced model trained and saved successfully!")

# ----------------------------
# 4️⃣ Recommendation function (multi-track aggregation)
# ----------------------------
def recommend_tracks(track_ids_input, n_recommend=10):
    """
    Get recommendations for multiple input tracks using weighted mean aggregation.
    Ensures no duplicate tracks even if they differ only by genre.
    """
    if isinstance(track_ids_input, str):
        track_ids_input = [track_ids_input]
    
    valid_indices = []
    for track_id in track_ids_input:
        if track_id not in track_ids:
            print(f"Track ID {track_id} not found, skipping.")
        else:
            valid_indices.append(np.where(track_ids == track_id)[0][0])
    
    if len(valid_indices) == 0:
        return []

    # Weighted mean vector of input tracks
    mean_vector = np.mean(features_norm[valid_indices], axis=0)
    
    # Query more neighbors than needed to handle duplicates
    distances, neighbors_idx = knn_model.kneighbors([mean_vector], n_neighbors=n_recommend*3)
    
    # Sort by distance
    neighbors_distances = list(zip(neighbors_idx[0], distances[0]))
    neighbors_distances.sort(key=lambda x: x[1])  # ascending distance = more similar
    
    recommended_tracks = []
    seen_tracks = set()  # track_name + artist combination
    input_tracks_set = set(df.loc[np.array(valid_indices), 'track_name'] + "||" + df.loc[np.array(valid_indices), 'artists'])
    
    for idx, dist in neighbors_distances:
        row = df.iloc[idx]
        key = row['track_name'] + "||" + row['artists']
        if key not in seen_tracks and key not in input_tracks_set:
            recommended_tracks.append(row['track_id'])
            seen_tracks.add(key)
        if len(recommended_tracks) >= n_recommend:
            break

    # ----------------------------
    # Print input tracks info
    print("\n--- Input Tracks ---")
    input_info = df[df['track_id'].isin(track_ids_input)].drop_duplicates(subset=['track_name','artists'])
    for _, row in input_info.iterrows():
        print(f"{row['track_name']} | Artist: {row['artists']} | Genre: {row['track_genre']} | Popularity: {row['popularity']} | Danceability: {row['danceability']} | Energy: {row['energy']} | Tempo: {row['tempo']}")
    
    # Print recommended tracks info
    print("\n--- Recommended Tracks ---")
    for tid in recommended_tracks:
        row = df[df['track_id'] == tid].iloc[0]  # pick first row
        print(f"{row['track_name']} | Artist: {row['artists']} | Genre: {row['track_genre']} | Popularity: {row['popularity']} | Danceability: {row['danceability']} | Energy: {row['energy']} | Tempo: {row['tempo']}")
    
    return recommended_tracks




