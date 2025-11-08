import random
from music_recommender import recommend_tracks, df  # df used to map track_id -> artists
import joblib
import numpy as np
import pandas as pd

# Load saved model and metadata (not strictly needed if music_recommender already loads them,
# but harmless to keep if you want to ensure files exist)
knn_model = joblib.load("model/music_knn_model_enhanced.joblib")
scaler = joblib.load("model/music_scaler_enhanced.joblib")
encoder = joblib.load("model/music_genre_encoder_enhanced.joblib")
track_ids = np.load("model/track_ids.npy", allow_pickle=True)

# ----------------------------
# Load and parse playlists
# ----------------------------
playlists = pd.read_csv("playlists_export.csv")

def parse_playlist(s):
    """
    Convert a string like '[id1,id2,id3]' → ['id1','id2','id3']
    """
    if pd.isna(s):
        return []
    s = s.strip().lstrip("[").rstrip("]")
    ids = [x.strip() for x in s.split(",") if x.strip()]
    return ids

playlists['playlist'] = playlists['playlist'].apply(parse_playlist)

# ----------------------------
# Evaluation config
# ----------------------------
num_iterations = 5      # how many times to run per playlist
n_recommended = 5       # how many tracks to recommend per run
hidden_k = 5            # how many tracks to hide per playlist

positive_count = 0      # how many recommended tracks were actually hidden
total_recs = 0          # total number of recommendations evaluated

# Only include playlists that can support hiding 5 tracks and still have training data
valid_playlists = [
    pl for pl in playlists['playlist']
    if isinstance(pl, list) and len(pl) > hidden_k
]

full_count = num_iterations * len(valid_playlists)
current_count = 0

# ----------------------------
# Main loop
# ----------------------------
for playlist in valid_playlists:
    for iteration in range(num_iterations):

        # 1️⃣ Sample exactly `hidden_k` tracks to hide
        test_set = random.sample(playlist, k=hidden_k)

        # Remaining tracks are the observed playlist
        train_set = [tid for tid in playlist if tid not in test_set]

        # 2️⃣ Collect artists of hidden tracks (the clue)
        test_artists = set(
            df.loc[df['track_id'].isin(test_set), 'artists'].astype(str).tolist()
        )

        # 3️⃣ Get recommendations restricted to those artists
        recommended_tracks = recommend_tracks(
            train_set,
            n_recommend=n_recommended,
            allowed_artists=test_artists,
            verbose=False
        )

        # Progress display
        current_count += 1
        print(f"{(current_count / full_count) * 100:.2f}%")

        # 4️⃣ Evaluate overlap
        for track in recommended_tracks:
            if track in test_set:
                positive_count += 1

        total_recs += len(recommended_tracks)

# ----------------------------
# Final metric
# ----------------------------
if total_recs > 0:
    final_rate = positive_count / total_recs
else:
    final_rate = 0.0

print(f"Final rate of positives: {final_rate:.4f}")
