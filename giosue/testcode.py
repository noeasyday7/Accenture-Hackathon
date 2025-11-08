import random
from music_recommender import recommend_tracks
import joblib
import pandas as pd
import numpy as np
import ast


num_iteractions = 50

# Read the CSV file
df = pd.read_csv('giosue/playlists_df_processed.csv')
dataset = pd.read_csv('dataset_and_requirements/dataset.csv')

# Cycle through each row
playlists_counter = 0
sum_scores = 0

for index, row in df.iterrows():
    playlist_name = row['playlist_name']
    playlist_data = row['playlist']
    

    print("-" * 50)
    print(f"Playlist: {playlist_name}")

    playlist = ast.literal_eval(playlist_data)

    positive_count = 0

    for iteraction in range(num_iteractions):
        
        # 1️⃣ Shuffle and split playlist
        # ----------------------------
        random.shuffle(playlist)
        split_idx = int(len(playlist) * 0.9)
        train_set = playlist[:split_idx]
        test_set = playlist[split_idx:]

        #print("Training set:", train_set)
        #print("Test set:", test_set)

        # ----------------------------
        # 2️⃣ Call recommendation function
        # ----------------------------
        # Here we ask the function to return just 1 track for simplicity
        target_artists = list(dataset[dataset['track_id'].isin(test_set)]['artists'].dropna().unique())
        n_recommended = 5
        recommended_tracks = recommend_tracks(train_set, target_artists = target_artists,n_recommend=n_recommended, prints=False)

        # ----------------------------
        # 3️⃣ Check if recommendation is in test set
        # ----------------------------

        for track in recommended_tracks:
            if track in test_set:
                #print(f"✅ Recommended track {track} is in the test set!")
                positive_count += 1
            #else:
                #print(f"❌ Recommended track {track} is NOT in the test set.")

    score = positive_count/num_iteractions/n_recommended
    sum_scores += score
    playlists_counter += 1

    print(f"Finial rate of positives: {score}")
    print(f"Overall score: {sum_scores/playlists_counter}")
        
print(f"Overall score: {sum_scores/playlists_counter}")

