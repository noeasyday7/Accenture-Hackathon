import random
from music_recommender import recommend_tracks
import joblib
import pandas as pd
import numpy as np
import ast

# Load saved model and metadata
knn_model = joblib.load("giosue/model/music_knn_model_enhanced.joblib")
scaler = joblib.load("giosue/model/music_scaler_enhanced.joblib")
encoder = joblib.load("giosue/model/music_genre_encoder_enhanced.joblib")
track_ids = np.load("giosue/model/track_ids.npy", allow_pickle=True)



# Example playlist: replace with your actual track IDs
playlist = ['4uUG5RXrOk84mYEfFvj3cK','6DXLO8LndZMVOHM0wNbpzg','6h5PAsRni4IRlxWr6uDPTP','1oew3nFNY3vMacJAsvry0S','3eJH2nAjvNXdmPfBkALiPZ','73vIOb4Q7YN6HeJTbscRx5','213x4gsFDm04hSqIUkg88w','1r8ZCjfrQxoy2wVaBUbpwg','5Hp4xFihdOE2dmDzxWcBFb','6Pgkp4qUoTmJIPn7ReaGxL','5lfWrciYtohtIMVDVZd0Rf','4h9wh7iOZ0GGn8QVp4RAOB','0WtM2NBVQNNJLh6scP13H8','0sBJA2OCEECMs0HsdIQhvR','72yP0DUlWPyH8P7IoxskwN','0e8nrvls4Qqv5Rfa2UhqmO','3AVXwaOGCEL8cmBecfcsFJ','69kOkLUCkxIZYexIgSG8rq','5r43qanLhUCdBj8HN3fa6B','1PckUlxKqWQs3RlWXVBLw3','4RAR8g8fZNB106ezUurnE0','1TQXIltqoZ5XXyfCbAeSQQ','5iFwAOB2TFkPJk8sMlxP8g','6Z8R6UsFuGXGtiIxiD8ISb','71qB68guEJjbvtjlkZ8DF5','2DB4DdfCFMw1iaR6JaR03a','3Wrjm47oTz2sjIgck11l5e','1bgKMxPQU7JIZEhNsM1vFs','023H4I7HJnxRqsc9cqeFKV','6f5ExP43esnvdKPddwKXJH','1r9xUipOqoNwggBpENDsvJ','2UbVnbE5FH6008mAm6Mmgw','4qtdkdTY1t3RmlmSbWykzR','5SFXOMJJ334Wn2PwBHeRZN','4iJyoBOLtHqaGxP12qzhQI','6Skh3CBum0pZw9TOr7FQnX','1kPpge9JDLpcj15qgrPbYX','4fouWK6XVHhzl78KzQ1UjL','0nrRP2bk19rLc0orkWPQk2','4lhqb6JvbHId48OUJGwymk','4VEEDnEFLI9dUy5QA51rom','0ct6r3EGTcMLPtrXHDvVjc','5f1joOtoMeyppIcJGZQvqJ','6Q3K9gVUZRMZqZKrXovbM2','5nujrmhLynf4yMoMtj8AQF','3eR23VReFzcdmS7TYCrhCe','4ut5G4rgB1ClpMTMfjoIuy','59qrUpoplZxbIZxk6X0Bm3','6hHZNKWzcZ1wSf0NnagKba','2u6Jm2klS4yvAlbSHlxUwI','3ZFTkvIE7kyPt6Nu3PEa7V','21te7Pz3VowOuFho1nfdR0','1FP9s72rDYty6mfoOEYKnE','5K6Ssv4Z3zRvxt0P6EKUAP','5PjdY0CKGZdEuoNab3yDmX','0bI7K9Becu2dtXK1Q3cZNB','30cW9fD87IgbYFl8o0lUze','2PzU4IB8Dr6mxV3lHuaG34','6Owc2SuzwO3LW1SAODYK3l','0rmGAIH9LNJewFw7nKzZnc','6y6xhAgZjvxy5kR5rigpY3','3uUuGVFu1V7jTQL60S1r8z','1gihuPhrLraKYrJMAEONyc','0cvMWzztDy1wNQkBqae8w4','3yvHcaY61FpeSiLiHiIST4','14sOS5L36385FJ3OL8hew4']

num_iteractions = 200
positive_count = 0

for iteraction in range(num_iteractions):
    
    # 1️⃣ Shuffle and split playlist
    # ----------------------------
    random.shuffle(playlist)
    split_idx = int(len(playlist) * 0.7)
    train_set = playlist[:split_idx]
    test_set = playlist[split_idx:]

    #print("Training set:", train_set)
    #print("Test set:", test_set)

    # ----------------------------
    # 2️⃣ Call recommendation function
    # ----------------------------
    # Here we ask the function to return just 1 track for simplicity
    n_recommended = 5
    recommended_tracks = recommend_tracks(train_set, n_recommend=n_recommended,print=False)

    # ----------------------------
    # 3️⃣ Check if recommendation is in test set
    # ----------------------------

    for track in recommended_tracks:
        if track in test_set:
            #print(f"✅ Recommended track {track} is in the test set!")
            positive_count += 1
        #else:
            #print(f"❌ Recommended track {track} is NOT in the test set.")

print(f"Finial rate of positives: {positive_count/num_iteractions/n_recommended}")
      
