import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.models.hybrid_recommender import HybridRecommender


def test_hybrid_recommender_returns_expected_number_of_results():
    songs = pd.read_csv(
        "data/processed/collab_filtered_songs.csv"
    )

    track_ids = np.load(
        "data/processed/track_ids.npy",
        allow_pickle=True
    )

    content_matrix = load_npz(
        "data/processed/transformed_hybrid_data.npz"
    )

    interaction_matrix = load_npz(
        "data/processed/interaction_matrix.npz"
    )

    recommender = HybridRecommender(
        n_recommendations=5,
        content_weight=0.5
    )

    recommendations = recommender.recommend(
        song_name="Hips Don't Lie",
        artist_name="Shakira",
        songs_data=songs,
        track_ids=track_ids,
        content_matrix=content_matrix,
        interaction_matrix=interaction_matrix
    )

    assert len(recommendations) == 5


def test_hybrid_recommender_does_not_return_input_song():
    songs = pd.read_csv(
        "data/processed/collab_filtered_songs.csv"
    )

    track_ids = np.load(
        "data/processed/track_ids.npy",
        allow_pickle=True
    )

    content_matrix = load_npz(
        "data/processed/transformed_hybrid_data.npz"
    )

    interaction_matrix = load_npz(
        "data/processed/interaction_matrix.npz"
    )

    recommender = HybridRecommender(
        n_recommendations=5,
        content_weight=0.5
    )

    recommendations = recommender.recommend(
        song_name="Hips Don't Lie",
        artist_name="Shakira",
        songs_data=songs,
        track_ids=track_ids,
        content_matrix=content_matrix,
        interaction_matrix=interaction_matrix
    )

    input_song_present = (
        (recommendations["name"] == "hips don't lie")
        & (recommendations["artist"] == "shakira")
    ).any()

    assert not input_song_present