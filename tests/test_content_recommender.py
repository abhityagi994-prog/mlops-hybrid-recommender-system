import pandas as pd
from scipy.sparse import load_npz

from src.models.content_recommender import ContentRecommender


def test_content_recommender_returns_expected_number_of_results():
    songs = pd.read_csv("data/processed/cleaned_songs.csv")
    content_matrix = load_npz(
        "data/processed/transformed_content_data.npz"
    )

    recommender = ContentRecommender(
        n_recommendations=5
    )

    recommendations = recommender.recommend(
        song_name="Whenever, Wherever",
        artist_name="Shakira",
        songs_data=songs,
        content_matrix=content_matrix
    )

    assert len(recommendations) == 5



def test_content_recommender_does_not_return_input_song():
    songs = pd.read_csv("data/processed/cleaned_songs.csv")
    content_matrix = load_npz(
        "data/processed/transformed_content_data.npz"
    )

    recommender = ContentRecommender(
        n_recommendations=5
    )

    recommendations = recommender.recommend(
        song_name="Whenever, Wherever",
        artist_name="Shakira",
        songs_data=songs,
        content_matrix=content_matrix
    )

    input_song_present = (
        (recommendations["name"] == "whenever, wherever")
        & (recommendations["artist"] == "shakira")
    ).any()

    assert not input_song_present