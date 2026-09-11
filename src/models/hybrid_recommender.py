import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    def __init__(
        self,
        n_recommendations: int = 10,
        content_weight: float = 0.5
    ):
        if not 0 <= content_weight <= 1:
            raise ValueError("content_weight must be between 0 and 1.")

        self.n_recommendations = n_recommendations
        self.content_weight = content_weight
        self.collaborative_weight = 1 - content_weight

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.zeros_like(scores)

        return (scores - minimum) / (maximum - minimum)

    @staticmethod
    def _find_song_index(
        song_name: str,
        artist_name: str,
        songs_data: pd.DataFrame
    ) -> int:
        matches = songs_data[
            (songs_data["name"] == song_name.lower())
            & (songs_data["artist"] == artist_name.lower())
        ]

        if matches.empty:
            raise ValueError(
                f"Song '{song_name}' by '{artist_name}' was not found."
            )

        return matches.index[0]

    def recommend(
        self,
        song_name: str,
        artist_name: str,
        songs_data: pd.DataFrame,
        track_ids: np.ndarray,
        content_matrix,
        interaction_matrix
    ) -> pd.DataFrame:

        song_index = self._find_song_index(
            song_name=song_name,
            artist_name=artist_name,
            songs_data=songs_data
        )

        input_track_id = songs_data.loc[song_index, "track_id"]

        track_match = np.where(track_ids == input_track_id)[0]

        if len(track_match) == 0:
            raise ValueError(
                "This song does not have collaborative interaction data."
            )

        collaborative_index = track_match[0]

        content_vector = content_matrix[song_index]

        content_scores = cosine_similarity(
            content_vector,
            content_matrix
        ).ravel()

        collaborative_vector = interaction_matrix[collaborative_index]

        collaborative_scores = cosine_similarity(
            collaborative_vector,
            interaction_matrix
        ).ravel()

        normalized_content = self._normalize(content_scores)
        normalized_collaborative = self._normalize(
            collaborative_scores
        )

        combined_scores = (
            self.content_weight * normalized_content
            + self.collaborative_weight * normalized_collaborative
        )

        ranked_indices = np.argsort(combined_scores)[::-1]

        ranked_indices = ranked_indices[
            ranked_indices != song_index
        ]

        recommendation_indices = ranked_indices[
            :self.n_recommendations
        ]

        recommendations = songs_data.loc[
            recommendation_indices,
            [
                "name",
                "artist",
                "spotify_preview_url"
            ]
        ].copy()

        recommendations["hybrid_score"] = combined_scores[
            recommendation_indices
        ]

        return recommendations.reset_index(drop=True)



if __name__ == "__main__":
    from pathlib import Path

    from scipy.sparse import load_npz

    project_root = Path(__file__).resolve().parents[2]

    songs = pd.read_csv(
        project_root
        / "data"
        / "processed"
        / "collab_filtered_songs.csv"
    )

    track_ids = np.load(
        project_root
        / "data"
        / "processed"
        / "track_ids.npy",
        allow_pickle=True
    )

    content_matrix = load_npz(
        project_root
        / "data"
        / "processed"
        / "transformed_hybrid_data.npz"
    )

    interaction_matrix = load_npz(
        project_root
        / "data"
        / "processed"
        / "interaction_matrix.npz"
    )

    recommender = HybridRecommender(
        n_recommendations=10,
        content_weight=0.5
    )

    result = recommender.recommend(
        song_name="Crazy in Love",
        artist_name="Beyoncé",
        songs_data=songs,
        track_ids=track_ids,
        content_matrix=content_matrix,
        interaction_matrix=interaction_matrix
    )

    print(result)