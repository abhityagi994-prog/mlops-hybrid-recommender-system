import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, n_recommendations: int = 10):
        self.n_recommendations = n_recommendations

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
        content_matrix
    ) -> pd.DataFrame:

        song_index = self._find_song_index(
            song_name=song_name,
            artist_name=artist_name,
            songs_data=songs_data
        )

        input_vector = content_matrix[song_index]

        similarity_scores = cosine_similarity(
            input_vector,
            content_matrix
        ).ravel()

        ranked_indices = np.argsort(similarity_scores)[::-1]

        # Remove the queried song itself
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

        recommendations["similarity_score"] = similarity_scores[
            recommendation_indices
        ]

        return recommendations.reset_index(drop=True)

