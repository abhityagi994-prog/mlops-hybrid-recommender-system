from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILTERED_SONGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "collab_filtered_songs.csv"
)

TRANSFORMER_PATH = (
    PROJECT_ROOT
    / "models"
    / "content_transformer.joblib"
)

HYBRID_CONTENT_MATRIX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed_hybrid_data.npz"
)


def prepare_content_features(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_remove = [
        "track_id",
        "name",
        "spotify_preview_url",
        "spotify_id",
        "genre"
    ]

    features = data.drop(columns=columns_to_remove).copy()
    features["year"] = features["year"].astype(str)

    return features


def main():
    songs = pd.read_csv(FILTERED_SONGS_PATH)

    transformer = joblib.load(TRANSFORMER_PATH)

    content_features = prepare_content_features(songs)

    transformed_data = transformer.transform(content_features)

    save_npz(
        HYBRID_CONTENT_MATRIX_PATH,
        transformed_data
    )

    print(f"Filtered songs: {len(songs):,}")
    print(f"Transformed hybrid shape: {transformed_data.shape}")
    print(f"Saved to: {HYBRID_CONTENT_MATRIX_PATH}")


if __name__ == "__main__":
    main()