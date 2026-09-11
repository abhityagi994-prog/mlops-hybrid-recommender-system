from pathlib import Path

import joblib
import pandas as pd
from category_encoders.count import CountEncoder
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_songs.csv"
TRANSFORMED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transformed_content_data.npz"
TRANSFORMER_PATH = PROJECT_ROOT / "models" / "content_transformer.joblib"


FREQUENCY_ENCODE_COLUMNS = ["year"]

ONE_HOT_COLUMNS = [
    "artist",
    "time_signature",
    "key"
]

TFIDF_COLUMN = "tags"

STANDARD_SCALE_COLUMNS = [
    "duration_ms",
    "loudness",
    "tempo"
]

MIN_MAX_SCALE_COLUMNS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]


def prepare_content_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select and prepare features used by the content-based recommender."""

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


def build_transformer() -> ColumnTransformer:
    """Create the preprocessing pipeline for song content features."""

    return ColumnTransformer(
        transformers=[
            (
                "year_frequency",
                CountEncoder(normalize=True, return_df=True),
                FREQUENCY_ENCODE_COLUMNS
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ONE_HOT_COLUMNS
            ),
            (
                "tags_tfidf",
                TfidfVectorizer(max_features=85),
                TFIDF_COLUMN
            ),
            (
                "standard_scaling",
                StandardScaler(),
                STANDARD_SCALE_COLUMNS
            ),
            (
                "minmax_scaling",
                MinMaxScaler(),
                MIN_MAX_SCALE_COLUMNS
            )
        ],
        remainder="passthrough",
        n_jobs=-1
    )


def main():
    songs = pd.read_csv(CLEANED_DATA_PATH)

    content_features = prepare_content_features(songs)

    transformer = build_transformer()

    transformed_data = transformer.fit_transform(content_features)

    TRANSFORMED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRANSFORMER_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_npz(TRANSFORMED_DATA_PATH, transformed_data)
    joblib.dump(transformer, TRANSFORMER_PATH)

    print(f"Input shape: {content_features.shape}")
    print(f"Transformed shape: {transformed_data.shape}")
    print(f"Saved transformed data to: {TRANSFORMED_DATA_PATH}")
    print(f"Saved transformer to: {TRANSFORMER_PATH}")


if __name__ == "__main__":
    main()