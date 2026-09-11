from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "songs" / "Music Info.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_songs.csv"


def clean_songs_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw song metadata."""

    cleaned_data = (
        data
        .drop_duplicates(subset=["spotify_id", "year", "duration_ms"])
        .fillna({"tags": "no_tags"})
        .assign(
            name=lambda df: df["name"].str.lower(),
            artist=lambda df: df["artist"].str.lower(),
            tags=lambda df: df["tags"].str.lower()
        )
        .reset_index(drop=True)
    )

    return cleaned_data


def main():
    # Load raw song data
    songs = pd.read_csv(RAW_DATA_PATH)

    # Clean the dataset
    cleaned_songs = clean_songs_data(songs)

    # Make sure the output directory exists
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    cleaned_songs.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Raw rows: {len(songs):,}")
    print(f"Cleaned rows: {len(cleaned_songs):,}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()