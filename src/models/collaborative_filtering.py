from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "songs"
    / "User Listening History.csv"
)

CLEANED_SONGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_songs.csv"
)

FILTERED_SONGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "collab_filtered_songs.csv"
)

TRACK_IDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "track_ids.npy"
)

INTERACTION_MATRIX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interaction_matrix.npz"
)


def filter_songs(
    songs: pd.DataFrame,
    track_ids: np.ndarray
) -> pd.DataFrame:
    """Keep only songs that appear in listening history."""

    filtered_songs = (
        songs[songs["track_id"].isin(track_ids)]
        .sort_values("track_id")
        .reset_index(drop=True)
        .copy()
    )

    return filtered_songs


def build_interaction_matrix(
    history: dd.DataFrame
) -> tuple[csr_matrix, np.ndarray]:
    """Build a sparse track-by-user playcount matrix."""

    history = history.copy()

    history["playcount"] = history["playcount"].astype(np.float64)

    history = history.categorize(
        columns=["user_id", "track_id"]
    )

    track_ids = history["track_id"].cat.categories.to_numpy()

    history = history.assign(
        user_idx=history["user_id"].cat.codes,
        track_idx=history["track_id"].cat.codes
    )

    interactions = (
        history
        .groupby(["track_idx", "user_idx"])["playcount"]
        .sum()
        .reset_index()
        .compute()
    )

    row_indices = interactions["track_idx"].to_numpy()
    col_indices = interactions["user_idx"].to_numpy()
    values = interactions["playcount"].to_numpy()

    n_tracks = len(track_ids)
    n_users = int(col_indices.max()) + 1

    interaction_matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_tracks, n_users)
    )

    return interaction_matrix, track_ids


def main():
    print("Loading listening history...")

    history = dd.read_csv(HISTORY_DATA_PATH)
    songs = pd.read_csv(CLEANED_SONGS_PATH)

    # Build the full collaborative interaction matrix
    interaction_matrix, track_ids = build_interaction_matrix(history)

    # Keep only tracks that also exist in the song metadata
    available_track_mask = np.isin(
        track_ids,
        songs["track_id"].values
    )

    interaction_matrix = interaction_matrix[available_track_mask]
    track_ids = track_ids[available_track_mask]

    # Arrange song metadata in exactly the same order as track_ids
    filtered_songs = (
        songs
        .set_index("track_id")
        .loc[track_ids]
        .reset_index()
    )

    FILTERED_SONGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    filtered_songs.to_csv(
        FILTERED_SONGS_PATH,
        index=False
    )

    np.save(
        TRACK_IDS_PATH,
        track_ids,
        allow_pickle=True
    )

    save_npz(
        INTERACTION_MATRIX_PATH,
        interaction_matrix
    )

    density = (
        interaction_matrix.nnz
        / (
            interaction_matrix.shape[0]
            * interaction_matrix.shape[1]
        )
        * 100
    )

    print(f"Filtered songs: {len(filtered_songs):,}")
    print(f"Track IDs: {len(track_ids):,}")
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"Non-zero interactions: {interaction_matrix.nnz:,}")
    print(f"Matrix density: {density:.6f}%")

    print(f"Saved filtered songs to: {FILTERED_SONGS_PATH}")
    print(f"Saved track IDs to: {TRACK_IDS_PATH}")
    print(f"Saved interaction matrix to: {INTERACTION_MATRIX_PATH}")


if __name__ == "__main__":
    main()