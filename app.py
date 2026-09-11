from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz

from src.models.content_recommender import ContentRecommender
from src.models.hybrid_recommender import HybridRecommender


PROJECT_ROOT = Path(__file__).resolve().parent

CLEANED_SONGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_songs.csv"
)

CONTENT_MATRIX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed_content_data.npz"
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

HYBRID_CONTENT_MATRIX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformed_hybrid_data.npz"
)


@st.cache_data
def load_song_data():
    cleaned_songs = pd.read_csv(CLEANED_SONGS_PATH)
    filtered_songs = pd.read_csv(FILTERED_SONGS_PATH)

    return cleaned_songs, filtered_songs


@st.cache_resource
def load_model_artifacts():
    content_matrix = load_npz(CONTENT_MATRIX_PATH)

    track_ids = np.load(
        TRACK_IDS_PATH,
        allow_pickle=True
    )

    interaction_matrix = load_npz(
        INTERACTION_MATRIX_PATH
    )

    hybrid_content_matrix = load_npz(
        HYBRID_CONTENT_MATRIX_PATH
    )

    return (
        content_matrix,
        track_ids,
        interaction_matrix,
        hybrid_content_matrix
    )


cleaned_songs, filtered_songs = load_song_data()

(
    content_matrix,
    track_ids,
    interaction_matrix,
    hybrid_content_matrix
) = load_model_artifacts()

st.set_page_config(
    page_title="Hybrid Music Recommender",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Hybrid Music Recommender")
st.write(
    "Enter a song and artist to get recommendations using "
    "content-based or hybrid filtering."
)

song_name = st.text_input("Song name")
artist_name = st.text_input("Artist name")

n_recommendations = st.selectbox(
    "Number of recommendations",
    [5, 10, 15, 20],
    index=1
)

song_name_clean = song_name.strip().lower()
artist_name_clean = artist_name.strip().lower()

song_in_catalogue = (
    (cleaned_songs["name"] == song_name_clean)
    & (cleaned_songs["artist"] == artist_name_clean)
).any()

song_in_collaborative = (
    (filtered_songs["name"] == song_name_clean)
    & (filtered_songs["artist"] == artist_name_clean)
).any()

content_weight = 0.5

if song_in_collaborative:
    st.success("Hybrid recommendation available")

    content_weight = st.slider(
        "Recommendation balance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help=(
            "0 = more collaborative listening behaviour. "
            "1 = more content/audio similarity."
        )
    )

    st.write(
        f"Content: **{content_weight:.0%}** | "
        f"Collaborative: **{1 - content_weight:.0%}**"
    )

elif song_in_catalogue:
    st.info(
        "No collaborative listening history is available for this song. "
        "Content-based recommendations will be used."
    )

if st.button("Get Recommendations", type="primary"):
    if not song_name_clean or not artist_name_clean:
        st.warning("Please enter both a song name and artist name.")

    elif not song_in_catalogue:
        st.error(
            f"'{song_name}' by '{artist_name}' was not found in the catalogue."
        )

    else:
        try:
            if song_in_collaborative:
                recommender = HybridRecommender(
                    n_recommendations=n_recommendations,
                    content_weight=content_weight
                )

                recommendations = recommender.recommend(
                    song_name=song_name_clean,
                    artist_name=artist_name_clean,
                    songs_data=filtered_songs,
                    track_ids=track_ids,
                    content_matrix=hybrid_content_matrix,
                    interaction_matrix=interaction_matrix
                )

                score_column = "hybrid_score"
                recommendation_mode = "Hybrid Recommender"

            else:
                recommender = ContentRecommender(
                    n_recommendations=n_recommendations
                )

                recommendations = recommender.recommend(
                    song_name=song_name_clean,
                    artist_name=artist_name_clean,
                    songs_data=cleaned_songs,
                    content_matrix=content_matrix
                )

                score_column = "similarity_score"
                recommendation_mode = "Content-Based Recommender"

            st.subheader("Recommendations")
            st.caption(f"Mode used: {recommendation_mode}")

            for index, row in recommendations.iterrows():
                st.markdown(
                    f"### {index + 1}. {row['name'].title()}"
                )
                st.write(f"**Artist:** {row['artist'].title()}")

                if recommendation_mode == "Hybrid Recommender":
                    st.write(
                        f"**Hybrid score:** {row[score_column]:.3f}"
                    )
                else:
                    st.write(
                        f"**Similarity score:** {row[score_column]:.3f}"
                    )

                preview_url = row["spotify_preview_url"]

                if pd.notna(preview_url) and preview_url:
                    st.audio(preview_url)
                else:
                    st.caption("Preview unavailable.")

                st.divider()

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "Something went wrong while generating recommendations."
            )
            st.exception(error)