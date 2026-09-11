# Hybrid Music Recommender System

An end-to-end hybrid music recommendation system combining
**content-based filtering** and **collaborative filtering**, with an
automatic content-based fallback for songs without sufficient
interaction history.

The project extends beyond recommendation modelling by implementing a
reproducible MLOps workflow using DVC, automated testing, GitHub
Actions, Docker, AWS S3, Amazon ECR, and Amazon EC2.

## Overview

The system recommends songs using both song characteristics and
listening behaviour.

-   **Content-Based Filtering** recommends songs using similarity
    between song features.
-   **Hybrid Filtering** combines content similarity with collaborative
    listening behaviour.
-   If collaborative interaction history is unavailable for a song, the
    application automatically routes the request to the content-based
    recommender.
-   For hybrid-supported songs, users can adjust the balance between
    content and collaborative signals.

## Key Features

-   Hybrid recommendation engine combining content and collaborative
    signals
-   Automatic content-based fallback for songs without collaborative
    history
-   Adjustable content/collaborative weighting
-   Top-N recommendation selection
-   Spotify preview audio where available
-   Interactive Streamlit application
-   Sparse-matrix feature and interaction storage
-   Reproducible DVC data pipeline
-   Automated testing with Pytest
-   Continuous integration with GitHub Actions
-   Dockerized application
-   DVC artifact storage on Amazon S3
-   Docker image storage on Amazon ECR
-   Application deployment on Amazon EC2
-   IAM role-based ECR access from EC2

## Recommendation Architecture

``` text
                         Song + Artist
                              |
                              v
                     Song Catalogue Check
                              |
                +-------------+-------------+
                |                           |
                v                           v
     Collaborative history          No collaborative
           available                    history
                |                           |
                v                           v
       Hybrid Recommender          Content Recommender
                |                           |
        +-------+-------+                   |
        |               |                   |
        v               v                   |
 Content Similarity  Collaborative          |
                     Similarity             |
        |               |                   |
        +-------+-------+                   |
                |                           |
                v                           |
        Weighted Hybrid Score               |
                |                           |
                +-------------+-------------+
                              |
                              v
                    Top-N Recommendations
```

## MLOps Architecture

``` text
Data
 |
 v
Cleaning / Preprocessing
 |
 +---------------------------+
 |                           |
 v                           v
Collaborative Filtering   Content Transformation
 |                           |
 +-------------+-------------+
               |
               v
       Hybrid Transformation
               |
               v
      Recommendation Engine
               |
               v
       Streamlit Application
               |
               v
             Docker
               |
               v
          Amazon ECR
               |
               v
          Amazon EC2
```

DVC tracks the processing pipeline and generated artifacts, with Amazon
S3 configured as the remote artifact store.

## DVC Pipeline

``` text
                 clean_data
                /          \
               v            v
collaborative_filtering   transform_content
               \            /
                v          v
                transform_hybrid
```

This dependency graph makes the data-processing workflow reproducible
rather than relying on manually executed scripts.

## Tech Stack

  -----------------------------------------------------------------------
  Area                                Technologies
  ----------------------------------- -----------------------------------
  Language                            Python

  Data Processing                     Pandas, NumPy, SciPy

  Machine Learning                    Scikit-learn

  Recommendation                      Content-Based Filtering,
                                      Collaborative Filtering, Hybrid
                                      Scoring

  Application                         Streamlit

  Testing                             Pytest

  Data / Artifact Versioning          DVC

  Cloud Storage                       Amazon S3

  Containerization                    Docker

  Container Registry                  Amazon ECR

  Deployment                          Amazon EC2

  CI                                  GitHub Actions

  Cloud Security                      AWS IAM

  Version Control                     Git, GitHub
  -----------------------------------------------------------------------

## Project Structure

``` text
.
├── .dvc/
├── .github/
│   └── workflows/
│       └── ci.yaml
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── src/
│   ├── data/
│   ├── features/
│   │   └── transform_hybrid_data.py
│   └── models/
│       ├── content_recommender.py
│       └── hybrid_recommender.py
├── tests/
├── app.py
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── requirements-freeze.txt
└── README.md
```

> Generated artifacts, caches, and environment-specific files are
> omitted from this simplified structure.

## Running Locally

Clone the repository and create a virtual environment:

``` bash
git clone <repository-url>
cd mlops-hybrid-recommender-system
python -m venv .venv
```

Activate the environment on Windows:

``` powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

``` bash
python -m pip install -r requirements.txt
```

`requirements.txt` contains the human-readable direct dependencies.
`requirements-freeze.txt` records the fully resolved development
environment.

If valid AWS credentials for the configured DVC remote are available,
restore the DVC-managed artifacts:

``` bash
dvc pull
```

Run the application:

``` bash
streamlit run app.py
```

## Testing

Run the automated test suite with:

``` bash
python -m pytest -v
```

## Docker

Build the image:

``` bash
docker build -t hybrid-music-recommender .
```

Run the application locally in Docker:

``` bash
docker run -p 8501:8501 hybrid-music-recommender
```

The application is then available at `localhost:8501`.

## Continuous Integration

GitHub Actions is used to validate repository changes. The CI workflow
covers the relevant project checks, including dependency setup, access
to DVC-managed artifacts where required, application startup validation,
and automated tests.

AWS credentials required by CI are supplied through **GitHub Secrets**
rather than committed to the repository.

## AWS Deployment

The application was deployed to AWS to validate the complete container
deployment workflow.

``` text
Source Code
    |
    v
Docker Build
    |
    v
Amazon ECR
    |
    v
EC2 Instance
    |
    v
IAM Instance Role
    |
    v
Pull Private ECR Image
    |
    v
Docker Container
    |
    v
Streamlit Application
```

The EC2 instance used an IAM instance profile with read-only ECR
permissions, avoiding permanent AWS access keys on the server. The
container exposed Streamlit's internal port `8501` through HTTP port
`80` on the EC2 host.

After successful end-to-end deployment validation, temporary EC2, ECR,
and deployment-specific IAM resources were decommissioned to avoid
unnecessary cloud costs. DVC artifacts are retained separately in Amazon
S3.

## Reproducibility

The project separates source code from large generated artifacts:

-   Git/GitHub tracks source code and configuration.
-   DVC tracks data and generated artifacts.
-   Amazon S3 acts as the DVC remote.
-   `dvc.yaml` defines processing dependencies.
-   `requirements.txt` documents direct dependencies.
-   `requirements-freeze.txt` records the resolved development
    environment.
-   Docker provides a consistent application runtime.

## What This Project Demonstrates

This project covers the lifecycle required to turn a recommendation
model into a deployable data product:

``` text
Data Processing
      +
Recommendation Modelling
      +
Reproducible Pipelines
      +
Automated Testing
      +
Continuous Integration
      +
Containerization
      +
Cloud Artifact Storage
      +
Cloud Deployment
```

It also handles a practical recommendation-system limitation:
collaborative data is not available for every song. Instead of
restricting the application to the collaborative subset, the system
dynamically falls back to content-based recommendations when
collaborative history is unavailable.

## Future Improvements

-   User-level recommendation profiles
-   Larger interaction datasets
-   Offline recommender evaluation metrics
-   Experiment tracking and model comparison
-   Automated continuous deployment
-   Recommendation feedback collection
-   Model and application monitoring
-   Infrastructure as Code
-   Managed container deployment and autoscaling

## License

This project is available under the repository's MIT License.
