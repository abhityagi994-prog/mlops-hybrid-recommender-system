import time

import requests


APP_URL = "http://localhost:8501"


def test_streamlit_app_is_running():
    time.sleep(5)

    response = requests.get(
        APP_URL,
        timeout=10
    )

    assert response.status_code == 200