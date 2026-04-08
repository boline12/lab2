import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_no_image(client):
    """Test that the API returns 400 if no image is sent."""
    response = client.post('/predict')
    assert response.status_code == 400