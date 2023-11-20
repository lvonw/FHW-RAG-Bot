import json
import pytest
from rest import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_chat_endpoint(client):
    response = client.post('/chat', json={'prompt': 'What is 2 + 2?'})
    data = json.loads(response.data.decode('utf-8'))
    assert response.status_code == 200
    assert 'response' in data
    print(data)

# def test_invalid_chat_request(client):
#     response = client.post('/chat', json={})
#     assert response.status_code == 400
#     data = json.loads(response.data.decode('utf-8'))
#     assert 'error' in data
#     assert 'Missing "prompt" element in request body' in data['error']
