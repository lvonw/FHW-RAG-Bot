import json
import pytest
from unittest.mock import patch
from rest import app
import constants

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('rest.main.invoke_chain')
def test_chat_endpoint(mock_invoke_chain, client):
    mock_invoke_chain.return_value = 'Mocked response'

    args = constants.DefaultArgs
    args.question = 'What is 2 + 2?'
    response = client.post('/chat', json={'prompt': args.question})
    data = json.loads(response.data.decode('utf-8'))

    assert response.status_code == 200
    assert 'response' in data
    assert data['response'] == 'Mocked response'
    mock_invoke_chain.assert_called_once_with(args)



@patch('rest.main.invoke_chain')
def test_chat_endpoint_all(mock_invoke_chain, client):
    mock_invoke_chain.return_value = 'Mocked response'

    args = constants.DefaultArgs
    args.question = 'What is 2 + 2?'
    args.database = 'PDF Loader'
    args.model = 'TOOLS'
    args.init = False
    args.validate = False
    args.cv = True
    

    response = client.post('/chat', json={
        'prompt': args.question,
        'database': args.database,
        'model': args.model,
        'init': args.init,
        'validate': args.validate,
        'cv': args.cv,
    })

    data = json.loads(response.data.decode('utf-8'))

    assert response.status_code == 200
    assert 'response' in data
    assert data['response'] == 'Mocked response'
    mock_invoke_chain.assert_called_once_with(args)

@patch('rest.main.invoke_chain')
def test_chat_endpoint_init(mock_invoke_chain, client):
    mock_invoke_chain.return_value = 'Mocked response'

    args = constants.DefaultArgs
    args.init = True

    response = client.post('/chat', json={'init': args.init})
    data = json.loads(response.data.decode('utf-8'))

    assert response.status_code == 200
    assert 'response' in data
    assert data['response'] == 'Mocked response'
    mock_invoke_chain.assert_called_once_with(args)

@patch('rest.main.invoke_chain')
def test_chat_endpoint_missing_prompt(mock_invoke_chain, client):
    mock_invoke_chain.return_value = 'Mocked response'

    response = client.post('/chat', json={})
    assert response.status_code == 400
    data = json.loads(response.data.decode('utf-8'))

    assert 'error' in data
    assert 'Missing "prompt" element in request body' in data['error']
    mock_invoke_chain.assert_not_called()


@patch('rest.main.invoke_chain')
def test_chat_endpoint_prompt_and_init(mock_invoke_chain, client):
    mock_invoke_chain.return_value = 'Mocked response'

    response = client.post('/chat', json={
        'prompt': "What's 2+2?",
        'init': True,
    })
    assert response.status_code == 400
    data = json.loads(response.data.decode('utf-8'))

    assert 'error' in data
    assert 'Cannot initialize and respond to prompt at the same time' in data['error']
    mock_invoke_chain.assert_not_called()

@patch('rest.main.invoke_chain')
def test_chat_endpoint_empty_prompt(mock_invoke_chain, client):
    mock_invoke_chain.return_value = 'Mocked response'

    response = client.post('/chat', json={
        'prompt': "",
    })
    assert response.status_code == 400
    data = json.loads(response.data.decode('utf-8'))

    assert 'error' in data
    assert 'Empty prompt' in data['error']
    mock_invoke_chain.assert_not_called()