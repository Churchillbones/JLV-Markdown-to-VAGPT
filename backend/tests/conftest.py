import pytest
from app import app as flask_app # Assuming your Flask app instance is named 'app' in backend/app.py
from datastore import clear_store

@pytest.fixture(scope='session')
def app():
    """Create and configure a new app instance for each test session."""
    flask_app.config.update({
        "TESTING": True,
    })
    # Other setup can go here

    yield flask_app

    # Clean up / reset resources here if necessary for the session scope

@pytest.fixture()
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture(autouse=True)
def cleanup_datastore():
    """Fixture to automatically clear the datastore before each test."""
    clear_store()
    yield # Test runs here
    clear_store() # Cleanup after test
