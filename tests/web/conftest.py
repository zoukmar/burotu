import pytest
from fastapi.testclient import TestClient

from burotu_web.main import create_app
from burotu_web.settings import Settings

ADMIN_HOST = "admin.burotu.test"
PUBLIC_HOST = "burotu.test"


@pytest.fixture()
def app():
    return create_app(Settings(public_hosts=(PUBLIC_HOST,), admin_host=ADMIN_HOST))


@pytest.fixture()
def public(app):
    with TestClient(app, base_url=f"http://{PUBLIC_HOST}") as c:
        yield c


@pytest.fixture()
def retired(app):
    with TestClient(app, base_url=f"http://{ADMIN_HOST}") as c:
        yield c
