import re
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from burotu_web.main import create_app
from burotu_web.settings import Settings

ADMIN_HOST = "admin.burotu.test"
PUBLIC_HOST = "burotu.test"


@pytest.fixture()
def app(tmp_path):
    settings = Settings(
        public_hosts=(PUBLIC_HOST,),
        admin_host=ADMIN_HOST,
        admin_user="admin",
        admin_password="pw",
        secret_key="test-secret",
        db_path=tmp_path / "site.db",
    )
    return create_app(settings)


@pytest.fixture()
def public(app):
    with TestClient(app, base_url=f"http://{PUBLIC_HOST}") as c:
        yield c


@pytest.fixture()
def admin(app):
    with TestClient(app, base_url=f"http://{ADMIN_HOST}") as c:
        yield c


@pytest.fixture()
def logged_admin(admin):
    r = admin.post("/login", data={"csrf": csrf(admin, "/login"), "username": "admin", "password": "pw"})
    assert r.status_code == 200
    return admin


def csrf(client: TestClient, url: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', client.get(url).text).group(1)


def next_weekday() -> str:
    d = date.today() + timedelta(days=2)
    while d.weekday() > 4:
        d += timedelta(days=1)
    return d.isoformat()


def book(client: TestClient, slot: str = "08:00", site: int = 1, **overrides):
    data = {
        "csrf": csrf(client, "/en/"),
        "site_id": site,
        "day": next_weekday(),
        "slot": slot,
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": "ada@example.com",
        "phone": "+44 20 0000 0000",
        "birth_year": "1960",
        "doctor": "Dr Babbage",
        "consent": "1",
        **overrides,
    }
    return client.post("/en/book", data=data, follow_redirects=False)
