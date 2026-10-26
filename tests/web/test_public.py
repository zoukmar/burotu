from fastapi.testclient import TestClient

from tests.web.conftest import PUBLIC_HOST, book, csrf, next_weekday


def test_root_redirects_by_language(public):
    r = public.get("/", headers={"accept-language": "es-ES,es;q=0.9"}, follow_redirects=False)
    assert r.headers["location"] == "/es/"
    assert public.get("/", follow_redirects=False).headers["location"] == "/en/"


def test_onepager_in_three_languages(public):
    for lang, word in [("en", "Book a blood draw"), ("fr", "Prendre rendez-vous"), ("es", "Reservar extracción")]:
        r = public.get(f"/{lang}/")
        assert r.status_code == 200
        assert word in r.text and 'id="book"' in r.text
    assert public.get("/de/").status_code == 404


def test_security_headers(public):
    h = public.get("/en/").headers
    assert "default-src 'self'" in h["content-security-policy"]
    assert h["x-frame-options"] == "DENY"


def test_admin_routes_not_on_public_host(public):
    assert public.get("/login").status_code == 404
    assert public.get("/export.csv").status_code == 404


def test_booking_flow_and_confirmation(public):
    r = book(public)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("/en/?booked=BRT-") and loc.endswith("#book")
    page = public.get(loc)
    assert "Booking confirmed" in page.text


def test_confirmation_only_for_the_booking_session(app, public):
    loc = book(public).headers["location"]
    with TestClient(app, base_url=f"http://{PUBLIC_HOST}") as stranger:
        assert "Booking confirmed" not in stranger.get(loc).text


def test_slot_capacity_is_enforced(public):
    assert book(public).status_code == 303
    assert book(public).status_code == 303  # 2 personnes par créneau par défaut
    assert book(public).status_code == 422
    slots = public.get(f"/api/slots?site=1&day={next_weekday()}").json()["slots"]
    assert "08:00" not in slots and "08:15" in slots


def test_validation_errors(public):
    assert book(public, consent="").status_code == 422
    assert book(public, email="nope").status_code == 422
    assert book(public, birth_year="1800").status_code == 422
    assert book(public, website="spam").status_code == 422  # pot de miel
    assert book(public, slot="03:00").status_code == 422  # hors horaires


def test_csrf_required(public):
    csrf(public, "/en/")
    assert public.post("/en/book", data={"csrf": "forged"}).status_code == 400


def test_marketing_headline_and_honest_disclaimer(public):
    page = public.get("/en/").text
    assert ">50 cancers.</span>" in page and ">One blood draw.</span>" in page
    assert "Not a clinically validated test" in page
    assert "€2,690" in page


def test_doctor_is_optional(public):
    assert book(public, doctor="").status_code == 303
