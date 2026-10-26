from tests.web.conftest import book, csrf


def test_login_required(admin):
    for path in ("/", "/sites", "/export.csv", "/bookings/BRT-000000"):
        r = admin.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login"


def test_wrong_password_and_throttle(admin):
    for _ in range(8):
        r = admin.post("/login", data={"csrf": csrf(admin, "/login"), "username": "admin", "password": "bad"})
        assert r.status_code == 401
    r = admin.post("/login", data={"csrf": csrf(admin, "/login"), "username": "admin", "password": "pw"})
    assert r.status_code == 429


def test_admin_not_indexed(admin):
    assert "Disallow: /" in admin.get("/robots.txt").text


def test_sample_tracking(public, logged_admin):
    ref = book(public).headers["location"].split("booked=")[1].split("#")[0]

    listing = logged_admin.get("/")
    assert ref in listing.text

    detail = logged_admin.get(f"/bookings/{ref}")
    assert detail.status_code == 200 and "Requested" in detail.text
    token = csrf(logged_admin, f"/bookings/{ref}")
    for status in ("confirmed", "drawn", "in_lab"):
        r = logged_admin.post(
            f"/bookings/{ref}/status", data={"csrf": token, "status": status, "note": f"stage {status}"}
        )
        assert r.status_code == 200
    page = logged_admin.get(f"/bookings/{ref}").text
    assert "In the lab" in page and "stage drawn" in page

    assert logged_admin.post(f"/bookings/{ref}/status", data={"csrf": token, "status": "bogus"}).status_code == 400


def test_cancel_frees_the_slot(public, logged_admin):
    refs = [book(public).headers["location"].split("booked=")[1].split("#")[0] for _ in range(2)]
    assert book(public).status_code == 422
    token = csrf(logged_admin, "/")
    logged_admin.post(f"/bookings/{refs[0]}/status", data={"csrf": token, "status": "cancelled"})
    assert book(public).status_code == 303


def test_csv_export_neutralises_formulas(public, logged_admin):
    book(public, first_name="=HYPERLINK(1)")
    body = logged_admin.get("/export.csv").text
    assert "'=HYPERLINK(1)" in body


def test_site_management(logged_admin):
    token = csrf(logged_admin, "/sites")
    r = logged_admin.post(
        "/sites", data={"csrf": token, "name": "Burotu Point Test", "city": "Nice", "address": "1 Test Street"}
    )
    assert r.status_code == 200 and "Burotu Point Test" in r.text
    bad = logged_admin.post(
        "/sites/1",
        data={
            "csrf": token,
            "name": "a",
            "city": "b",
            "address": "c",
            "open_from": "12:00",
            "open_to": "08:00",
            "slot_minutes": 15,
            "capacity": 2,
        },
    )
    assert bad.status_code == 400
