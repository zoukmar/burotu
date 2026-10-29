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


def _ref(public):
    return book(public).headers["location"].split("booked=")[1].split("#")[0]


def test_analyse_a_demo_sample(public, logged_admin):
    ref = _ref(public)
    token = csrf(logged_admin, f"/bookings/{ref}")
    r = logged_admin.post(f"/bookings/{ref}/analyse", data={"csrf": token, "demo": "BRT-DEMO-05"})
    assert r.status_code == 200
    page = r.text
    assert "Lab result" in page and "Signal detected" in page and "Kidney" in page
    assert "Every model on this sample" in page and "Logistic regressions" in page and "Follow-up" in page
    why = page.split('id="why"')[1].split("</table>")[0]
    assert "Rank among healthy people" in why and "Marker sites lit" in why and "Kidney" in why
    letter = page.split('id="letter"')[1].split("</blockquote>")[0]
    assert "unusual methylation signal" in letter and "cancer" not in letter.lower()
    assert "Simulation ground truth: Kidney" in page
    assert "Analysed" in page  # stage moved forward


def test_quality_failure_does_not_give_a_result(public, logged_admin):
    ref = _ref(public)
    token = csrf(logged_admin, f"/bookings/{ref}")
    page = logged_admin.post(f"/bookings/{ref}/analyse", data={"csrf": token, "demo": "BRT-DEMO-08"}).text
    assert "No result" in page and "new blood draw" in page


def test_unreadable_upload_is_explained(public, logged_admin):
    ref = _ref(public)
    token = csrf(logged_admin, f"/bookings/{ref}")
    files = {"lab_file": ("bad.tsv", b"not a lab file", "text/tab-separated-values")}
    page = logged_admin.post(f"/bookings/{ref}/analyse", data={"csrf": token}, files=files).text
    assert "This file cannot be read" in page


def test_tracking_speaks_french_and_spanish(public, logged_admin):
    ref = _ref(public)
    token = csrf(logged_admin, f"/bookings/{ref}")
    logged_admin.post(f"/bookings/{ref}/analyse", data={"csrf": token, "demo": "BRT-DEMO-05"})
    r = logged_admin.get(f"/lang/fr?next=/bookings/{ref}", follow_redirects=True)
    assert "Résultat du laboratoire" in r.text and "Signal détecté" in r.text and "Rein" in r.text
    r = logged_admin.get(f"/bookings/{ref}?letter=es")
    letter = r.text.split('id="letter"')[1].split("</blockquote>")[0]
    assert "señal de metilación inusual" in letter and "cáncer" not in letter.lower()
    r = logged_admin.get("/lang/es?next=/", follow_redirects=True)
    assert "Citas" in r.text
    r = logged_admin.get("/lang/fr?next=//evil.example", follow_redirects=False)
    assert r.headers["location"] == "/"


def test_report_pdf_in_each_language(public, logged_admin):
    ref = _ref(public)
    assert logged_admin.get(f"/bookings/{ref}/report.pdf").status_code == 404  # nothing analysed yet
    token = csrf(logged_admin, f"/bookings/{ref}")
    logged_admin.post(f"/bookings/{ref}/analyse", data={"csrf": token, "demo": "BRT-DEMO-05"})
    for lang in ("en", "fr", "es"):
        r = logged_admin.get(f"/bookings/{ref}/report.pdf?lang={lang}")
        assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
        assert r.content.startswith(b"%PDF") and len(r.content) > 3000


def test_report_pdf_needs_sign_in(public, admin):
    ref = _ref(public)
    assert admin.get(f"/bookings/{ref}/report.pdf", follow_redirects=False).status_code in (302, 303, 401)


def test_models_page_lists_every_run(logged_admin):
    r = logged_admin.get("/models")
    assert r.status_code == 200
    for name in ("Muse Glimmer v1", "Muse Glimmer v2", "Muse Glimmer v3", "Shield", "Galleri"):
        assert name in r.text


def test_ask_a_question_about_the_result(public, logged_admin):
    ref = _ref(public)
    token = csrf(logged_admin, f"/bookings/{ref}")
    logged_admin.post(f"/bookings/{ref}/analyse", data={"csrf": token, "demo": "BRT-DEMO-05"})
    token = csrf(logged_admin, f"/bookings/{ref}")
    data = {"csrf": token, "question": "Should I repeat the test?", "lang": "fr", "audience": "patient"}
    page = logged_admin.post(f"/bookings/{ref}/ask", data=data).text
    assert "Should I repeat the test?" in page and "Posez-la au médecin" in page  # offline in tests


def test_map_page_lists_models_data_and_latest_analyses(public, logged_admin):
    ref = _ref(public)
    token = csrf(logged_admin, f"/bookings/{ref}")
    logged_admin.post(f"/bookings/{ref}/analyse", data={"csrf": token, "demo": "BRT-DEMO-05"})
    page = logged_admin.get("/map").text
    assert "Map of Burotu" in page and "Muse Glimmer v5" in page and "CpGPT" in page
    assert "TCGA tumours" in page and ref in page
    fr = logged_admin.get("/lang/fr?next=/map", follow_redirects=True).text
    assert "Carte de Burotu" in fr and "Appris sur" in fr
