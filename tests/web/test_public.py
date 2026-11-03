import pytest


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("fr-FR,fr;q=0.9,en;q=0.8", "/fr/"),
        ("es-ES,es;q=0.9", "/es/"),
        ("de-DE,de;q=0.9,es;q=0.7,fr;q=0.5", "/es/"),  # first supported language wins
        ("en;q=0.3,fr;q=0.8", "/fr/"),  # q-values beat header order
        ("de-DE,ja;q=0.9", "/en/"),  # nothing supported: English
        ("", "/en/"),  # no header: English
    ],
)
def test_root_follows_browser_language(public, header, expected):
    r = public.get("/", headers={"accept-language": header}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == expected


def test_home_in_three_languages(public):
    for lang, word in [
        ("en", "We taught a language model to read DNA."),
        ("fr", "Nous avons appris"),
        ("es", "Enseñamos"),
    ]:
        r = public.get(f"/{lang}/")
        assert r.status_code == 200 and word in r.text
        for anchor in ('id="how"', 'id="runs"', 'id="demo"', 'id="results"', 'id="engine"'):
            assert anchor in r.text
    assert public.get("/de/").status_code == 404


def test_home_tells_the_six_runs_and_links_everything(public):
    html = public.get("/en/").text
    for n in range(1, 7):
        assert f">v{n}<" in html
    assert 'datetime="2026-10-27"' in html and 'datetime="2026-11-02"' in html
    assert "youtu.be/LQVoz8USAHo" in html
    assert "github.com/zoukmar/burotu" in html and "huggingface.co/zoukmar75/burotu-glimmer" in html
    assert "Not a medical device" in html


def test_no_booking_left(public):
    html = public.get("/en/").text
    assert "Book a blood draw" not in html and 'id="book"' not in html and "/static/book.js" not in html
    r = public.get("/en/book", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == "/en/"
    assert public.post("/en/book").status_code == 405
    assert public.get("/api/slots?site=1&day=2026-11-05").status_code == 404


def test_demo_links_lead_to_the_demo_section(public):
    r = public.get("/demo", headers={"accept-language": "fr"}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/fr/#demo"
    r = public.get("/en/demo", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == "/en/#demo"


def test_security_headers(public):
    h = public.get("/en/").headers
    assert "default-src 'self'" in h["content-security-policy"]
    assert h["x-frame-options"] == "DENY"


def test_retired_tracking_host_redirects_to_the_site(retired):
    r = retired.get("/login", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == "https://burotu.com/"


def test_technology_page_in_three_languages(public):
    for lang, word in (("en", "Under the hood"), ("fr", "Sous le capot"), ("es", "Bajo el capó")):
        r = public.get(f"/{lang}/technology")
        assert r.status_code == 200 and word in r.text
        assert "Muse Glimmer" in r.text and "70%" in r.text  # Glimmer's tissue accuracy at 1%


def test_live_demo_reads_a_lab_file_and_reveals_the_truth(public):
    titles = [("en", "Watch a Meta model read DNA"), ("fr", "Regardez un modèle de Meta"), ("es", "Mire a un modelo")]
    for lang, word in titles:
        r = public.get(f"/{lang}/")  # the demo is a section of the home page
        assert r.status_code == 200 and word in r.text and "BRT-DEMO-05" in r.text and 'id="demo"' in r.text
    d = public.post("/api/demo/BRT-DEMO-05?lang=fr").json()
    assert d["quality"]["status"] == "pass" and d["models"]["classic"]["signal"]
    assert d["truth"]["label"] == "Kidney" and "Usage recherche" in d["report"]["text"]
    assert "Kidney" in d["tissues"] and d["tissues"]["Kidney"] == "rein"
    assert public.post("/api/demo/BRT-DEMO-06").status_code == 404  # not offered on the public page
    shallow = public.post("/api/demo/BRT-DEMO-08").json()
    assert shallow["quality"]["status"] == "fail" and shallow["models"] == {}


def test_paper_is_served(public):
    r = public.get("/paper", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/static/burotu-paper.pdf"
    pdf = public.get("/static/burotu-paper.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert 'href="/paper"' in public.get("/en/").text
