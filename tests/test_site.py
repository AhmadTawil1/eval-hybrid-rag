import pathlib
import re

from fastapi.testclient import TestClient

from app import main
from app.schemas import MAX_QUERY_CHARS

client = TestClient(main.app)  # no lifespan: no Qdrant or model needed

STATIC = pathlib.Path("app/static")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
JS = (STATIC / "app.js").read_text(encoding="utf-8")


def test_the_page_is_served_at_the_root_with_a_strict_csp():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "Try it" in response.text


def test_the_page_has_no_inline_script_or_style():
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", HTML)
    assert " style=" not in HTML and "<style" not in HTML


def test_page_assets_and_media_are_served():
    expectations = {
        "/static/app.js": "javascript",
        "/static/style.css": "text/css",
        "/charts/02_ragas_metrics.png": "image/png",
        "/charts/06_answerable_outcomes.png": "image/png",
        "/assets/architecture.png": "image/png",
        "/report.pdf": "application/pdf",
    }
    for path, content_type in expectations.items():
        response = client.get(path)
        assert response.status_code == 200, path
        assert content_type in response.headers["content-type"], path


def test_security_headers_are_present_on_api_responses_too():
    response = client.get("/api/v1/metrics")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" not in response.headers  # would break the Swagger page at /docs


def test_the_script_never_inserts_server_text_as_html():
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert forbidden not in JS, forbidden


def test_every_element_id_the_script_uses_exists_in_the_page():
    used = set(re.findall(r'\$\("([\w-]+)"\)', JS))
    present = set(re.findall(r'\bid="([\w-]+)"', HTML))
    assert used <= present, used - present


def test_example_questions_fit_the_query_limit():
    block = JS.split("const EXAMPLES = [", 1)[1].split("];", 1)[0]
    examples = re.findall(r'"((?:[^"\\]|\\.)*)"', block)
    assert len(examples) == 3
    assert all(0 < len(e) <= MAX_QUERY_CHARS for e in examples)


def test_the_page_links_only_to_things_that_exist():
    for path in ("/report.pdf", "/docs"):
        assert f'href="{path}"' in HTML
        assert client.get(path).status_code == 200


def test_report_returns_404_when_the_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "REPORT_PATH", tmp_path / "missing.pdf")
    assert client.get("/report.pdf").status_code == 404
