import filecmp
import json
import pathlib
import re

import pytest

from app.generation import INSUFFICIENT

SITE = pathlib.Path("site")
HTML = (SITE / "index.html").read_text(encoding="utf-8")
JS = (SITE / "site.js").read_text(encoding="utf-8")
EXPLORER = json.loads((SITE / "data" / "explorer.json").read_text(encoding="utf-8"))
RESULTS = json.loads(pathlib.Path("eval/results.json").read_text(encoding="utf-8"))
MODES = ["dense", "sparse", "hybrid"]


def test_explorer_has_every_question_in_every_mode():
    questions = EXPLORER["questions"]
    assert len(questions) == RESULTS["n"]
    assert len({q["id"] for q in questions}) == len(questions)
    sets = [q["set"] for q in questions]
    assert (sets.count("direct"), sets.count("multihop"), sets.count("trap")) == (20, 20, 10)
    for q in questions:
        assert set(q["modes"]) == set(MODES)
        for data in q["modes"].values():
            assert data["answer"].strip()
            assert len(data["passages"]) == RESULTS["k"]
            assert data["refused"] == (data["answer"].strip() == INSUFFICIENT)
            assert 0 <= data["evidence_found"] <= q["evidence_needed"]
            assert sum(p["needed"] for p in data["passages"]) == data["evidence_found"]


def test_trap_questions_need_no_evidence_and_answerable_ones_have_scores():
    for q in EXPLORER["questions"]:
        if q["set"] == "trap":
            assert q["evidence_needed"] == 0
            assert all("ragas" not in d for d in q["modes"].values())
        else:
            assert q["evidence_needed"] >= 1
            assert all(set(d["ragas"]) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"} for d in q["modes"].values())


@pytest.mark.parametrize("mode", MODES)
def test_explorer_agrees_with_the_published_results(mode):
    questions = EXPLORER["questions"]
    trap = [q["modes"][mode]["refused"] for q in questions if q["set"] == "trap"]
    answerable = [q["modes"][mode]["refused"] for q in questions if q["set"] != "trap"]
    published = RESULTS["modes"][mode]
    assert sum(trap) == published["trap_refusal"]["refused"]
    assert sum(answerable) == published["false_refusal"]["refused"]
    for metric, value in published["ragas"].items():
        scored = [q["modes"][mode]["ragas"][metric] for q in questions if q["set"] != "trap"]
        assert sum(scored) / len(scored) == pytest.approx(value, abs=0.002)


def test_site_copies_are_not_stale():
    assert json.loads((SITE / "data" / "results.json").read_text(encoding="utf-8")) == RESULTS
    assert filecmp.cmp("paper/experiment_report.pdf", SITE / "report.pdf", shallow=False)
    assert filecmp.cmp("assets/architecture.png", SITE / "img" / "architecture.png", shallow=False)
    for chart in ("02_ragas_metrics", "04_latency", "05_evidence_retrieved", "06_answerable_outcomes"):
        assert filecmp.cmp(f"eval/charts/{chart}.png", SITE / "img" / f"{chart}.png", shallow=False), chart


def test_every_local_reference_exists_and_paths_are_relative():
    refs = re.findall(r'(?:href|src)="([^"#]+)"', HTML)
    local = [r for r in refs if not r.startswith(("http://", "https://", "data:"))]
    assert local
    for ref in local:
        assert not ref.startswith("/"), f"absolute path breaks hosting under a sub-path: {ref}"
        assert (SITE / ref).is_file(), ref
    for path in re.findall(r'loadJson\("([^"]+)"\)', JS):
        assert not path.startswith("/") and (SITE / path).is_file(), path


def test_page_declares_a_strict_csp_and_has_no_inline_code():
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', HTML).group(1)
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", HTML)
    assert " style=" not in HTML and "<style" not in HTML


def test_script_never_inserts_data_as_html():
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert forbidden not in JS, forbidden


def test_every_element_id_the_script_uses_exists_in_the_page():
    used = set(re.findall(r'\$\("([\w-]+)"\)', JS))
    present = set(re.findall(r'\bid="([\w-]+)"', HTML))
    assert used <= present, used - present


def test_featured_and_default_questions_exist():
    ids = {q["id"] for q in EXPLORER["questions"]}
    featured = set(re.findall(r'data-featured="(\w+)"', HTML))
    default = re.search(r'DEFAULT_QUESTION = "(\w+)"', JS).group(1)
    assert featured and featured <= ids
    assert default in ids
