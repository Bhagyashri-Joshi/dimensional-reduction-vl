"""Integration tests: drive the real Streamlit pages with synthetic data (no network)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    "pages/1_Overview.py", "pages/2_Aim.py", "pages/3_Theory.py", "pages/4_Pre_test.py",
    "pages/5_Simulation.py", "pages/6_Procedure.py", "pages/7_Observations.py", "pages/8_Exercises.py",
    "pages/9_Post_test.py", "pages/10_References.py", "pages/11_Contributors.py", "pages/12_Feedback.py",
]


@pytest.fixture()
def app(har_dir, monkeypatch):
    monkeypatch.setenv("HAR_DATA_DIR", str(har_dir))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]


def _submit(at, label):
    btn = next(b for b in at.button if label in b.label)
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("algo", ["PCA", "Truncated SVD", "t-SNE", "UMAP"])
def test_simulation_runs_each_algorithm(app, algo):
    app.switch_page("pages/5_Simulation.py").run()
    app.selectbox(key="sim_algo").select(algo).run()
    _submit(app, "Run algorithm")
    res = app.session_state["sim_result"]
    assert res.algorithm == algo and res.embedding.shape[0] > 0
    assert len(app.session_state["sim_history"]) == 1


def test_classification_comparison_and_observations(app):
    app.switch_page("pages/5_Simulation.py").run()
    app.select_slider(key="cmp_dims").set_value(5).run()
    _submit(app, "Run comparison")
    results = app.session_state["cmp_results"]
    assert [r.method for r in results] == ["Baseline", "PCA", "Truncated SVD", "UMAP"]
    app.switch_page("pages/7_Observations.py").run()
    assert not app.exception
    assert len(app.dataframe) >= 1


def test_quiz_scoring_flow(app):
    from src.content import PRE_TEST

    app.switch_page("pages/4_Pre_test.py").run()
    for i, q in enumerate(PRE_TEST):
        app.radio(key=f"pre_q{i}").set_value(q.options[q.answer])
    _submit(app, "Submit answers")
    assert app.session_state["pre_result"]["score"] == 10
