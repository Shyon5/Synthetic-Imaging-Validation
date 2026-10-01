"""Widget tests use labels, not positional indices, to tolerate layout changes."""
from pathlib import Path
import pytest
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from apps.local_validation.service import make_demo

APP = Path(__file__).resolve().parents[1] / "app.py"


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def launch(tmp_path, monkeypatch):
    monkeypatch.setenv("SIV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SIV_OUTPUT_DIR", str(tmp_path / "results"))
    return AppTest.from_file(str(APP), default_timeout=45).run()


def test_demo_browser_workflow(tmp_path, monkeypatch):
    app = launch(tmp_path, monkeypatch)
    widget(app, "button", "Preview pairing").click().run()
    assert not app.exception and len(app.dataframe) == 1
    widget(app, "multiselect", "Experimental scores (optional)").select("similarity")
    widget(app, "button", "Apply evaluation settings").click().run()
    widget(app, "button", "Run validation").click().run()
    assert not app.exception and not app.error
    assert len(app.get("download_button")) == 6
    assert app.metric[0].value == "4"
    assert "similarity_score.value" in app.session_state["completed"][0]["summary"]["metrics"]
    app.run()
    assert len(list((tmp_path / "results").glob("*/results.json"))) == 1


def test_mask_and_missing_data(tmp_path, monkeypatch):
    app = launch(tmp_path, monkeypatch)
    widget(app, "radio", "What are you evaluating?").set_value("Binary masks").run()
    widget(app, "button", "Run validation").click().run()
    assert not app.exception and not app.error
    widget(app, "radio", "Input mode").set_value("Two files").run()
    widget(app, "button", "Run validation").click().run()
    assert not app.exception and app.error


def test_manifest_and_directory(tmp_path, monkeypatch):
    make_demo(tmp_path / "data")
    app = launch(tmp_path, monkeypatch)
    widget(app, "radio", "Input mode").set_value("CSV manifest").run()
    widget(app, "text_input", "Group results by column (optional)").set_value("label").run()
    widget(app, "button", "Run validation").click().run()
    assert not app.exception and not app.error
    assert app.metric[2].value == "2"
    widget(app, "radio", "Input mode").set_value("Two folders").run()
    widget(app, "button", "Run validation").click().run()
    assert not app.exception and not app.error
    assert app.metric[0].value == "4" and app.metric[2].value == "0"


def test_viewer_cache_and_labels(tmp_path, monkeypatch):
    app = launch(tmp_path, monkeypatch)
    assert "MS-SSIM — Multi-scale Structural Similarity" in widget(app, "multiselect", "Metrics").options
    app.toggle(key="open_viewer").set_value(True).run()
    assert not app.exception and not app.error
    cached = app.session_state["_preview_cache"]
    assert "sliceValue" in cached[1]
    app.run()
    assert app.session_state["_preview_cache"][0] == cached[0]
    app.selectbox(key="preview_case").set_value(2).run()
    assert app.session_state["_preview_cache"][0] != cached[0]
    assert not (tmp_path / "results").exists()
    app.toggle(key="open_viewer").set_value(False).run()
    assert "_preview_cache" not in app.session_state


def test_original_viewer_option_warning_and_plain_title(tmp_path, monkeypatch):
    from apps.local_validation.tests.test_client_viewer import payload
    app = launch(tmp_path, monkeypatch)
    assert any("<h1>Synthetic Imaging Validation</h1>" in item.value for item in app.markdown)
    assert not any("From image pairs to clear results" in item.value for item in app.markdown)
    app.toggle(key="open_viewer").set_value(True).run()
    control = widget(app, "selectbox", "Preview resolution")
    assert control.options == ["Up to 128 per axis", "Up to 256 per axis", "Original (all voxels)"]
    control.set_value("original").run()
    assert not app.exception and not app.error
    assert any("Original loads every voxel" in warning.value for warning in app.warning)
    assert payload(app.session_state["_preview_cache"][1])["step"] == 1
    assert not (tmp_path / "results").exists()
    widget(app, "selectbox", "Preview resolution").set_value(256).run()
    assert not any("Original loads every voxel" in warning.value for warning in app.warning)


def test_bounds_inspection_and_custom_score_interval(tmp_path, monkeypatch):
    import numpy as np
    root = tmp_path / "data"
    root.mkdir()
    np.save(root / "ct.npy", np.linspace(-1000, 2000, 64*64).reshape(64, 64))
    app = launch(tmp_path, monkeypatch)
    widget(app, "radio", "Input mode").set_value("Two files").run()
    widget(app, "button", "Inspect intensity bounds").click().run()
    assert any("-1000" in info.value for info in app.info)
    widget(app, "multiselect", "Experimental scores (optional)").select("similarity")
    widget(app, "number_input", "Score interval: lower bound").set_value(-1000.)
    widget(app, "number_input", "Score interval: upper bound").set_value(2000.)
    widget(app, "button", "Apply evaluation settings").click().run()
    widget(app, "button", "Run validation").click().run()
    assert not app.exception and not app.error
    assert app.session_state["completed"][0]["pairs"][0]["metrics"]["similarity_score"]["value"] == pytest.approx(100.)


def test_workspace_change_and_extra_pages(tmp_path, monkeypatch):
    root = tmp_path / "other"
    make_demo(root)
    app = launch(tmp_path, monkeypatch)
    widget(app, "text_input", "Data folder").set_value(str(root))
    widget(app, "text_input", "Results folder").set_value(str(tmp_path / "other_outputs"))
    widget(app, "button", "Apply folders").click().run()
    assert app.session_state["workspace"][0] == root
    for page in ("Feature distributions", "Independent intensity cohorts", "Reports and history"):
        widget(app, "selectbox", "Workspace").set_value(page).run()
        assert not app.exception and not app.error


def test_feature_page_evaluation(tmp_path, monkeypatch):
    import numpy as np
    root = tmp_path / "data"
    root.mkdir()
    np.save(root / "features.npy", np.random.default_rng(3).normal(size=(12, 4)))
    app = launch(tmp_path, monkeypatch)
    widget(app, "selectbox", "Workspace").set_value("Feature distributions").run()
    widget(app, "button", "Evaluate features").click().run()
    assert not app.exception and not app.error
    assert app.session_state["advanced_completed"][0]["frechet"] == pytest.approx(0, abs=1e-10)
