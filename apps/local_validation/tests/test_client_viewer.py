"""Payload and actual canvas regression tests (browser tests are optional locally)."""
import base64
import gzip
import json
import os
import numpy as np
import pytest
from apps.local_validation.viewer import prepare_image, extract_slice
from apps.local_validation.client_viewer import viewer_html


def payload(html):
    return json.loads(html.split("const cfg=", 1)[1].split(";\n", 1)[0])


def test_payload_is_bounded_exact_for_small_arrays(tmp_path):
    a = np.arange(4*5*6, dtype=np.float32).reshape(4, 5, 6)
    np.save(tmp_path / "v.npy", a)
    image = prepare_image(tmp_path / "v.npy")
    html = viewer_html(image, image)
    p = payload(html)
    decoded = np.frombuffer(gzip.decompress(base64.b64decode(p["volumes"][0]["data"])), dtype="<f4").reshape(a.shape)
    np.testing.assert_array_equal(a, decoded)
    assert p["compatible"] and p["step"] == 1
    assert "https://" not in html and "fetch(" not in html
    np.save(tmp_path / "large.npy", np.zeros((300, 5, 6), dtype=np.float32))
    large = prepare_image(tmp_path / "large.npy")
    assert payload(viewer_html(large, large))["step"] == 3
    with pytest.raises(ValueError):
        viewer_html(image, image, threshold=float("nan"))
    np.save(tmp_path / "boundary.npy", np.array([[0.5-1e-12, 0.5], [1., 0.]], dtype=np.float64))
    mask = prepare_image(tmp_path / "boundary.npy")
    encoded = payload(viewer_html(mask, mask, masks=True))["volumes"][0]["data"]
    assert np.frombuffer(gzip.decompress(base64.b64decode(encoded)), dtype="<f4").tolist() == [0., 1., 1., 0.]


def test_original_keeps_every_voxel_and_limits_do_not_upscale(tmp_path):
    array = np.arange(300*5*6, dtype=np.float32).reshape(300, 5, 6)
    np.save(tmp_path / "volume.npy", array)
    image = prepare_image(tmp_path / "volume.npy")
    for limit, step in ((128, 3), (256, 2), (None, 1)):
        p = payload(viewer_html(image, image, max_side=limit))
        assert p["step"] == step
        for volume in p["volumes"]:
            values = np.frombuffer(gzip.decompress(base64.b64decode(volume["data"])), dtype="<f4").reshape(volume["shape"])
            np.testing.assert_array_equal(values, array[::step, ::step, ::step])
    # A 192-axis input fits a 256 limit and must never be enlarged to 256.
    np.save(tmp_path / "smaller.npy", array[:192])
    smaller = prepare_image(tmp_path / "smaller.npy")
    assert payload(viewer_html(smaller, smaller, max_side=256))["volumes"][0]["shape"] == [192, 5, 6]
    np.testing.assert_array_equal(np.load(tmp_path / "volume.npy"), array)


def test_original_memory_guard_runs_before_encoding(tmp_path, monkeypatch):
    from apps.local_validation import client_viewer
    a = np.zeros((20, 20), dtype=np.float32)
    np.save(tmp_path / "volume.npy", a)
    image = prepare_image(tmp_path / "volume.npy")
    # Exercise the memory guard without allocating a large medical volume.
    monkeypatch.setattr(client_viewer, "MAX_PREVIEW_BYTES", 2*a.nbytes)
    assert payload(viewer_html(image, image, max_side=None))["step"] == 1
    monkeypatch.setattr(client_viewer, "MAX_PREVIEW_BYTES", 2*a.nbytes-1)
    def should_not_encode(*args, **kwargs):
        pytest.fail("Encoding should not start above the memory limit")
    monkeypatch.setattr(client_viewer.gzip, "compress", should_not_encode)
    with pytest.raises(ValueError, match="browser safety limit"):
        viewer_html(image, image, max_side=None)


@pytest.mark.skipif(os.environ.get("SIV_BROWSER_TESTS") != "1", reason="Set SIV_BROWSER_TESTS=1 and install Playwright's browser for canvas checks")
def test_canvas_all_planes_windows_and_masks(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    import nibabel as nib
    array = np.arange(4*5*6, dtype=np.float32).reshape(4, 5, 6)
    nib.save(nib.Nifti1Image(array, np.diag([-2., 3., 4., 1.])), tmp_path / "volume.nii.gz")
    real = prepare_image(tmp_path / "volume.nii.gz")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("SIV_BROWSER_CHANNEL") or None, headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content(viewer_html(real, real, max_side=None))
        page.wait_for_function("window.viewerReady === true")
        for axis in (0, 1, 2):
            page.select_option("#plane", str(axis))
            for index in (0, real.array.shape[axis]-1):
                page.locator("#slice").evaluate("(el, value) => {el.value=value; el.dispatchEvent(new Event('input'));}", index)
                page.wait_for_function("i => document.getElementById('index').textContent === String(i)", arg=index)
                # Compare JS voxel lookup against the established Python orientation.
                expected = np.flipud(extract_slice(real, axis, index)[0])
                values = page.evaluate("([a,i,w,h]) => Array.from({length:h}, (_,y) => Array.from({length:w}, (_,x) => sliceValue(cfg.volumes[0],a,i,x,y)))", [axis,index,expected.shape[1],expected.shape[0]])
                np.testing.assert_array_equal(values, expected)
        page.fill("#high", "-1")
        page.wait_for_function("document.getElementById('status').textContent.includes('must exceed')")
        assert not errors
        np.save(tmp_path / "mask.npy", np.array([[0., .5], [1., 0.]], dtype=np.float32))
        mask = prepare_image(tmp_path / "mask.npy")
        page.goto("about:blank")
        page.set_content(viewer_html(mask, mask, masks=True))
        page.wait_for_function("window.viewerReady === true")
        pixel = page.locator("#delta").evaluate("el => Array.from(el.getContext('2d').getImageData(1,0,1,1).data)")
        assert pixel == [236, 245, 252, 255]
        assert not page.locator("#sliceLabel").is_visible()
        assert not page.locator("#window").is_visible()
        assert not errors
        browser.close()
