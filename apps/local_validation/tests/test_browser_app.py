"""End-to-end check of the actual Streamlit iframe, forms and report uploader."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SIV_BROWSER_TESTS") != "1",
                               reason="Set SIV_BROWSER_TESTS=1 with Playwright installed")


def test_live_layout_client_slicing_and_history(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    root = Path(__file__).resolve().parents[3]
    data, output = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    np.save(data / "volume.npy", np.arange(16*17*18, dtype=np.float32).reshape(16, 17, 18))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "SIV_DATA_DIR": str(data), "SIV_OUTPUT_DIR": str(output)}
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "apps/local_validation/app.py",
                               "--server.address=127.0.0.1", f"--server.port={port}", "--server.fileWatcherType=none"],
                               cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 40
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline or process.poll() is not None:
                    pytest.fail("Test Streamlit server did not start.")
                time.sleep(0.2)
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(channel=os.environ.get("SIV_BROWSER_CHANNEL") or None, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            messages = []
            page.on("websocket", lambda ws: ws.on("framesent", lambda payload: messages.append(payload)))
            page.goto(f"http://127.0.0.1:{port}")
            page.get_by_text("Open image viewer", exact=True).wait_for()
            hero = page.locator(".hero").bounding_box()
            header = page.get_by_test_id("stHeader").bounding_box()
            assert hero["y"] >= header["y"] + header["height"]
            page.get_by_text("Two files", exact=True).click()
            page.get_by_text("Open image viewer", exact=True).click()
            frame = page.frame_locator("iframe").first
            frame.locator("#status").filter(has_text="Shared window").wait_for()
            resolution = page.get_by_test_id("stSelectbox").filter(has_text="Preview resolution").get_by_role("combobox")
            resolution.focus()
            resolution.press("ArrowDown")
            page.get_by_text("Original (all voxels)", exact=True).click()
            page.get_by_text("Original loads every voxel", exact=False).wait_for()
            frame.locator("#status").filter(has_text="Shared window").wait_for()
            before = len(messages)
            for index in (0, 5, 17, 3):
                frame.locator("#slice").evaluate("(el, i) => {el.value=i;el.dispatchEvent(new Event('input'));}", index)
                frame.locator("#index").filter(has_text=str(index)).wait_for()
            frame.locator("#cmap").select_option("magma")
            page.wait_for_timeout(200)
            assert len(messages) == before, "Slice/window controls must not trigger Streamlit requests"
            # Editing a buffered setting must likewise wait for form submission.
            width = page.get_by_role("textbox", name="Intensity range width", exact=True)
            width.fill("5000")
            width.press("Tab")
            page.wait_for_timeout(200)
            assert len(messages) == before
            assert not output.exists()
            # Saved reports can be used without a CLI invocation or image upload.
            workspace = page.get_by_test_id("stSelectbox").filter(has_text="Workspace").get_by_role("combobox")
            # Finish scrolling before opening the menu. A popup opened during
            # programmatic focus scrolling can be dismissed by the scroll event.
            workspace.evaluate("el => el.scrollIntoView({behavior: 'instant', block: 'center'})")
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            workspace.click()
            page.get_by_role("option", name="Reports and history", exact=True).click()
            page.get_by_text("Reports and epoch history", exact=True).wait_for()
            page.locator('input[type="file"]').first.set_input_files({"name": "results.json", "mimeType": "application/json",
                                              "buffer": json.dumps({"mae": 0.2, "ssim": 0.8}).encode()})
            page.get_by_role("button", name="Append this result to history", exact=True).click()
            page.get_by_text("Recorded step 0 in history.json.", exact=True).wait_for()
            assert json.loads((output / "history.json").read_text())["records"][0]["metrics"]["mae"] == .2
            page.get_by_role("button", name="Export JSON, CSV, PDF, LaTeX and plots", exact=True).click()
            page.get_by_role("button", name="results_bundle.zip", exact=True).wait_for()
            assert len(list(output.glob("validation_*/results.pdf"))) == 1
            assert page.get_by_test_id("stException").count() == 0
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
