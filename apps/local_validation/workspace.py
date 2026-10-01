"""Workspace selection without granting the container additional host access."""
import os
from pathlib import Path
import streamlit as st


def select_folder(base, value, *, confined=False, must_exist=False):
    """Resolve a folder, including symlinks, before accepting it as a workspace."""
    base = Path(base).resolve()
    target = (base / value).resolve() if confined else Path(value).expanduser().resolve()
    if confined and not target.is_relative_to(base):
        raise ValueError("Choose a folder inside the existing Docker mount.")
    if must_exist and not target.is_dir():
        raise ValueError("The data folder must already exist.")
    if target.exists() and not target.is_dir():
        raise ValueError("The selected path is not a folder.")
    return target


def compose_environment(data, results):
    """Compose .env values use single quotes to avoid dollar interpolation."""
    paths = [str(v).strip().replace("\\", "/") for v in (data, results)]
    if any(not v or any(c in v for c in "\n\r'\x00") for v in paths):
        raise ValueError("Use non-empty paths without newlines or single quotes.")
    return f"SIV_DATA_DIR='{paths[0]}'\nSIV_OUTPUT_DIR='{paths[1]}'\n"


def workspace_controls(default_data, default_output):
    """Apply folder changes together. Changing host mounts still needs Docker Compose."""
    confined = os.environ.get("SIV_CONTAINER") == "1"
    current_data, current_output = st.session_state.get("workspace", (default_data, default_output))
    with st.sidebar.expander("Change folders"):
        st.caption("Choose a subfolder within the data and results folders already connected to Docker." if confined else "Enter folder paths on the computer running this app.")
        with st.form("workspace_form"):
            data = st.text_input("Data folder", value=str(current_data), help="Existing folder. In Docker only mounted paths are accessible.")
            output = st.text_input("Results folder", value=str(current_output), help="Created when a report is saved. Docker: must remain within the writable results mount.")
            apply = st.form_submit_button("Apply folders")
        if apply:
            try:
                current_data = select_folder(default_data, data, confined=confined, must_exist=True)
                current_output = select_folder(default_output, output, confined=confined)
                st.session_state["workspace"] = (current_data, current_output)
                st.session_state.pop("_catalogue", None)
                st.session_state.pop("_preview_cache", None)
                st.success("Folders updated.")
            except (OSError, ValueError) as exc:
                current_data, current_output = st.session_state.get("workspace", (default_data, default_output))
                st.error(str(exc))
        if confined:
            st.markdown("To use different folders **on your computer**, enter their paths below. Download the resulting `.env` file, put it beside `compose.yaml`, and run the command shown below to reconnect Docker to those folders.")
            with st.form("mount_form"):
                host_data = st.text_input("Host data folder", placeholder="C:/project/data or /home/user/data")
                host_results = st.text_input("Host results folder", placeholder="C:/project/results or /home/user/results")
                generate = st.form_submit_button("Prepare mount configuration")
            if generate:
                try:
                    st.session_state["mount_env"] = compose_environment(host_data, host_results)
                except ValueError as exc:
                    st.error(str(exc))
            if "mount_env" in st.session_state:
                st.download_button("Download .env", st.session_state["mount_env"], file_name=".env", on_click="ignore")
            st.code("docker compose up -d --force-recreate", language="bash")
            st.caption("Preserve SIV_UID/SIV_GID from an existing .env on Linux. No host folder is mounted automatically by the browser.")
    return current_data, current_output
