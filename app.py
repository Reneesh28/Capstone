import streamlit as st
import os
import subprocess
import sys

st.title("FitTrack AI - Phase 1 Setup Checker")

# Check Python version
python_version = sys.version.split()[0]
st.write(f"**Python version:** {python_version}")

if python_version < "3.11":
    st.error("Python 3.11 or higher is recommended.")

# Check if .venv is active (heuristic)
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    st.success("Virtual environment is activated.")
else:
    st.warning("Virtual environment might NOT be activated.")

# Check required packages
required_packages = ["streamlit", "langchain", "chromadb", "faiss-cpu", "openai", "requests"]
st.subheader("Required Python packages")

missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
        st.write(f"✔️ {pkg} is installed")
    except ImportError:
        missing_packages.append(pkg)
        st.error(f"❌ {pkg} is NOT installed")

if missing_packages:
    st.warning("Run `pip install -r requirements.txt` to install missing packages.")

# Check folder structure
st.subheader("Project folders check")
required_folders = ["app", "services", "utils", "assets"]

for folder in required_folders:
    if os.path.isdir(folder):
        st.write(f"✔️ Folder `{folder}` exists")
    else:
        st.error(f"❌ Folder `{folder}` is MISSING")

# Check sample PDFs in assets
st.subheader("Sample PDFs in assets/")
assets_path = "assets"
pdf_files = []
if os.path.isdir(assets_path):
    pdf_files = [f for f in os.listdir(assets_path) if f.lower().endswith(".pdf")]
    if pdf_files:
        st.write(f"✔️ Found sample PDFs: {', '.join(pdf_files)}")
    else:
        st.warning("No PDF files found in assets/. Add sample PDFs for testing.")
else:
    st.error("assets/ folder is missing.")

# GitLab repo check (check remote URL)
st.subheader("GitLab repository remote URL")
try:
    remote_url = subprocess.check_output(["git", "remote", "get-url", "origin"]).decode().strip()
    st.write(f"✔️ Git remote origin URL: {remote_url}")
    if "gitlab.com" not in remote_url:
        st.warning("Remote URL does not appear to be a GitLab repository.")
except Exception as e:
    st.error(f"❌ Could not get Git remote URL: {e}")

st.markdown("---")
st.info("If any checks fail, fix them before moving to Phase 2.")

