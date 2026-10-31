"""Streamlit front end. Run: streamlit run app.py

On Streamlit Community Cloud, put API keys in the app's Secrets (GEMINI_API_KEY, GROQ_API_KEY).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception: 
    pass

from ai_report.data_loader import DataError  # noqa: E402
from ai_report.config import SETTINGS  # noqa: E402
from ai_report.llm import PROVIDERS, LLMError, get_client  # noqa: E402
from ai_report.pipeline import build_report  # noqa: E402
from ai_report.render import to_pdf  # noqa: E402

SAMPLE = Path(__file__).parent / "data" / "sample" / "orders.csv"

st.set_page_config(page_title="AI Data Report", layout="wide")
st.title("AI-written data report")
st.caption("Figures are calculated in code. The language model only writes the explanation, and every number it writes is checked.")

with st.sidebar:
    st.header("Settings")
    configured = SETTINGS.llm_provider if SETTINGS.llm_provider in PROVIDERS else "offline"
    provider = st.selectbox("Writer", list(PROVIDERS), index=list(PROVIDERS).index(configured))
    # One box per provider, so switching provider never sends one provider's model name to another.
    model = st.text_input(
        "Model (blank = default)",
        value=SETTINGS.llm_model if provider == configured else "",
        placeholder=PROVIDERS[provider].default_model or "model name",
        key=f"model_{provider}",
    )
    context = st.text_area("Business context (optional)", placeholder="e.g. Wholesale promotion ran all October.")
    upload = st.file_uploader("Data export (CSV or Excel)", type=["csv", "xlsx"])
    run = st.button("Build report", type="primary")

if run:
    tmp_path = None
    if upload:
        suffix = Path(upload.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getbuffer())
            data_path = tmp_path = tmp.name
    else:
        data_path = str(SAMPLE)
        st.info("No file uploaded: using the anonymised sample data.")

    try:
        with st.spinner("Calculating figures and writing the report..."):
            result = build_report(data_path, get_client(provider, model or None), context)
    except (DataError, LLMError) as e:
        st.error(str(e))
        st.stop()
    finally:
        if tmp_path:                    
            Path(tmp_path).unlink(missing_ok=True)

    n = result.narrative
    if n.fell_back:
        st.warning("The AI's text did not pass the accuracy checks, so the built-in rule-based writer was used. "
                   "The figures are unaffected.")
    elif n.attempts > 1:
        st.success(f"Written by {n.source}. The first draft had figures that did not match the data; "
                   f"it was corrected automatically and passed all checks on attempt {n.attempts}.")
    else:
        st.success(f"Written by {n.source}. Passed all accuracy checks on the first attempt.")
    if n.issues:
        with st.expander("Accuracy check details"):
            for issue in n.issues:
                st.caption(issue)

    with tempfile.TemporaryDirectory() as d:
        pdf_bytes = to_pdf(result.html, Path(d) / "report.pdf").read_bytes()
    c1, c2 = st.columns(2)
    c1.download_button("Download PDF", pdf_bytes, "report.pdf", "application/pdf")
    c2.download_button("Download HTML", result.html, "report.html", "text/html")

    st.iframe(result.html, height=2400)