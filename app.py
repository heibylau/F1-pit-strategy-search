'''
Streamlit dashboard for the F1 pit stop strategy search.

Running this file automatically (re-)executes data_pipeline.ipynb (in the
background, on a scratch copy) and the Levin Tree Search, then renders the
result charts directly in the dashboard — nothing is written to images/.

    streamlit run app.py

Prerequisites: pip install -r requirements.txt
'''

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import run_search
from dashboard.plots import build_gapper_plot, build_stint_chart

ROOT      = Path(__file__).parent
NOTEBOOK  = ROOT / "data_pipeline.ipynb"
PARAM_DIR = ROOT / "data" / "parameter"

REQUIRED_PARAM_FILES = [
    "max_stint_lengths.csv", "per_lap_temperatures.csv",
    "tire_degradation_model.csv", "traffic_penalties.csv",
]


def notebook_artifacts_exist() -> bool:
    return all((PARAM_DIR / f).exists() for f in REQUIRED_PARAM_FILES)


def run_data_pipeline(log) -> None:
    '''
    Executes a scratch copy of the notebook so the user's own notebook file
    (which may have unsaved edits) is never touched or overwritten.
    '''
    with tempfile.NamedTemporaryFile(
        dir=ROOT, prefix=".dashboard_run_", suffix=".ipynb", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copyfile(NOTEBOOK, tmp_path)
        log.write("Running data_pipeline.ipynb (cached — no re-fetch if raw data already exists) ...")
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=1800",
                str(tmp_path),
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error(result.stderr[-4000:])
            raise RuntimeError("data_pipeline.ipynb failed — see log above.")
        log.write("data_pipeline.ipynb finished.")
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_data(force: bool = False) -> dict:
    if force or not notebook_artifacts_exist():
        with st.status("Preparing race data...", expanded=True) as status:
            run_data_pipeline(status)
            status.update(label="Race data ready.", state="complete", expanded=False)
    with st.spinner("Running Levin Tree Search..."):
        return run_search()


st.set_page_config(page_title="F1 Pit Strategy Search", layout="wide")
st.title("F1 Pit Stop Strategy Search — Levin Tree Search vs. Sainz")

with st.sidebar:
    st.header("Pipeline")
    st.caption(
        "On first load this runs data_pipeline.ipynb"
        "and the Levin Tree Search, "
        "then renders the results below."
    )
    force = st.button("Re-run pipeline now", use_container_width=True)

if "result" not in st.session_state or force:
    try:
        st.session_state["result"] = ensure_data(force=force)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

result = st.session_state["result"]
path_levin = result["path_levin"]
path_sainz = result["path_sainz"]

levin_total = result["cost"]
sainz_total = path_sainz[-1]["total_time"]
time_gap    = sainz_total - levin_total
levin_pits  = sum(1 for e in path_levin if e["action"] and e["action"].startswith("pit_"))
sainz_pits  = sum(1 for e in path_sainz if e["action"] and e["action"].startswith("pit_"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Levin total race time", f"{levin_total:.1f}s")
col2.metric("Sainz total race time", f"{sainz_total:.1f}s")
col3.metric("Time gained vs. Sainz", f"{time_gap:+.1f}s")
col4.metric("Levin pit stops", levin_pits, delta=f"{levin_pits - sainz_pits:+d} vs Sainz")

st.plotly_chart(build_stint_chart(path_levin, path_sainz), use_container_width=True)
st.plotly_chart(build_gapper_plot(path_levin, path_sainz, result["traffic_penalties"]),
                use_container_width=True)

with st.expander("Pruning threshold sweep"):
    st.dataframe(result["sweep"], use_container_width=True, hide_index=True)

with st.expander("Strategy detail"):
    tab1, tab2 = st.tabs(["Levin", "Sainz"])
    with tab1:
        st.dataframe(path_levin, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(path_sainz, use_container_width=True, hide_index=True)
