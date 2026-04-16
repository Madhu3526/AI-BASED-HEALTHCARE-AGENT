from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.hospital_config import HOSPITAL_CONFIGS
from dashboard.ui_core import render_hospital_page

st.set_page_config(page_title="Hospital A", page_icon="A", layout="wide", initial_sidebar_state="expanded")

render_hospital_page(HOSPITAL_CONFIGS["Hospital_A"])
