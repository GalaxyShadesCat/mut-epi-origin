"""Compatibility wrapper for the renamed state validation dashboard.

Preferred entrypoint:
streamlit run tools/state_validation_dashboard.py
"""

from __future__ import annotations

from tools.state_validation_dashboard import run_dashboard


if __name__ == "__main__":
    run_dashboard()
