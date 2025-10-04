# layout_header.py
# Single source of truth for page width. Import/apply at the very top of each app.

import streamlit as st

def _get(key: str, default: str) -> str:
    try:
        v = st.secrets.get(key, default)
    except Exception:
        v = default
    return str(v)

def apply_layout() -> None:
    # Must be the first Streamlit call
    st.set_page_config(
        page_title=_get("page_title", "Vendors"),
        layout="wide",
        initial_sidebar_state=_get("sidebar_state", "collapsed"),
    )

    # Enforce a hard cap with a selector that *always* wins
    page_max_width_px = _get("page_max_width_px", "3000")
    st.markdown(
        f"""
        <style>
          [data-testid="stAppViewContainer"] .main .block-container {{
            max-width: {page_max_width_px}px;
            padding-left: 1rem;
            padding-right: 1rem;
          }}
        </style>
        """,
        unsafe_allow_html=True
    )
