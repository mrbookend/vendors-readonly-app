# layout_header.py
# Centralized control of Streamlit page width, title, and sidebar state.
# Usage (put at TOP of each app file, before any other st.* call):
#   from layout_header import apply_layout
#   apply_layout()

from __future__ import annotations
import streamlit as st

def _as_int(val, default: int) -> int:
    try:
        return int(str(val))
    except Exception:
        return default

def _from_secrets(key: str, default):
    # st.secrets entries are optional and often strings
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def apply_layout() -> None:
    """
    Apply page-wide layout settings:
      - Wide layout
      - Sidebar initial state (collapsed/expanded/auto)
      - Page title
      - Max content width (px), default 3000, override via secrets
    Optional secrets.toml keys:
      page_title = "Vendors"
      page_max_width_px = "3000"
      sidebar_state = "collapsed"  # "collapsed" | "expanded" | "auto"
    """
    # 1) Configure page BEFORE any other Streamlit call
    page_title = str(_from_secrets("page_title", "Vendors"))
    sidebar_state = str(_from_secrets("sidebar_state", "collapsed"))
    if sidebar_state not in {"collapsed", "expanded", "auto"}:
        sidebar_state = "collapsed"

    st.set_page_config(
        page_title=page_title,
        layout="wide",
        initial_sidebar_state=sidebar_state,
    )

    # 2) Max width (cap). Defaults to 3000px if not provided in secrets.
    page_max_width_px = _as_int(_from_secrets("page_max_width_px", 3000), 3000)

    # 3) Inject CSS
    st.markdown(
        f"""
        <style>
          /* Central content container */
          .block-container {{
            max-width: {page_max_width_px}px;
            padding-left: 1rem;
            padding-right: 1rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
