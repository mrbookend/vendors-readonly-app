# app_readonly.py
# Read-only Vendors view with:
# - Turso (libSQL) via SQLITE+LIBSQL URL, token from secrets; local SQLite fallback
# - Page layout from secrets (title, sidebar, max width)
# - Column labels and pixel widths from secrets (COLUMN_WIDTHS_PX_READONLY)
# - Quick filter (client-side, matches across all text columns)
# - Click-to-sort on every column (st.dataframe)
# - Wrapped cells; rows auto-grow in height
# - Optional sticky first column (CSS-based best effort)
# - Help section from top-level secrets (Markdown), with raw debug toggle

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine


# -----------------------------
# Early secrets + env helpers
# -----------------------------
def _read_secret_early(name: str, default=None):
    # Try Streamlit secrets first (safe if not available), then env var, else default
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def _get_secret_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return default


# -----------------------------
# Page layout (must be first)
# -----------------------------
def _apply_page_layout():
    page_title = _read_secret_early("page_title", "HCR Vendors (Read-Only)")
    sidebar_state = _read_secret_early("sidebar_state", "expanded")
    max_width = _read_secret_early("page_max_width_px", 2300)

    # Set page config
    st.set_page_config(page_title=page_title, layout="wide", initial_sidebar_state=sidebar_state)

    # Constrain main content width with CSS
    try:
        max_width = int(max_width)
    except Exception:
        max_width = 2300
    st.markdown(
        f"""
        <style>
        .block-container {{
            max-width: {max_width}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_page_layout()


# -----------------------------
# Secrets-driven display config
# -----------------------------
COLUMN_WIDTHS_PX_READONLY: Dict[str, int] = _read_secret_early("COLUMN_WIDTHS_PX_READONLY", {}) or {}

LABEL_OVERRIDES: Dict[str, str] = _read_secret_early("READONLY_COLUMN_LABELS", {}) or {}

READONLY_STICKY_FIRST_COL = _get_secret_bool(_read_secret_early("READONLY_STICKY_FIRST_COL", False), False)

# Help secrets (top-level preferred; fallback to overrides for backward-compat)
HELP_TITLE = (
    _read_secret_early("READONLY_HELP_TITLE")
    or LABEL_OVERRIDES.get("readonly_help_title")
    or "Providers Help / Tips"
)
HELP_MD = (
    _read_secret_early("READONLY_HELP_MD")
    or LABEL_OVERRIDES.get("readonly_help_md")
    or ""
)
HELP_DEBUG = _get_secret_bool(
    _read_secret_early("READONLY_HELP_DEBUG"),
    _get_secret_bool(LABEL_OVERRIDES.get("readonly_help_debug"), False),
)


# -----------------------------
# DB engine (Turso first; SQLite fallback)
# -----------------------------
def _make_engine() -> Engine:
    # Preferred: full URL in secrets (already includes ?secure=true for libsql)
    url = _read_secret_early(
        "TURSO_DATABASE_URL",
        "sqlite+libsql://vendors-prod-mrbookend.aws-us-west-2.turso.io?secure=true",
    )
    token = _read_secret_early("TURSO_AUTH_TOKEN", None)

    if url and url.startswith("sqlite+libsql://"):
        connect_args = {}
        if token:
            # SQLAlchemy libsql dialect supports "auth_token" in connect_args
            connect_args["auth_token"] = token
        try:
            eng = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
            # quick sanity ping
            with eng.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            return eng
        except Exception as e:
            st.warning(f"Turso connection failed ({e}). Falling back to local SQLite vendors.db.")

    # Fallback: local SQLite file bundled with the app
    local_path = "/mount/src/vendors-readonly-app/vendors.db"
    if not os.path.exists(local_path):
        # also try relative path (local dev)
        local_path = "vendors.db"
    eng = create_engine(f"sqlite:///{local_path}")
    return eng


engine = _make_engine()


# -----------------------------
# Data access
# -----------------------------
DISPLAY_COL_ORDER = [
    "id",
    "business_name",
    "category",
    "service",
    "contact_name",
    "phone",
    "address",
    "website",
    "notes",
    "keywords",
]


def load_vendors() -> pd.DataFrame:
    query = f"""
        SELECT {", ".join(DISPLAY_COL_ORDER)}
        FROM vendors
        ORDER BY category COLLATE NOCASE ASC,
                 business_name COLLATE NOCASE ASC,
                 id ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(sql_text(query), conn)
    # Ensure consistent dtypes for display
    for col in DISPLAY_COL_ORDER:
        if col not in df.columns:
            df[col] = ""
    # Normalize phone to string for consistent rendering
    if "phone" in df.columns:
        df["phone"] = df["phone"].astype(str).fillna("").replace("nan", "")
    # Website ensure string
    if "website" in df.columns:
        df["website"] = df["website"].astype(str).fillna("").replace("nan", "")
    return df[DISPLAY_COL_ORDER]


# -----------------------------
# UI helpers
# -----------------------------
def _apply_css_wrapping_and_widths(field_order: List[str], widths_px: Dict[str, int], sticky_first: bool):
    """Inject CSS that:
    - Wraps cell text
    - Sets per-column width hints in px (min/max width)
    - Optionally sticky first column
    Note: relies on current Streamlit HTML structure; harmless if it ever changes.
    """
    # Build nth-child selectors for widths (1-based, matching visible column order)
    width_css_rules = []
    for idx, field in enumerate(field_order, start=1):
        px = widths_px.get(field)
        if not px:
            continue
        width_css_rules.append(
            f'''
            /* Column {idx} ({field}) width */
            div[data-testid="stDataFrame"] table tbody tr td:nth-child({idx}),
            div[data-testid="stDataFrame"] table thead tr th:nth-child({idx}) {{
                min-width: {px}px !important;
                max-width: {px}px !important;
                width: {px}px !important;
                white-space: normal !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
            }}
            '''
        )
    width_css = "\n".join(width_css_rules)

    sticky_css = ""
    if sticky_first and field_order:
        # Best-effort sticky first column
        sticky_css = f"""
        /* Sticky first column */
        div[data-testid="stDataFrame"] table tbody tr td:nth-child(1),
        div[data-testid="stDataFrame"] table thead tr th:nth-child(1) {{
            position: sticky;
            left: 0;
            z-index: 2;  /* stay above other cells */
            background: var(--background-color, white);
            box-shadow: 1px 0 0 0 rgba(0,0,0,0.05);
        }}
        div[data-testid="stDataFrame"] table thead tr th:nth-child(1) {{
            z-index: 3;  /* above data cells */
        }}
        """

    st.markdown(
        f"""
        <style>
        /* Wrap all cells + headers */
        div[data-testid="stDataFrame"] table {{
            table-layout: fixed !important;
        }}
        div[data-testid="stDataFrame"] table td,
        div[data-testid="stDataFrame"] table th {{
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }}
        {width_css}
        {sticky_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _rename_columns_for_display(df: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    # Only rename keys that exist; leave others alone
    renames = {k: v for k, v in label_map.items() if k in df.columns and isinstance(v, str) and v}
    return df.rename(columns=renames)


def _build_quick_filter(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Browse Providers")
    q = st.text_input("Quick filter (matches any column)", value="", placeholder="e.g., plumb, roof, 78245…")
    q = q.strip()
    if not q:
        return df

    # Case-insensitive contains across all string columns
    mask = pd.Series([False] * len(df))
    lowered = df.copy()
    for col in lowered.columns:
        if pd.api.types.is_string_dtype(lowered[col]) or pd.api.types.is_object_dtype(lowered[col]):
            lowered[col] = lowered[col].fillna("").astype(str).str.lower()
            mask = mask | lowered[col].str.contains(q.lower(), na=False)
    return df[mask]


# -----------------------------
# Help section (no nested expanders)
# -----------------------------
def render_help_section():
    # If no help configured, show nothing (or a small info)
    if not HELP_MD:
        with st.expander(HELP_TITLE, expanded=False):
            st.info("No help content has been configured yet.")
        return

    with st.expander(HELP_TITLE, expanded=False):
        st.markdown(HELP_MD)
        if HELP_DEBUG:
            st.caption("Raw help (debug preview)")
            st.code(HELP_MD, language=None)


# -----------------------------
# Main app
# -----------------------------
def main():
    # Debug header (optional)
    with st.expander("Status & Secrets (debug)", expanded=False):
        # Backend info
        backend = "libsql" if str(_read_secret_early("TURSO_DATABASE_URL", "")).startswith("sqlite+libsql://") else "sqlite"
        dsn = _read_secret_early("TURSO_DATABASE_URL", "")
        auth = "token_set" if bool(_read_secret_early("TURSO_AUTH_TOKEN")) else "none"
        st.write("DB")
        st.code(
            {
                "backend": backend,
                "dsn": dsn,
                "auth": auth,
            }
        )

        # Secrets keys present
        keys = []
        try:
            keys = list(st.secrets.keys())
        except Exception:
            pass
        st.write("Secrets keys")
        st.code(keys)

        # Help presence
        st.write("Help MD:", "present" if bool(HELP_MD) else "(missing or empty)")

        # Raw widths and labels
        st.write("Raw COLUMN_WIDTHS_PX_READONLY (type)", type(COLUMN_WIDTHS_PX_READONLY).__name__)
        st.code(COLUMN_WIDTHS_PX_READONLY)
        st.write("Loaded column widths (effective)")
        st.code(COLUMN_WIDTHS_PX_READONLY)

        st.write("Sticky first col enabled:", READONLY_STICKY_FIRST_COL)

        st.write("Column label overrides (if any)")
        st.code(LABEL_OVERRIDES)

    # Load data
    df = load_vendors()

    # Apply widths & wrapping CSS (before rendering the dataframe)
    _apply_css_wrapping_and_widths(DISPLAY_COL_ORDER, COLUMN_WIDTHS_PX_READONLY, READONLY_STICKY_FIRST_COL)

    # Rename columns for display
    df_display = _rename_columns_for_display(df, LABEL_OVERRIDES)

    # Render Help section (safe single expander)
    render_help_section()

    # Quick filter (client-side)
    df_filtered = _build_quick_filter(df_display)

    # Linkify website if it looks like a URL (ensure startswith http)
    if "website" in df.columns:
        # We won't mutate df_display's column names here—website kept as 'website' internally,
        # but it might be renamed in df_display. Find the displayed name for website:
        website_display_name = LABEL_OVERRIDES.get("website", "website")
        if website_display_name in df_filtered.columns:
            def _ensure_http(u: str) -> str:
                u = (u or "").strip()
                if not u:
                    return ""
                if not re.match(r"(?i)^https?://", u):
                    return "http://" + u
                return u
            df_filtered[website_display_name] = df_filtered[website_display_name].apply(_ensure_http)

    # Show the table (click-to-sort is built-in)
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
